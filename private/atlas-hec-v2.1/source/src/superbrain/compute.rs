//! 极致计算核心 - SIMD硬化 + 无分支 + 缓存预取

use super::*;
use core::arch::x86_64::*;

/// 神经元状态（缓存行对齐，批量处理）
#[repr(C, align(64))]
pub struct NeuronState {
    /// 膜电位（32路并行）
    pub v: [f32; 32],
    /// 恢复变量
    pub u: [f32; 32],
    /// 脉冲发放标志（位掩码，0/1）
    pub spikes: u32,
    /// 突触电流
    pub i_syn: [f32; 32],
}

impl NeuronState {
    /// LIF动力学更新（SIMD优化）
    /// 
    /// 公式：
    /// v = v * (1 - dt/tau) + i_syn * (dt/tau)
    /// if v > v_th: v = v_reset, spike = 1
    #[inline(always)]
    pub fn update_lif_simd(&mut self, dt: f32, tau: f32, v_th: f32, v_reset: f32) {
        unsafe {
            // 加载AVX寄存器（256-bit = 8x f32）
            let v_ptr = self.v.as_ptr();
            let i_ptr = self.i_syn.as_ptr();
            
            // 衰减因子：1 - dt/tau
            let decay = _mm256_set1_ps(1.0 - dt / tau);
            let v_th_vec = _mm256_set1_ps(v_th);
            let v_reset_vec = _mm256_set1_ps(v_reset);
            let zero = _mm256_setzero_ps();
            
            // 处理4个AVX向量（32个神经元）
            let mut new_spikes: u32 = 0;
            
            for i in 0..4 {
                let offset = i * 8;
                
                // 预取下一缓存行
                if i < 3 {
                    prefetch_hot(v_ptr.add(offset + 8));
                }
                
                // 加载v和i_syn
                let v_vec = _mm256_load_ps(v_ptr.add(offset));
                let i_vec = _mm256_load_ps(i_ptr.add(offset));
                
                // v = v * decay + i_syn * (1 - decay)
                let v_scaled = _mm256_mul_ps(v_vec, decay);
                let i_scaled = _mm256_mul_ps(i_vec, _mm256_set1_ps(dt / tau));
                let v_new = _mm256_add_ps(v_scaled, i_scaled);
                
                // 脉冲检测：v > v_th（生成掩码）
                let spike_mask = _mm256_cmp_ps(v_new, v_th_vec, _CMP_GT_OQ);
                let spike_int = _mm256_movemask_ps(spike_mask);
                
                // 设置新spike位
                new_spikes |= (spike_int as u32) << (i * 8);
                
                // v = spike ? v_reset : v_new
                let v_final = _mm256_blendv_ps(v_new, v_reset_vec, spike_mask);
                
                // 存储结果
                _mm256_store_ps(self.v.as_mut_ptr().add(offset), v_final);
                
                // 清零突触电流（消费掉）
                _mm256_store_ps(self.i_syn.as_mut_ptr().add(offset), zero);
            }
            
            self.spikes = new_spikes;
        }
    }
    
    /// Izhikevich模型更新（更生物真实）
    /// 
    /// 公式：
    /// v' = 0.04v² + 5v + 140 - u + I
    /// u' = a(bv - u)
    /// if v >= 30: v = c, u += d
    #[inline(always)]
    pub fn update_izhikevich_simd(&mut self, a: f32, b: f32, c: f32, d: f32, dt: f32) {
        unsafe {
            let v_ptr = self.v.as_ptr();
            let u_ptr = self.u.as_ptr();
            let i_ptr = self.i_syn.as_ptr();
            
            let a_vec = _mm256_set1_ps(a);
            let b_vec = _mm256_set1_ps(b);
            let c_vec = _mm256_set1_ps(c);
            let d_vec = _mm256_set1_ps(d);
            let dt_vec = _mm256_set1_ps(dt);
            let thirty = _mm256_set1_ps(30.0);
            let _140 = _mm256_set1_ps(140.0);
            let five = _mm256_set1_ps(5.0);
            let point04 = _mm256_set1_ps(0.04);
            
            let mut new_spikes: u32 = 0;
            
            for i in 0..4 {
                let offset = i * 8;
                
                let v = _mm256_load_ps(v_ptr.add(offset));
                let u = _mm256_load_ps(u_ptr.add(offset));
                let i_syn = _mm256_load_ps(i_ptr.add(offset));
                
                // v' = 0.04v² + 5v + 140 - u + I
                let v_sq = _mm256_mul_ps(v, v);
                let term1 = _mm256_mul_ps(point04, v_sq);
                let term2 = _mm256_mul_ps(five, v);
                let dv = _mm256_sub_ps(
                    _mm256_add_ps(_mm256_add_ps(term1, term2), _140),
                    _mm256_sub_ps(u, i_syn)
                );
                
                // u' = a(bv - u)
                let du = _mm256_mul_ps(a_vec, _mm256_sub_ps(_mm256_mul_ps(b_vec, v), u));
                
                // 更新
                let v_new = _mm256_add_ps(v, _mm256_mul_ps(dv, dt_vec));
                let u_new = _mm256_add_ps(u, _mm256_mul_ps(du, dt_vec));
                
                // 脉冲检测：v >= 30
                let spike_mask = _mm256_cmp_ps(v_new, thirty, _CMP_GE_OQ);
                let spike_int = _mm256_movemask_ps(spike_mask);
                new_spikes |= (spike_int as u32) << (i * 8);
                
                // 重置脉冲神经元：v = c, u += d
                let v_final = _mm256_blendv_ps(v_new, c_vec, spike_mask);
                let u_reset = _mm256_add_ps(u_new, d_vec);
                let u_final = _mm256_blendv_ps(u_new, u_reset, spike_mask);
                
                _mm256_store_ps(self.v.as_mut_ptr().add(offset), v_final);
                _mm256_store_ps(self.u.as_mut_ptr().add(offset), u_final);
                _mm256_store_ps(self.i_syn.as_mut_ptr().add(offset), _mm256_setzero_ps());
            }
            
            self.spikes = new_spikes;
        }
    }
}

/// SIMD突触电流累加（无分支）
/// 
/// 将输入spikes转换为突触后电流
#[inline(always)]
pub unsafe fn accumulate_synaptic_current(
    weights: &[f32],      // [n_post, n_pre]
    pre_spikes: u32,      // 位掩码（32个突触前神经元）
    post_current: &mut [f32], // [n_post]
    n_post: usize,
) {
    // 每个突触前神经元影响所有突触后神经元
    // 使用AVX并行处理4个突触后神经元
    
    let w_ptr = weights.as_ptr();
    let c_ptr = post_current.as_mut_ptr();
    
    let mut spike_mask = pre_spikes;
    
    for pre_idx in 0..32 {
        if spike_mask & 1 != 0 {
            // 该突触前神经元发放了
            let w_offset = pre_idx * n_post;
            
            // SIMD累加
            let mut post_idx = 0;
            while post_idx + 8 <= n_post {
                let w = _mm256_load_ps(w_ptr.add(w_offset + post_idx));
                let c = _mm256_load_ps(c_ptr.add(post_idx));
                let new_c = _mm256_add_ps(c, w);
                _mm256_store_ps(c_ptr.add(post_idx), new_c);
                post_idx += 8;
            }
            
            // 剩余部分（标量处理）
            while post_idx < n_post {
                *c_ptr.add(post_idx) += *w_ptr.add(w_offset + post_idx);
                post_idx += 1;
            }
        }
        spike_mask >>= 1;
    }
}

/// 8-bit量化矩阵乘法（INT8 + AVX2）
/// 
/// 公式：C = A @ B，其中A/B为8-bit，C为32-bit累加
#[inline(always)]
pub unsafe fn quantized_matmul_i8_avx2(
    a: &[i8],      // [m, k]
    b: &[i8],      // [k, n]
    c: &mut [i32], // [m, n]
    m: usize,
    n: usize,
    k: usize,
) {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();
    
    for i in 0..m {
        for j in (0..n).step_by(8) {
            // 累加器（8路并行）
            let mut acc = _mm256_setzero_si256();
            
            for l in 0..k {
                let a_val = *a_ptr.add(i * k + l) as i16;
                let a_vec = _mm256_set1_epi16(a_val);
                
                // 加载8个B元素
                let b_offset = l * n + j;
                let b_vec = _mm256_loadu_si256(b_ptr.add(b_offset) as *const __m256i);
                
                // 扩展到16-bit并相乘
                let b_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 0));
                let b_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 1));
                
                // 累加（低8位和高8位分开）
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_vec, b_low));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_vec, b_high));
            }
            
            // 存储结果
            _mm256_storeu_si256(c_ptr.add(i * n + j) as *mut __m256i, acc);
        }
    }
}

/// STDP学习规则（在线突触可塑性）
/// 
/// 脉冲时间依赖可塑性（Spike-Timing-Dependent Plasticity）
#[repr(C)]
pub struct STDPParams {
    pub a_plus: f32,   // LTP幅度
    pub a_minus: f32,  // LTD幅度
    pub tau_plus: f32, // LTP时间常数
    pub tau_minus: f32,// LTD时间常数
}

impl Default for STDPParams {
    fn default() -> Self {
        Self {
            a_plus: 0.1,
            a_minus: -0.105,
            tau_plus: 20.0,
            tau_minus: 20.0,
        }
    }
}

/// 应用STDP更新权重
#[inline(always)]
pub fn apply_stdp(
    weights: &mut [f32],
    pre_spikes: u32,
    post_spikes: u32,
    pre_times: &[u64], // 上次脉冲时间
    post_times: &[u64],
    current_time: u64,
    params: &STDPParams,
) {
    let dt_pre = |idx: usize| current_time.saturating_sub(pre_times[idx]) as f32;
    let dt_post = |idx: usize| current_time.saturating_sub(post_times[idx]) as f32;
    
    // 遍历所有突触后发放
    for post_idx in 0..32 {
        if (post_spikes >> post_idx) & 1 == 0 {
            continue;
        }
        
        // 遍历所有突触前发放
        for pre_idx in 0..32 {
            if (pre_spikes >> pre_idx) & 1 == 0 {
                continue;
            }
            
            let w_idx = pre_idx * 32 + post_idx;
            
            // LTD：突触前在突触后之前发放
            let dt = dt_pre(pre_idx);
            if dt > 0.0 && dt < 100.0 {
                let dw = params.a_minus * (-dt / params.tau_minus).exp();
                weights[w_idx] += dw;
            }
            
            // LTP：突触后在突触前之前发放
            let dt = dt_post(post_idx);
            if dt > 0.0 && dt < 100.0 {
                let dw = params.a_plus * (-dt / params.tau_plus).exp();
                weights[w_idx] += dw;
            }
            
            // 权重限制
            weights[w_idx] = weights[w_idx].clamp(-1.0, 1.0);
        }
    }
}

/// 批量神经元更新（缓存友好布局）
#[repr(C, align(4096))] // 页对齐
pub struct NeuronBatch {
    /// 状态数组（SOA布局）
    pub v: AlignedVec<f32, 4096>,
    pub u: AlignedVec<f32, 4096>,
    pub i_syn: AlignedVec<f32, 4096>,
    /// 脉冲记录（位掩码数组）
    pub spikes: AlignedVec<u32, 4096>,
}

/// 对齐向量（编译期确定对齐）
#[repr(C, align(4096))]
pub struct AlignedVec<T, const ALIGN: usize> {
    data: Vec<T>,
}

impl<T: Default + Clone, const ALIGN: usize> AlignedVec<T, ALIGN> {
    pub fn new(size: usize) -> Self {
        let mut v = vec![T::default(); size];
        // 确保对齐（实际Vec通常已对齐到页大小）
        assert_eq!(v.as_ptr() as usize % ALIGN, 0);
        Self { data: v }
    }
    
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] { &self.data }
    
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] { &mut self.data }
    
    #[inline(always)]
    pub fn len(&self) -> usize { self.data.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lif_simd() {
        let mut state = NeuronState {
            v: [0.0; 32],
            u: [0.0; 32],
            spikes: 0,
            i_syn: [1.0; 32], // 注入电流
        };
        
        state.update_lif_simd(0.1, 20.0, 1.0, 0.0);
        
        // 检查是否有spikes（取决于输入电流）
        println!("Spikes: {:032b}", state.spikes);
    }
    
    #[test]
    fn test_izhikevich_simd() {
        let mut state = NeuronState {
            v: [-70.0; 32],
            u: [-14.0; 32],
            spikes: 0,
            i_syn: [10.0; 32],
        };
        
        // Regular spiking参数
        state.update_izhikevich_simd(0.02, 0.2, -65.0, 8.0, 0.1);
        
        println!("Spikes: {:032b}", state.spikes);
    }
    
    #[test]
    fn test_stdp() {
        let mut weights = vec![0.5; 32 * 32];
        let mut pre_times = vec![0u64; 32];
        let mut post_times = vec![0u64; 32];
        
        // 模拟pre在t=10发放
        pre_times[0] = 10;
        // 模拟post在t=20发放（pre在post之前 → LTD）
        post_times[0] = 20;
        
        let params = STDPParams::default();
        apply_stdp(
            &mut weights,
            1, // pre_spikes
            1, // post_spikes
            &pre_times,
            &post_times,
            20, // current_time
            &params,
        );
        
        // 权重应该减小（LTD）
        assert!(weights[0] < 0.5);
    }
}
