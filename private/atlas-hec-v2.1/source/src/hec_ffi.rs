//! Atlas-HEC v2.1 Heterogeneous FFI Module
//! Rust <-> C++ Bridge for CPU+GPU+RAM协同

use std::os::raw::{c_int, c_char, c_void};
use std::ffi::CString;

#[link(name = "hec_bridge")]
extern "C" {
    fn hec_init(n_neurons: usize, n_synapses: usize, gpu_id: c_int) -> c_int;
    fn hec_step_hybrid(sensory: *const u8, motor: *mut f32, dt: f32) -> c_int;
    fn hec_stdp_async(reward: f32) -> c_int;
    fn hec_cleanup() -> c_int;
    fn hec_status(buf: *mut c_char, bufsize: usize) -> c_int;
}

/// Heterogeneous Atlas Superbrain
pub struct HecSuperbrain {
    num_neurons: usize,
    num_synapses: usize,
    gpu_id: i32,
    initialized: bool,
}

impl HecSuperbrain {
    pub fn new(neurons: usize, synapses: usize, gpu_id: i32) -> Result<Self, String> {
        let ret = unsafe { hec_init(neurons, synapses, gpu_id) };
        if ret != 0 {
            return Err(format!("HEC初始化失败: {}", ret));
        }
        
        Ok(HecSuperbrain {
            num_neurons: neurons,
            num_synapses: synapses,
            gpu_id,
            initialized: true,
        })
    }
    
    /// 异构步进: CPU编码 → GPU计算 → CPU解码
    pub fn step(&self, sensory: &[u8; 256]) -> Result<[f32; 5], String> {
        let mut motor = [0.0f32; 5];
        
        let ret = unsafe {
            hec_step_hybrid(sensory.as_ptr(), motor.as_mut_ptr(), 0.1)
        };
        
        if ret != 0 {
            return Err(format!("HEC步进失败: {}", ret));
        }
        
        Ok(motor)
    }
    
    /// 异步STDP更新（不阻塞CPU）
    pub fn stdp_async(&self, reward: f32) -> Result<(), String> {
        let ret = unsafe { hec_stdp_async(reward) };
        if ret != 0 {
            return Err(format!("STDP失败: {}", ret));
        }
        Ok(())
    }
    
    /// 获取状态
    pub fn status(&self) -> String {
        let mut buf = vec![0u8; 256];
        let ret = unsafe {
            hec_status(buf.as_mut_ptr() as *mut c_char, buf.len())
        };
        
        if ret == 0 {
            CString::new(buf).ok()
                .and_then(|c| c.into_string().ok())
                .unwrap_or_else(|| "无效状态".to_string())
        } else {
            "未初始化".to_string()
        }
    }
}

impl Drop for HecSuperbrain {
    fn drop(&mut self) {
        if self.initialized {
            unsafe { hec_cleanup(); }
        }
    }
}
