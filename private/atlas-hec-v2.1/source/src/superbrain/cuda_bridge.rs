//! Rust-CUDA Bridge - 零拷贝极致接口
//!
//! 优化目标：
//! - Unified Memory：零PCIe传输
//! - CUDA Graphs：<1μs kernel launch
//! - Async操作：CPU-GPU流水线
//! - 硬实时：10ms timestep保证

use super::*;
use cust::prelude::*;
use cust::memory::{DeviceBuffer, UnifiedBuffer};
use std::sync::Arc;

/// CUDA Superbrain上下文（线程安全）
pub struct CudaSuperbrainContext {
    device: Device,
    context: Context,
    stream: Stream,
    
    // CUDA Graphs（预捕获执行序列）
    graph_exec: cuda_sys::cudaGraphExec_t,
    
    // Unified Memory（零拷贝）
    neurons: UnifiedMemory<NeuronStateGPU>,
    synapses: UnifiedMemory<CsrSynapsesGPU>,
    queue: UnifiedMemory<SpikeQueueGPU>,
    stats: UnifiedMemory<RuntimeStatsGPU>,
    
    // 设备端函数
    module: Module,
    
    // 运行时参数
    n_neurons: u32,
    timestep_ms: f32,
    current_time: u32,
}

/// GPU端神经元状态（镜像atlas_superbrain.cuh）
#[repr(C, align(64))]
pub struct NeuronStateGPU {
    pub v_mem: *mut f32,
    pub u_mem: *mut f32,
    pub i_syn: *mut f32,
    pub last_spike: *mut u32,
    pub refractory: *mut u8,
}

/// GPU端CSR突触
#[repr(C, align(4096))]
pub struct CsrSynapsesGPU {
    pub row_ptr: *mut u32,
    pub col_idx: *mut u32,
    pub weight: *mut half::f16,  // FP16
    pub delay: *mut u8,
    pub r_row_ptr: *mut u32,
    pub r_col_idx: *mut u32,
    pub n_edges: u64,
}

/// GPU端脉冲队列
#[repr(C, align(64))]
pub struct SpikeQueueGPU {
    pub buffer: *mut u32,
    pub head: u32,
    pub tail: u32,
    pub capacity: u32,
    pub _pad: u32,
}

/// GPU端统计
#[repr(C, align(64))]
pub struct RuntimeStatsGPU {
    pub n_spikes: u64,
    pub n_synaptic_events: u64,
    pub n_stdp_updates: u64,
    pub avg_firing_rate: f32,
    pub queue_overflows: u32,
}

/// Unified Memory包装（零拷贝）
pub struct UnifiedMemory<T> {
    ptr: *mut T,
    size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> UnifiedMemory<T> {
    /// 分配Unified Memory（CPU和GPU同时可见）
    pub fn new(size: usize) -> Result<Self, CudaError> {
        let mut ptr: *mut T = std::ptr::null_mut();
        let bytes = size * std::mem::size_of::<T>();
        
        unsafe {
            // cudaMallocManaged
            let err = cuda_sys::cudaMallocManaged(
                &mut ptr as *mut *mut T as *mut *mut c_void,
                bytes,
                cuda_sys::cudaMemAttachGlobal,
            );
            
            if err != cuda_sys::cudaError::cudaSuccess {
                return Err(CudaError::from(err));
            }
            
            // 预取到GPU（建议，非强制）
            cuda_sys::cudaMemPrefetchAsync(ptr, bytes, 0, std::ptr::null_mut());
        }
        
        Ok(Self {
            ptr,
            size,
            _phantom: std::marker::PhantomData,
        })
    }
    
    /// 获取指针
    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }
    
    /// 获取可变引用（CPU访问）
    pub unsafe fn as_mut(&mut self) -> &mut T {
        &mut *self.ptr
    }
    
    /// 预取到GPU（异步）
    pub fn prefetch_to_gpu(&self, stream: &Stream) {
        unsafe {
            cuda_sys::cudaMemPrefetchAsync(
                self.ptr,
                self.size * std::mem::size_of::<T>(),
                0,  // device 0
                stream.as_inner() as *mut _,
            );
        }
    }
    
    /// 预取到CPU（异步）
    pub fn prefetch_to_cpu(&self, stream: &Stream) {
        unsafe {
            cuda_sys::cudaMemPrefetchAsync(
                self.ptr,
                self.size * std::mem::size_of::<T>(),
                cuda_sys::cudaCpuDeviceId,
                stream.as_inner() as *mut _,
            );
        }
    }
}

impl<T> Drop for UnifiedMemory<T> {
    fn drop(&mut self) {
        unsafe {
            cuda_sys::cudaFree(self.ptr);
        }
    }
}

unsafe impl<T> Send for UnifiedMemory<T> {}
unsafe impl<T> Sync for UnifiedMemory<T> {}

impl CudaSuperbrainContext {
    /// 初始化CUDA超脑上下文
    pub fn new(n_neurons: u32, n_synapses: u64) -> Result<Self, Box<dyn std::error::Error>> {
        // 初始化CUDA
        cust::init(cust::CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let context = Context::new(device)?;
        
        sb_log!("[CudaSuperbrain] 初始化设备: {}", device.name()?);
        sb_log!("[CudaSuperbrain] 神经元: {}, 突触: {}", n_neurons, n_synapses);
        
        // 分配Unified Memory
        let neurons = UnifiedMemory::new(n_neurons as usize)?;
        let synapses = UnifiedMemory::new(1)?;  // 结构体本身很小，数据在里面
        let queue = UnifiedMemory::new(1)?;
        let stats = UnifiedMemory::new(1)?;
        
        // 分配内部数组（Unified Memory）
        let v_mem = UnifiedMemory::<f32>::new(n_neurons as usize)?;
        let u_mem = UnifiedMemory::<f32>::new(n_neurons as usize)?;
        let i_syn = UnifiedMemory::<f32>::new(n_neurons as usize)?;
        let last_spike = UnifiedMemory::<u32>::new(n_neurons as usize)?;
        let refractory = UnifiedMemory::<u8>::new(n_neurons as usize)?;
        
        // 填充NeuronState结构
        unsafe {
            let state = &mut *(neurons.as_ptr() as *mut NeuronStateGPU);
            state.v_mem = v_mem.as_ptr();
            state.u_mem = u_mem.as_ptr();
            state.i_syn = i_syn.as_ptr();
            state.last_spike = last_spike.as_ptr();
            state.refractory = refractory.as_ptr();
        }
        
        // 加载PTX模块
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/atlas_superbrain.ptx"));
        let module = Module::from_ptx(ptx, &[])?;
        
        // 创建非阻塞流
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        // 初始化神经元（Kernel调用）
        let init_kernel = module.get_function("kernel_init_neurons")?;
        let block_size = 256;
        let grid_size = (n_neurons + block_size - 1) / block_size;
        
        unsafe {
            launch!(init_kernel<<<grid_size, block_size, 0, stream>>>(
                *(neurons.as_ptr() as *const NeuronStateGPU),
                n_neurons
            ))?;
        }
        
        // 初始化队列
        let init_queue = module.get_function("kernel_init_queue")?;
        unsafe {
            launch!(init_queue<<<1, 1, 0, stream>>>(queue.as_ptr()))?;
        }
        
        stream.synchronize()?;
        
        sb_log!("[CudaSuperbrain] Unified Memory分配完成");
        
        // TODO: 捕获CUDA Graphs
        let graph_exec = std::ptr::null_mut();  // 占位
        
        Ok(Self {
            device,
            context,
            stream,
            graph_exec,
            neurons,
            synapses,
            queue,
            stats,
            module,
            n_neurons,
            timestep_ms: 0.1,  // 0.1ms = 100μs
            current_time: 0,
        })
    }
    
    /// 执行一个timestep（硬实时关键）
    pub fn step(&mut self) -> Result<(), CudaError> {
        self.current_time += 1;
        
        // 方法1: 直接kernel launch（灵活但慢）
        // self.launch_kernels_direct()?;
        
        // 方法2: CUDA Graphs执行（快但静态）
        // self.launch_graph()?;
        
        // 简化：直接launch（Graphs在开发阶段）
        let block_size = 256;
        let grid_size = (self.n_neurons + block_size - 1) / block_size;
        
        // Kernel 1: Izhikevich更新
        let izh_kernel = self.module.get_function("kernel_izhikevich_update")?;
        unsafe {
            launch!(izh_kernel<<<grid_size, block_size, 0, &self.stream>>>(
                *(self.neurons.as_ptr() as *const NeuronStateGPU),
                self.queue.as_ptr(),
                self.stats.as_ptr(),
                self.timestep_ms,
                self.current_time
            ))?;
        }
        
        // 同步（硬实时必须）
        self.stream.synchronize()?;
        
        Ok(())
    }
    
    /// 获取脉冲统计（零拷贝，直接读Unified Memory）
    pub fn get_spike_count(&self) -> u32 {
        unsafe {
            let queue = &*(self.queue.as_ptr() as *const SpikeQueueGPU);
            (queue.head - queue.tail) & ((1 << 20) - 1)  // QUEUE_MASK
        }
    }
    
    /// 获取运行时统计
    pub fn get_stats(&self) -> RuntimeStatsSnapshot {
        unsafe {
            let stats = &*(self.stats.as_ptr() as *const RuntimeStatsGPU);
            RuntimeStatsSnapshot {
                n_spikes: stats.n_spikes,
                n_synaptic_events: stats.n_synaptic_events,
                n_stdp_updates: stats.n_stdp_updates,
            }
        }
    }
}

/// 运行时统计快照
#[derive(Debug, Clone, Copy)]
pub struct RuntimeStatsSnapshot {
    pub n_spikes: u64,
    pub n_synaptic_events: u64,
    pub n_stdp_updates: u64,
}

use std::os::raw::c_void;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_unified_memory() {
        let mem = UnifiedMemory::<f32>::new(1000).unwrap();
        assert!(!mem.as_ptr().is_null());
    }
    
    // 注意：CUDA测试需要实际GPU
    // #[test]
    // fn test_cuda_context() {
    //     let ctx = CudaSuperbrainContext::new(1000, 10000).unwrap();
    //     ctx.step().unwrap();
    // }
}
