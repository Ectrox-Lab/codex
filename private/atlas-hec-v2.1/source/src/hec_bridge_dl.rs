//! Atlas-HEC v2.1 Bridge - 动态加载版本
use std::os::raw::{c_int, c_char};
use libloading::{Library, Symbol};
use std::ffi::CString;

pub struct HecBridge {
    lib: Library,
    n_neurons: usize,
}

impl HecBridge {
    pub fn new(neurons: usize, synapses: usize, gpu_id: i32) -> Result<Self, String> {
        // 尝试加载共享库
        let lib = unsafe {
            Library::new("./hetero_bridge/libhec_bridge_v2.so")
                .or_else(|_| Library::new("../hetero_bridge/libhec_bridge_v2.so"))
                .or_else(|_| Library::new("libhec_bridge_v2.so"))
                .map_err(|e| format!("无法加载库: {}", e))?
        };
        
        // 获取hec_init函数
        let init: Symbol<unsafe extern "C" fn(usize, usize, c_int) -> c_int> = unsafe {
            lib.get(b"hec_init")
                .map_err(|e| format!("无法找到hec_init: {}", e))?
        };
        
        let ret = unsafe { init(neurons, synapses, gpu_id) };
        if ret != 0 {
            return Err(format!("hec_init失败: {}", ret));
        }
        
        Ok(HecBridge { lib, n_neurons: neurons })
    }
    
    pub fn step(&self, sensory: &[u8; 256]) -> Result<[f32; 5], String> {
        let step_fn: Symbol<unsafe extern "C" fn(*const u8, *mut f32, usize, usize) -> c_int> = unsafe {
            self.lib.get(b"hec_step_hybrid")
                .map_err(|e| format!("无法找到hec_step_hybrid: {}", e))?
        };
        
        let mut motor = [0.0f32; 5];
        let ret = unsafe {
            step_fn(sensory.as_ptr(), motor.as_mut_ptr(), 256, 5)
        };
        
        if ret != 0 {
            return Err(format!("hec_step_hybrid失败: {}", ret));
        }
        
        Ok(motor)
    }
    
    pub fn stdp(&self, reward: f32) -> Result<(), String> {
        let stdp_fn: Symbol<unsafe extern "C" fn(f32) -> c_int> = unsafe {
            self.lib.get(b"hec_stdp_async")
                .map_err(|e| format!("无法找到hec_stdp_async: {}", e))?
        };
        
        let ret = unsafe { stdp_fn(reward) };
        if ret != 0 {
            return Err(format!("hec_stdp_async失败: {}", ret));
        }
        
        Ok(())
    }
    
    pub fn status(&self) -> String {
        let status_fn: Symbol<unsafe extern "C" fn(*mut c_char, usize) -> c_int> = unsafe {
            match self.lib.get(b"hec_status") {
                Ok(f) => f,
                Err(_) => return "无法获取状态".to_string(),
            }
        };
        
        let mut buf = vec![0u8; 256];
        let ret = unsafe {
            status_fn(buf.as_mut_ptr() as *mut c_char, buf.len())
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

impl Drop for HecBridge {
    fn drop(&mut self) {
        if let Ok(cleanup) = unsafe { self.lib.get::<unsafe extern "C" fn() -> c_int>(b"hec_cleanup") } {
            unsafe { cleanup(); }
        }
    }
}
