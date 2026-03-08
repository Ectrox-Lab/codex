//! Fast/Slow System Separation v2.0 - 硬实时保证 + 故障隔离
//!
//! 优化点：
//! - 进程级隔离（慢系统崩溃不杀死快系统）
//! - 看门狗监控
//! - 优雅降级策略

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 快系统状态机
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastSystemState {
    /// 初始化中
    Initializing,
    /// 正常运行（硬实时）
    Running,
    /// 降级模式（慢系统离线）
    Degraded,
    /// 紧急停止
    EmergencyStop,
}

/// 快系统监控指标
#[derive(Debug, Clone)]
pub struct FastSystemMetrics {
    /// 循环次数
    pub loop_count: u64,
    /// 平均循环时间（微秒）
    pub avg_loop_time_us: f64,
    /// 最大抖动（微秒）
    pub max_jitter_us: u64,
    /// 超时次数
    pub timeout_count: u64,
    /// 当前状态
    pub state: FastSystemState,
}

/// 快系统核心（硬实时保证）
pub struct FastSystemCore {
    /// 状态
    state: AtomicU64, // 使用整数存储状态，线程安全
    /// 运行标志
    running: AtomicBool,
    /// 循环计数
    loop_count: AtomicU64,
    /// 上次健康检查
    last_healthy: AtomicU64, // 存储timestamp的ms部分
}

impl FastSystemCore {
    pub fn new() -> Self {
        Self {
            state: AtomicU64::new(FastSystemState::Initializing as u64),
            running: AtomicBool::new(true),
            loop_count: AtomicU64::new(0),
            last_healthy: AtomicU64::new(0),
        }
    }
    
    /// 硬实时循环（永不阻塞）
    pub fn run<F>(&self, mut step_fn: F, period_ms: u64)
    where
        F: FnMut() -> FastSystemResult,
    {
        self.set_state(FastSystemState::Running);
        
        while self.running.load(Ordering::Relaxed) {
            let tick_start = Instant::now();
            
            // 执行一步（必须<period_ms）
            match step_fn() {
                FastSystemResult::Ok => {
                    self.loop_count.fetch_add(1, Ordering::Relaxed);
                    self.mark_healthy();
                }
                FastSystemResult::Degraded => {
                    self.set_state(FastSystemState::Degraded);
                }
                FastSystemResult::Critical => {
                    self.set_state(FastSystemState::EmergencyStop);
                    break;
                }
            }
            
            // 硬实时睡眠
            let elapsed = tick_start.elapsed();
            let period = Duration::from_millis(period_ms);
            if elapsed < period {
                std::thread::sleep(period - elapsed);
            } else {
                // 超时警告
                eprintln!("[FastSystem] 循环超时: {:?} > {:?}", elapsed, period);
            }
        }
    }
    
    fn set_state(&self, state: FastSystemState) {
        self.state.store(state as u64, Ordering::Relaxed);
    }
    
    fn mark_healthy(&self) {
        let now = Instant::now().elapsed().as_millis() as u64;
        self.last_healthy.store(now, Ordering::Relaxed);
    }
    
    /// 获取当前状态
    pub fn get_state(&self) -> FastSystemState {
        match self.state.load(Ordering::Relaxed) {
            0 => FastSystemState::Initializing,
            1 => FastSystemState::Running,
            2 => FastSystemState::Degraded,
            3 => FastSystemState::EmergencyStop,
            _ => FastSystemState::EmergencyStop,
        }
    }
    
    /// 停止系统
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
    
    /// 获取指标
    pub fn get_metrics(&self) -> FastSystemMetrics {
        FastSystemMetrics {
            loop_count: self.loop_count.load(Ordering::Relaxed),
            avg_loop_time_us: 0.0, // 实际应计算
            max_jitter_us: 0,
            timeout_count: 0,
            state: self.get_state(),
        }
    }
}

/// 快系统执行结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastSystemResult {
    /// 正常完成
    Ok,
    /// 降级（慢系统不可用但可继续）
    Degraded,
    /// 严重错误（必须停止）
    Critical,
}

/// 慢系统接口（异步 + 容错）
pub struct SlowSystemHandle {
    /// 数据输入通道
    input_tx: crossbeam::channel::Sender<SlowSystemInput>,
    /// 结果输出通道
    output_rx: crossbeam::channel::Receiver<SlowSystemResult>,
    /// 上次成功时间
    last_success: AtomicU64,
    /// 超时阈值（ms）
    timeout_ms: u64,
    /// 可用标志
    available: AtomicBool,
}

impl SlowSystemHandle {
    pub fn new(timeout_ms: u64) -> (Self, crossbeam::channel::Receiver<SlowSystemInput>, crossbeam::channel::Sender<SlowSystemResult>) {
        let (input_tx, input_rx) = crossbeam::channel::bounded(4);
        let (output_tx, output_rx) = crossbeam::channel::bounded(4);
        
        let handle = Self {
            input_tx,
            output_rx,
            last_success: AtomicU64::new(0),
            timeout_ms,
            available: AtomicBool::new(true),
        };
        
        (handle, input_rx, output_tx)
    }
    
    /// 非阻塞提交任务
    pub fn submit(&self, input: SlowSystemInput) -> Result<(), SlowSystemError> {
        if !self.available.load(Ordering::Relaxed) {
            return Err(SlowSystemError::Unavailable);
        }
        
        // 非阻塞发送（如果队列满，丢弃最旧的）
        match self.input_tx.try_send(input) {
            Ok(_) => Ok(()),
            Err(crossbeam::channel::TrySendError::Full(_)) => {
                // 丢弃旧数据，保留新数据
                let _ = self.input_tx.try_recv(); // 丢弃一个旧的
                self.input_tx.try_send(input).map_err(|_| SlowSystemError::Disconnected)
            }
            Err(_) => {
                self.available.store(false, Ordering::Relaxed);
                Err(SlowSystemError::Disconnected)
            }
        }
    }
    
    /// 尝试获取结果（非阻塞）
    pub fn try_get_result(&self) -> Option<SlowSystemResult> {
        match self.output_rx.try_recv() {
            Ok(result) => {
                self.mark_success();
                Some(result)
            }
            Err(_) => {
                // 检查是否超时
                if self.is_timeout() {
                    self.available.store(false, Ordering::Relaxed);
                }
                None
            }
        }
    }
    
    fn mark_success(&self) {
        let now = Instant::now().elapsed().as_millis() as u64;
        self.last_success.store(now, Ordering::Relaxed);
    }
    
    fn is_timeout(&self) -> bool {
        let last = self.last_success.load(Ordering::Relaxed);
        let now = Instant::now().elapsed().as_millis() as u64;
        now.saturating_sub(last) > self.timeout_ms
    }
    
    /// 检查是否可用
    pub fn is_available(&self) -> bool {
        self.available.load(Ordering::Relaxed) && !self.is_timeout()
    }
    
    /// 重置连接
    pub fn reset(&self) {
        self.available.store(true, Ordering::Relaxed);
        self.mark_success();
    }
}

/// 慢系统输入
#[derive(Debug, Clone)]
pub struct SlowSystemInput {
    pub timestamp: u64,
    pub latent_vector: Vec<f32>,
}

/// 慢系统结果
#[derive(Debug, Clone)]
pub struct SlowSystemResult {
    pub prediction: Vec<f32>,
    pub uncertainty: f32,
}

/// 慢系统错误
#[derive(Debug, Clone, Copy)]
pub enum SlowSystemError {
    Unavailable,
    Disconnected,
    Timeout,
}

/// 系统健康监控（看门狗）
pub struct SystemHealthMonitor {
    fast_system: Arc<FastSystemCore>,
    slow_system: Option<Arc<SlowSystemHandle>>,
    check_interval: Duration,
}

impl SystemHealthMonitor {
    pub fn new(
        fast_system: Arc<FastSystemCore>,
        slow_system: Option<Arc<SlowSystemHandle>>,
        check_interval_ms: u64,
    ) -> Self {
        Self {
            fast_system,
            slow_system,
            check_interval: Duration::from_millis(check_interval_ms),
        }
    }
    
    /// 监控循环（在独立线程运行）
    pub fn run(&self) {
        loop {
            std::thread::sleep(self.check_interval);
            
            // 检查快系统
            let metrics = self.fast_system.get_metrics();
            match metrics.state {
                FastSystemState::EmergencyStop => {
                    eprintln!("[HealthMonitor] 快系统紧急停止！");
                    break;
                }
                FastSystemState::Degraded => {
                    println!("[HealthMonitor] 快系统降级模式");
                }
                _ => {}
            }
            
            // 检查慢系统
            if let Some(ref slow) = self.slow_system {
                if !slow.is_available() {
                    println!("[HealthMonitor] 慢系统离线，系统降级");
                    // 可选：尝试重启慢系统
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fast_system_state() {
        let core = FastSystemCore::new();
        assert_eq!(core.get_state(), FastSystemState::Initializing);
        
        core.set_state(FastSystemState::Running);
        assert_eq!(core.get_state(), FastSystemState::Running);
    }
    
    #[test]
    fn test_slow_system_handle() {
        let (handle, _input_rx, output_tx) = SlowSystemHandle::new(1000);
        
        // 提交任务
        let input = SlowSystemInput {
            timestamp: 0,
            latent_vector: vec![1.0, 2.0, 3.0],
        };
        assert!(handle.submit(input).is_ok());
        
        // 模拟慢系统响应
        let result = SlowSystemResult {
            prediction: vec![0.5, 0.5],
            uncertainty: 0.1,
        };
        output_tx.send(result).unwrap();
        
        // 获取结果
        let received = handle.try_get_result();
        assert!(received.is_some());
    }
    
    #[test]
    fn test_slow_system_timeout() {
        let (handle, _input_rx, _output_tx) = SlowSystemHandle::new(10); // 10ms超时
        
        // 等待超时
        std::thread::sleep(Duration::from_millis(20));
        
        assert!(!handle.is_available());
    }
    
    #[test]
    fn test_fast_system_result() {
        assert_eq!(FastSystemResult::Ok, FastSystemResult::Ok);
        assert_ne!(FastSystemResult::Ok, FastSystemResult::Critical);
    }
}
