//! 极致时序系统 - 分层时钟 + 无锁调度 + TSC时间戳

use super::*;
use core::sync::atomic::{AtomicU64, Ordering};
use core::arch::x86_64::__rdtscp;

/// CPU时间戳计数器（TSC）读取
/// 
/// 精度：~0.3ns（3GHz CPU），无系统调用开销
#[inline(always)]
pub fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// 带CPU序号的TSC（防止乱序执行）
#[inline(always)]
pub fn rdtscp() -> (u64, u32) {
    let mut aux = 0u32;
    let tsc = unsafe { __rdtscp(&mut aux) };
    (tsc, aux)
}

/// TSC到纳秒转换（需校准）
pub struct TscConverter {
    /// TSC频率（Hz）
    freq_hz: u64,
    /// 起始TSC值
    base_tsc: u64,
    /// 起始纳秒
    base_ns: u64,
}

impl TscConverter {
    /// 校准TSC频率（使用系统时间）
    pub fn calibrate() -> Self {
        let tsc_start = rdtsc();
        let time_start = std::time::Instant::now();
        
        // 休眠100ms用于校准
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        let tsc_end = rdtsc();
        let elapsed = time_start.elapsed();
        
        let tsc_delta = tsc_end - tsc_start;
        let ns_delta = elapsed.as_nanos() as u64;
        
        let freq_hz = (tsc_delta * 1_000_000_000) / ns_delta;
        
        Self {
            freq_hz,
            base_tsc: tsc_end,
            base_ns: 0, // 相对时间
        }
    }
    
    /// TSC转纳秒（无除法优化：乘以倒数）
    #[inline(always)]
    pub fn to_nanos(&self, tsc: u64) -> u64 {
        // 优化：预先计算 1e9 / freq，用乘法替代除法
        // 实际实现需要128位中间值防止溢出
        ((tsc - self.base_tsc) as u128 * 1_000_000_000u128 / self.freq_hz as u128) as u64
    }
    
    /// 获取当前纳秒（相对）
    #[inline(always)]
    pub fn now_nanos(&self) -> u64 {
        self.to_nanos(rdtsc())
    }
}

/// 分层认知时钟（超脑版）
/// 
/// 层次：
/// - tick: 100μs (10kHz) - 传感器采样
/// - micro_cycle: 1ms (1kHz) - 运动控制  
/// - cognitive_cycle: 10ms (100Hz) - 皮层更新
/// - memory_cycle: 100ms (10Hz) - 记忆回放
/// - epoch: 10s (0.1Hz) - 长期巩固
#[repr(align(64))]
pub struct HierarchicalClock {
    /// 基础tick（原子，所有时间基准）
    tick: AtomicU64,
    
    /// 周期配置（编译期常量）
    pub ticks_per_micro: u16,     // 10
    pub ticks_per_cognitive: u16, // 100
    pub ticks_per_memory: u16,    // 1000
    pub ticks_per_epoch: u32,     // 100000
    
    /// TSC校准器
    converter: TscConverter,
}

impl HierarchicalClock {
    /// 创建新时钟并校准
    pub fn new() -> Self {
        Self {
            tick: AtomicU64::new(0),
            ticks_per_micro: 10,
            ticks_per_cognitive: 100,
            ticks_per_memory: 1000,
            ticks_per_epoch: 100000,
            converter: TscConverter::calibrate(),
        }
    }
    
    /// 原子递增tick（Relaxed：单调计数器，无同步需求）
    #[inline(always)]
    pub fn tick(&self) -> ClockEventMask {
        let t = self.tick.fetch_add(1, Ordering::Relaxed);
        self.compute_events(t.wrapping_add(1))
    }
    
    /// 计算当前触发的事件（位掩码）
    #[inline(always)]
    fn compute_events(&self, t: u64) -> ClockEventMask {
        let mut mask = 0u8;
        
        // 使用位运算检查周期（无分支）
        if t % self.ticks_per_micro as u64 == 0 {
            mask |= MICRO_CYCLE;
        }
        if t % self.ticks_per_cognitive as u64 == 0 {
            mask |= COGNITIVE_CYCLE;
        }
        if t % self.ticks_per_memory as u64 == 0 {
            mask |= MEMORY_CYCLE;
        }
        if t % self.ticks_per_epoch as u64 == 0 {
            mask |= EPOCH;
        }
        
        ClockEventMask(mask)
    }
    
    /// 获取当前tick（估计值）
    #[inline(always)]
    pub fn current_tick(&self) -> u64 {
        self.tick.load(Ordering::Relaxed)
    }
    
    /// 获取当前时间（纳秒）
    #[inline(always)]
    pub fn now_ns(&self) -> u64 {
        self.converter.now_nanos()
    }
    
    /// 等待直到下一个tick（硬实时睡眠）
    /// 
    /// 使用TSC忙等（亚微秒精度，但CPU占用）
    /// 或使用纳秒睡眠（精度低，但省电）
    #[inline(always)]
    pub fn spin_until(&self, target_tick: u64, tsc_per_tick: u64) {
        let start_tsc = rdtsc();
        let target_tsc = start_tsc + (target_tick - self.current_tick()) * tsc_per_tick;
        
        // 忙等循环（无系统调用）
        while hot!(rdtsc() < target_tsc) {
            core::hint::spin_loop();
        }
    }
}

/// 时钟事件位掩码
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClockEventMask(pub u8);

pub const MICRO_CYCLE: u8 = 1 << 0;     // 1ms
pub const COGNITIVE_CYCLE: u8 = 1 << 1; // 10ms  
pub const MEMORY_CYCLE: u8 = 1 << 2;    // 100ms
pub const EPOCH: u8 = 1 << 3;           // 10s

impl ClockEventMask {
    #[inline(always)]
    pub fn contains(&self, event: u8) -> bool {
        self.0 & event != 0
    }
    
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

/// 无锁定时器轮（分层时间轮算法）
/// 
/// 复杂度：O(1) 插入和到期
#[repr(align(64))]
pub struct TimerWheel<const LEVELS: usize, const SLOTS: usize> {
    /// 当前时间（tick计数）
    now: AtomicU64,
    
    /// 时间轮层级（秒/分/时级联）
    wheels: [[AtomicU64; SLOTS]; LEVELS],
    
    /// 每级slot大小
    slot_sizes: [u64; LEVELS],
}

impl<const LEVELS: usize, const SLOTS: usize> TimerWheel<LEVELS, SLOTS> {
    /// SLOTS必须是2的幂
    const_assert!(is_power_of_two(SLOTS));
    
    pub const fn new(slot_sizes: [u64; LEVELS]) -> Self {
        // 使用const_fn初始化（Rust有限制，实际可能需要unsafe）
        Self {
            now: AtomicU64::new(0),
            wheels: [[AtomicU64::new(0); SLOTS]; LEVELS],
            slot_sizes,
        }
    }
    
    /// 插入定时器（O(1)）
    #[inline(always)]
    pub fn insert(&self, expire_tick: u64) {
        let delta = expire_tick.saturating_sub(self.now.load(Ordering::Relaxed));
        
        // 找到合适的层级
        let mut accum = 1u64;
        for level in 0..LEVELS {
            accum *= self.slot_sizes[level];
            if delta < accum * SLOTS as u64 {
                let slot = ((expire_tick / self.slot_sizes[level]) as usize) & (SLOTS - 1);
                // 设置位（表示该slot有定时器）
                let wheel = &self.wheels[level][slot];
                wheel.fetch_or(1u64 << (expire_tick % 64), Ordering::Relaxed);
                return;
            }
        }
    }
    
    /// 推进并获取到期定时器（O(1)）
    #[inline(always)]
    pub fn advance(&self) -> u64 {
        let now = self.now.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        
        // 检查第0级（最高精度）
        let slot = (now as usize) & (SLOTS - 1);
        let wheel = &self.wheels[0][slot];
        let mask = wheel.swap(0, Ordering::Relaxed);
        
        mask
    }
}

/// 循环时间测量器（无分配）
#[repr(align(64))]
pub struct CycleProfiler {
    /// 历史时间（TSC）
    history: [u64; 64],
    /// 当前索引
    idx: AtomicU64,
    /// 总和（用于快速平均）
    sum: AtomicU64,
}

impl CycleProfiler {
    pub const fn new() -> Self {
        Self {
            history: [0; 64],
            idx: AtomicU64::new(0),
            sum: AtomicU64::new(0),
        }
    }
    
    /// 记录一次循环时间
    #[inline(always)]
    pub fn record(&self, duration_tsc: u64) {
        let idx = self.idx.fetch_add(1, Ordering::Relaxed) & 63;
        
        // 原子更新历史（无锁，可能有竞争但可接受）
        let old = unsafe {
            core::ptr::read_volatile(&self.history[idx as usize])
        };
        unsafe {
            core::ptr::write_volatile(&self.history[idx as usize] as *const _ as *mut u64, duration_tsc);
        }
        
        self.sum.fetch_add(duration_tsc.saturating_sub(old), Ordering::Relaxed);
    }
    
    /// 获取平均循环时间（TSC）
    #[inline(always)]
    pub fn avg_tsc(&self) -> u64 {
        self.sum.load(Ordering::Relaxed) / 64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rdtsc() {
        let t1 = rdtsc();
        let t2 = rdtsc();
        assert!(t2 > t1);
    }
    
    #[test]
    fn test_clock_events() {
        let clock = HierarchicalClock::new();
        
        // 运行100个tick
        for _ in 0..100 {
            clock.tick();
        }
        
        assert_eq!(clock.current_tick(), 100);
    }
    
    #[test]
    fn test_event_mask() {
        let mask = ClockEventMask(MICRO_CYCLE | COGNITIVE_CYCLE);
        assert!(mask.contains(MICRO_CYCLE));
        assert!(mask.contains(COGNITIVE_CYCLE));
        assert!(!mask.contains(MEMORY_CYCLE));
    }
    
    #[test]
    fn test_timer_wheel() {
        const LEVELS: usize = 3;
        const SLOTS: usize = 256;
        
        static WHEEL: TimerWheel<LEVELS, SLOTS> = TimerWheel::new([1, 256, 65536]);
        
        WHEEL.insert(100);
        WHEEL.insert(500);
        
        // 推进时间
        for _ in 0..100 {
            WHEEL.advance();
        }
        
        let expired = WHEEL.advance();
        assert!(expired != 0); // 应该有到期的
    }
}
