//! Cognitive Timing System v2.0 - 分层时钟 + 事件调度
//!
//! 优化点：
//! - 分层时钟（避免大整数溢出）
//! - 事件优先级队列
//! - 确定性调度（可复现）

use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// 分层时钟（避免u64溢出，支持无限运行）
///
/// 结构：
/// - tick: 100Hz计数（每10ms）
/// - cognitive_cycle: 10Hz计数（每100ms）
/// - memory_cycle: 1Hz计数（每1s）
/// - epoch: 大周期（每1000s）
pub struct HierarchicalClock {
    /// 基础滴答（100Hz）
    pub tick: u64,
    /// 认知周期（10Hz）
    pub cognitive_cycle: u32,
    /// 记忆周期（1Hz）
    pub memory_cycle: u32,
    /// 大周期（0.001Hz）
    pub epoch: u32,
    
    // 周期常量
    ticks_per_cognitive: u8,
    ticks_per_memory: u16,
    ticks_per_epoch: u32,
}

impl HierarchicalClock {
    pub fn new() -> Self {
        Self {
            tick: 0,
            cognitive_cycle: 0,
            memory_cycle: 0,
            epoch: 0,
            ticks_per_cognitive: 10,      // 100Hz / 10 = 10Hz
            ticks_per_memory: 100,        // 100Hz / 100 = 1Hz
            ticks_per_epoch: 100_000,     // 100Hz / 100000 = 0.001Hz
        }
    }
    
    /// 推进一个tick（100Hz）
    pub fn tick(&mut self) -> ClockEvents {
        self.tick = self.tick.wrapping_add(1);
        
        let mut events = ClockEvents::none();
        
        // 检查认知周期
        if self.tick % self.ticks_per_cognitive as u64 == 0 {
            self.cognitive_cycle = self.cognitive_cycle.wrapping_add(1);
            events.cognitive_update = true;
        }
        
        // 检查记忆周期
        if self.tick % self.ticks_per_memory as u64 == 0 {
            self.memory_cycle = self.memory_cycle.wrapping_add(1);
            events.memory_replay = true;
        }
        
        // 检查大周期
        if self.tick % self.ticks_per_epoch as u64 == 0 {
            self.epoch = self.epoch.wrapping_add(1);
            events.epoch_boundary = true;
        }
        
        events
    }
    
    /// 获取当前时间字符串（调试用）
    pub fn format_time(&self) -> String {
        let seconds = self.tick / 100;
        let ms = (self.tick % 100) * 10;
        format!("{}.{:03}s [E{}|M{}|C{}|T{}]", 
            seconds, ms, self.epoch, self.memory_cycle, 
            self.cognitive_cycle, self.tick % 100)
    }
    
    /// 估算实际时间（假设100Hz准确）
    pub fn estimated_real_time(&self) -> Duration {
        Duration::from_millis(self.tick * 10)
    }
}

/// 时钟触发的事件
#[derive(Debug, Clone, Copy)]
pub struct ClockEvents {
    /// 认知更新（10Hz）
    pub cognitive_update: bool,
    /// 记忆回放（1Hz）
    pub memory_replay: bool,
    /// 大周期边界（0.001Hz）
    pub epoch_boundary: bool,
}

impl ClockEvents {
    pub fn none() -> Self {
        Self {
            cognitive_update: false,
            memory_replay: false,
            epoch_boundary: false,
        }
    }
    
    pub fn any(&self) -> bool {
        self.cognitive_update || self.memory_replay || self.epoch_boundary
    }
}

/// 可比较的事件（用于优先级队列）
#[derive(Debug, Clone)]
pub struct ScheduledEvent {
    /// 执行时间（tick计数）
    pub execute_at: u64,
    /// 优先级（越小越高）
    pub priority: u8,
    /// 事件类型
    pub event_type: EventType,
    /// 事件数据
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    SensorRead,
    MotorActuate,
    WorldModelUpdate,
    MemoryConsolidation,
    LearningRateDecay,
    CheckpointSave,
}

impl ScheduledEvent {
    pub fn new(execute_at: u64, priority: u8, event_type: EventType) -> Self {
        Self {
            execute_at,
            priority,
            event_type,
            data: Vec::new(),
        }
    }
}

// 为BinaryHeap实现Ord（按execute_at和priority排序）
impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // 反向排序：时间早的在前，相同时间优先级高的在前
        other.execute_at.cmp(&self.execute_at)
            .then_with(|| other.priority.cmp(&self.priority))
    }
}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.execute_at == other.execute_at && self.priority == other.priority
    }
}

impl Eq for ScheduledEvent {}

/// 事件调度器
pub struct EventScheduler {
    clock: HierarchicalClock,
    queue: BinaryHeap<ScheduledEvent>,
}

impl EventScheduler {
    pub fn new() -> Self {
        Self {
            clock: HierarchicalClock::new(),
            queue: BinaryHeap::new(),
        }
    }
    
    /// 调度未来事件
    pub fn schedule(&mut self, delay_ticks: u64, priority: u8, event_type: EventType) {
        let execute_at = self.clock.tick.wrapping_add(delay_ticks);
        let event = ScheduledEvent::new(execute_at, priority, event_type);
        self.queue.push(event);
    }
    
    /// 获取当前tick
    pub fn current_tick(&self) -> u64 {
        self.clock.tick
    }
    
    /// 推进时钟并获取到期事件
    pub fn tick(&mut self) -> (ClockEvents, Vec<ScheduledEvent>) {
        let events = self.clock.tick();
        let mut ready = Vec::new();
        let current = self.clock.tick;
        
        // 收集到期事件
        while let Some(event) = self.queue.peek() {
            if event.execute_at <= current {
                ready.push(self.queue.pop().unwrap());
            } else {
                break;
            }
        }
        
        (events, ready)
    }
    
    /// 格式化当前时间
    pub fn now(&self) -> String {
        self.clock.format_time()
    }
}

use std::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hierarchical_clock() {
        let mut clock = HierarchicalClock::new();
        
        // 运行100个tick（1秒）
        for _ in 0..100 {
            let events = clock.tick();
            if clock.tick % 10 == 0 {
                assert!(events.cognitive_update);
            }
        }
        
        assert_eq!(clock.memory_cycle, 1);
        assert_eq!(clock.cognitive_cycle, 10);
    }
    
    #[test]
    fn test_clock_events() {
        let events = ClockEvents::none();
        assert!(!events.any());
        
        let events = ClockEvents {
            cognitive_update: true,
            memory_replay: false,
            epoch_boundary: false,
        };
        assert!(events.any());
    }
    
    #[test]
    fn test_event_scheduler() {
        let mut scheduler = EventScheduler::new();
        
        // 调度未来事件
        scheduler.schedule(10, 0, EventType::WorldModelUpdate);
        scheduler.schedule(5, 1, EventType::SensorRead);
        
        // 推进到第5tick
        for _ in 0..5 {
            let (_, ready) = scheduler.tick();
            if scheduler.current_tick() == 5 {
                assert_eq!(ready.len(), 1);
                assert_eq!(ready[0].event_type, EventType::SensorRead);
            }
        }
    }
    
    #[test]
    fn test_overflow_handling() {
        let mut clock = HierarchicalClock::new();
        clock.tick = u64::MAX - 5;
        
        // 测试溢出处理
        for _ in 0..10 {
            clock.tick();
        }
        
        // 应该正常环绕
        assert!(clock.tick < 10);
    }
    
    #[test]
    fn test_time_formatting() {
        let mut clock = HierarchicalClock::new();
        clock.tick = 12345; // 123.45秒
        let formatted = clock.format_time();
        assert!(formatted.contains("123."));
    }
}
