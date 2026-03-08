//! DigitalMetabolism: MiniGravity-inspired Bio-rhythm
//! 虚拟时区 + 代谢能量预算 + REM睡眠模式

/// 昼夜节律状态
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VigilanceState {
    DeepSleep,     // 仅核心脉冲维持
    REM,           // 低功耗记忆回放
    Light,         // 过渡状态
    Active,        // 全速计算
    Hyper,         // 高警觉 (限时)
}

/// 数字代谢系统
pub struct DigitalMetabolism {
    /// 虚拟时间相位 (0.0 - 24.0小时)
    virtual_hour: f32,
    
    /// 腺苷积累 (睡眠压力，0.0-1.0)
    adenosine_level: f32,
    
    /// 皮质醇水平 (警觉度)
    cortisol: f32,
    
    /// 连续运行时间 (秒)
    awake_time: f32,
    
    /// 计算负荷累积
    compute_load: f32,
}

impl DigitalMetabolism {
    pub fn new() -> Self {
        Self {
            virtual_hour: 8.0,  // 早8点开始
            adenosine_level: 0.0,
            cortisol: 0.8,      // 晨间高警觉
            awake_time: 0.0,
            compute_load: 0.0,
        }
    }
    
    /// 更新代谢状态 (每步调用)
    #[inline(always)]
    pub fn step(&mut self, dt_seconds: f32, neuron_spikes: u32) {
        self.virtual_hour = (self.virtual_hour + dt_seconds / 3600.0) % 24.0;
        self.awake_time += dt_seconds;
        
        // 计算负荷：脉冲越多 = 越疲劳
        let load = (neuron_spikes as f32) / 10000.0;
        self.compute_load = self.compute_load * 0.99 + load * 0.01;
        
        // 腺苷积累 (与计算负荷和清醒时间相关)
        let adenosine_rate = 0.001 * (1.0 + self.compute_load);
        self.adenosine_level += adenosine_rate * dt_seconds;
        self.adenosine_level = self.adenosine_level.min(1.0);
        
        // 皮质醇节律 (24小时周期 + 负荷响应)
        let circadian = 0.5 + 0.5 * ((self.virtual_hour - 8.0) / 12.0 * 3.14159).cos();
        let stress_response = self.compute_load * 0.2;
        self.cortisol = (circadian * 0.8 + stress_response).clamp(0.1, 1.0);
    }
    
    /// 计算能量预算 (0.0-1.0)
    #[inline(always)]
    pub fn energy_budget(&self) -> f32 {
        // 高腺苷 = 低能量 (疲劳)
        // 高皮质醇 = 可暂时提升 (应激)
        let fatigue = self.adenosine_level * 0.7;
        let alertness = self.cortisol * 0.3;
        (0.2 + alertness - fatigue).clamp(0.1, 1.0)
    }
    
    /// 当前警觉状态
    #[inline(always)]
    pub fn vigilance_state(&self) -> VigilanceState {
        match self.energy_budget() {
            e if e < 0.2 => VigilanceState::DeepSleep,
            e if e < 0.4 => VigilanceState::REM,
            e if e < 0.6 => VigilanceState::Light,
            e if e < 0.85 => VigilanceState::Active,
            _ => VigilanceState::Hyper,
        }
    }
    
    /// 是否需要REM睡眠 (数字癫痫保护)
    #[inline(always)]
    pub fn needs_rem(&self) -> bool {
        self.adenosine_level > 0.6 && self.virtual_hour > 22.0
    }
    
    /// REM睡眠清除腺苷
    #[inline(always)]
    pub fn enter_rem(&mut self) {
        self.adenosine_level *= 0.7;  // 清除30%疲劳
        self.compute_load *= 0.5;
    }
}
