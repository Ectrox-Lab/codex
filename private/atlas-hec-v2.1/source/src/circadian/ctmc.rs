//! Continuous-Time Markov Chain (CTMC)
//! MiniGravity节律引擎的连续时间版本

/// CTMC状态空间
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CircadianState {
    DeepSleep = 0,
    REM = 1,
    Light = 2,
    Active = 3,
    Hyper = 4,
}

/// 连续时间马尔科夫链
pub struct CircadianCTMC {
    /// 状态概率分布 [DeepSleep, REM, Light, Active, Hyper]
    state_probs: [f32; 5],
    
    /// 转移速率矩阵 Q (5x5)
    /// Q[i][j] = 从状态i到j的转移速率
    rates: [[f32; 5]; 5],
    
    /// 代谢负荷影响因子
    load_factor: f32,
}

impl CircadianCTMC {
    pub fn new() -> Self {
        let mut rates = [[0.0f32; 5]; 5];
        
        // 基础转移速率 (每小时)
        rates[0][1] = 0.1;  // DeepSleep -> REM
        rates[1][2] = 0.2;  // REM -> Light
        rates[2][3] = 0.3;  // Light -> Active
        rates[3][4] = 0.1;  // Active -> Hyper
        rates[4][3] = 0.4;  // Hyper -> Active (快速回落)
        rates[3][2] = 0.05; // Active -> Light
        rates[2][1] = 0.1;  // Light -> REM
        rates[1][0] = 0.15; // REM -> DeepSleep
        
        Self {
            state_probs: [0.0, 0.0, 0.2, 0.8, 0.0], // 初始Active为主
            rates,
            load_factor: 1.0,
        }
    }
    
    /// 更新代谢负荷 (影响转移速率)
    #[inline(always)]
    pub fn update_load(&mut self, compute_load: f32) {
        self.load_factor = 1.0 + compute_load;
        
        // 高负荷加速向低能量状态转移
        self.rates[3][2] = 0.05 * self.load_factor;  // Active -> Light
        self.rates[4][3] = 0.4 * self.load_factor;   // Hyper -> Active
    }
    
    /// CTMC步进 (Euler积分)
    #[inline(always)]
    pub fn step(&mut self, dt_hours: f32) {
        let mut new_probs = [0.0f32; 5];
        
        for i in 0..5 {
            // 流出
            let mut outflow = 0.0f32;
            for j in 0..5 {
                if i != j {
                    let rate = self.rates[i][j] * self.load_factor;
                    let flow = rate * self.state_probs[i] * dt_hours;
                    outflow += flow;
                    new_probs[j] += flow;
                }
            }
            // 流入减去流出
            new_probs[i] += self.state_probs[i] - outflow;
        }
        
        // 归一化
        let sum: f32 = new_probs.iter().sum();
        if sum > 0.0 {
            for i in 0..5 {
                self.state_probs[i] = (new_probs[i] / sum).max(0.0).min(1.0);
            }
        }
    }
    
    /// 获取主导状态
    #[inline(always)]
    pub fn dominant_state(&self) -> CircadianState {
        let max_idx = self.state_probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(3);
        match max_idx {
            0 => CircadianState::DeepSleep,
            1 => CircadianState::REM,
            2 => CircadianState::Light,
            3 => CircadianState::Active,
            _ => CircadianState::Hyper,
        }
    }
    
    /// 能量预算 (与DigitalMetabolism兼容)
    #[inline(always)]
    pub fn energy_budget(&self) -> f32 {
        self.state_probs[3] + self.state_probs[4] * 0.8
    }
}
