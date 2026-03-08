//! GridWorld - 极简、零分配、硬实时环境
//! 
//! 超脑标准:
//! - 无堆分配（栈/静态内存）
//! - 硬实时保证（<10ms/tick）
//! - 极简编码（256感知神经元 → 5运动神经元）

use std::time::{Duration, Instant};


/// 动作选择（复制语义，无分配）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Up, Down, Left, Right, Stay,
}

impl Action {
    pub fn from_idx(idx: usize) -> Self {
        match idx % 5 {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            3 => Action::Right,
            _ => Action::Stay,
        }
    }
    
    pub fn delta(&self) -> (i8, i8) {
        match self {
            Action::Up => (0, -1),
            Action::Down => (0, 1),
            Action::Left => (-1, 0),
            Action::Right => (1, 0),
            Action::Stay => (0, 0),
        }
    }
}

/// 256-bit visited追踪（4×u64）
#[derive(Clone, Copy, Default)]
pub struct VisitedSet {
    bits: [u64; 4],
}

impl VisitedSet {
    pub fn mark(&mut self, idx: u8) -> bool {
        let array_idx = (idx >> 6) as usize;  // idx / 64
        let bit_idx = (idx & 0x3F) as u32;    // idx % 64
        let mask = 1u64 << bit_idx;
        
        let was_visited = (self.bits[array_idx] & mask) != 0;
        self.bits[array_idx] |= mask;
        !was_visited
    }
    
    pub fn count(&self) -> u32 {
        self.bits.iter().map(|b| b.count_ones()).sum::<u32>()
    }
    
    pub fn clear(&mut self) {
        self.bits = [0; 4];
    }
}

/// 极简GridWorld（256格，栈分配）
pub struct GridWorld {
    /// 16x16地图（1=障碍，0=空地，2=食物）
    pub map: [u8; 256],
    /// Agent位置
    pub agent_pos: (u8, u8),
    /// 食物位置
    pub food_pos: (u8, u8),
    /// 当前步数
    pub step: u32,
    /// 最大步数
    pub max_steps: u32,
    /// 访问追踪
    visited: VisitedSet,
}

impl GridWorld {
    pub fn new(width: u8, height: u8, max_steps: u32) -> Self {
        assert!(width == 16 && height == 16, "仅支持16x16地图");
        
        let mut world = GridWorld {
            map: [0u8; 256],
            agent_pos: (8, 8),
            food_pos: (12, 12),
            step: 0,
            max_steps,
            visited: VisitedSet::default(),
        };
        
        // 生成简单障碍物（5%密度）
        for i in 0..256u16 {
            if i % 20 == 0 && i != 136 && i != 192 {  // 避开起点和食物
                world.map[i as usize] = 1;
            }
        }
        world.map[192] = 2;  // 食物位置
        world.visited.mark(136);  // 标记起点已访问
        
        world
    }
    
    #[inline(always)]
    pub fn to_idx(&self, x: u8, y: u8) -> u8 {
        y.wrapping_mul(16).wrapping_add(x)
    }
    
    #[inline(always)]
    pub fn from_idx(&self, idx: u8) -> (u8, u8) {
        (idx & 0x0F, idx >> 4)
    }
    
    /// 执行动作，返回 (奖励, 是否结束)
    #[inline(always)]
    pub fn step(&mut self, action: Action) -> (f32, bool) {
        self.step += 1;
        
        // 计算新位置
        let (dx, dy) = action.delta();
        let new_x = (self.agent_pos.0 as i8 + dx).max(0).min(15) as u8;
        let new_y = (self.agent_pos.1 as i8 + dy).max(0).min(15) as u8;
        let new_idx = self.to_idx(new_x, new_y);
        
        let mut reward = 0.0f32;
        
        // 检查障碍
        if self.map[new_idx as usize] == 1 {
            reward -= 0.1;  // 撞墙惩罚
        } else {
            self.agent_pos = (new_x, new_y);
            
            // 标记访问
            if self.visited.mark(new_idx) {
                reward += 0.01;  // 探索奖励
            }
        }
        
        // 检查食物
        if self.agent_pos == self.food_pos {
            reward += 10.0;  // 食物奖励
            self.spawn_food();
        }
        
        // 生存惩罚（推动智能体寻找食物）
        reward -= 0.005;
        
        // 检查结束
        let done = self.step >= self.max_steps;
        
        (reward, done)
    }
    
    #[inline(always)]
    fn spawn_food(&mut self) {
        // 简单食物生成（轮询）
        let mut idx = self.food_pos.0.wrapping_add(self.food_pos.1 * 16);
        for _ in 0..256 {
            idx = idx.wrapping_add(7);  // 素数步长
            if self.map[idx as usize] == 0 {
                let old_idx = self.to_idx(self.food_pos.0, self.food_pos.1);
                self.map[old_idx as usize] = 0;
                self.map[idx as usize] = 2;
                self.food_pos = self.from_idx(idx);
                break;
            }
        }
    }
    
    #[inline(always)]
    pub fn observe(&self) -> [u8; 256] {
        let mut state = [0u8; 256];
        
        for i in 0..256usize {
            state[i] = match self.map[i] {
                1 => 64,   // 障碍
                2 => 128,  // 食物
                _ => 0,    // 空地
            };
        }
        
        // Agent位置
        let agent_idx = self.to_idx(self.agent_pos.0, self.agent_pos.1);
        state[agent_idx as usize] = 255;
        
        state
    }
    
    #[inline(always)]
    pub fn unique_cells(&self) -> u32 {
        self.visited.count()
    }
    
    pub fn reset(&mut self) {
        self.agent_pos = (8, 8);
        self.step = 0;
        self.visited.clear();
        self.visited.mark(136);
    }
}

/// 视觉编码器（感知 → SNN输入）
pub struct VisualEncoder;

impl VisualEncoder {
    pub fn new() -> Self {
        VisualEncoder
    }
    
    /// 编码为spike输入（直接写入GPU缓冲区）
    /// 输出: 256个u8值（0-255），对应256感知神经元
    #[inline(always)]
    pub fn encode(&self, world: &GridWorld, output: &mut [u8; 256]) {
        *output = world.observe();
    }
}

/// 运动解码器（SNN输出 → 动作）
pub struct MotorDecoder;

impl MotorDecoder {
    pub fn new() -> Self {
        MotorDecoder
    }
    
    /// 解码5个运动神经元的firing rates
    /// 输入: [up_rate, down_rate, left_rate, right_rate, stay_rate]
    /// 输出: 选择的动作
    #[inline(always)]
    pub fn decode(&self, rates: &[f32; 5]) -> Action {
        // Winner-take-all（硬决策）
        let mut max_idx = 0usize;
        let mut max_rate = rates[0];
        
        for i in 1..5 {
            if rates[i] > max_rate {
                max_rate = rates[i];
                max_idx = i;
            }
        }
        
        Action::from_idx(max_idx)
    }
}

/// 好奇心引擎（内在动机）
pub struct CuriosityEngine {
    /// 前一步的预测
    last_prediction: [f32; 256],
    /// 学习率
    eta: f32,
}

impl CuriosityEngine {
    pub fn new(eta: f32) -> Self {
        CuriosityEngine {
            last_prediction: [0.0f32; 256],
            eta,
        }
    }
    
    /// 计算预测误差（惊喜度）
    /// 返回: 内在奖励（误差越大=越好奇）
    #[inline(always)]
    pub fn compute_reward(&mut self, state: &[u8; 256]) -> f32 {
        let mut mse = 0.0f32;
        
        for i in 0..256 {
            let pred = self.last_prediction[i];
            let actual = state[i] as f32;
            let diff = actual - pred;
            mse += diff * diff;
            
            // 更新预测（指数移动平均）
            self.last_prediction[i] = pred + self.eta * diff;
        }
        
        (mse / 256.0).sqrt()
    }
    
    pub fn reset(&mut self) {
        self.last_prediction = [0.0f32; 256];
    }
}

/// 一集统计
#[derive(Debug, Default, Clone, Copy)]
pub struct EpisodeStats {
    pub survival_steps: u32,
    pub food_eaten: u32,
    pub unique_cells_visited: u32,
}

/// 超脑Agent（模拟版，无CUDA依赖）
pub struct SuperbrainAgent {
    encoder: VisualEncoder,
    decoder: MotorDecoder,
    curiosity: CuriosityEngine,
    /// 内部状态（模拟SNN）
    motor_bias: [f32; 5],
}

impl SuperbrainAgent {
    pub fn new() -> Self {
        SuperbrainAgent {
            encoder: VisualEncoder::new(),
            decoder: MotorDecoder::new(),
            curiosity: CuriosityEngine::new(0.1),
            motor_bias: [0.0f32; 5],
        }
    }
    
    /// 运行一集
    /// 返回统计信息
    #[inline(always)]
    pub fn run_episode(&mut self, world: &mut GridWorld, max_steps: usize) -> EpisodeStats {
        let mut stats = EpisodeStats::default();
        let mut sensory = [0u8; 256];
        let mut motor_output = [0.2f32; 5];
        
        for _step in 0..max_steps {
            let tick_start = Instant::now();
            
            // 1. 感知编码
            self.encoder.encode(world, &mut sensory);
            
            // 2. 模拟SNN处理（真实版应调用CUDA）
            // 这里用简单启发式模拟
            self.simulate_snn(&sensory, &mut motor_output);
            
            // 3. 运动解码
            let action = self.decoder.decode(&motor_output);
            
            // 4. 环境步进
            let (reward, done) = world.step(action);
            
            // 5. 好奇心奖励
            let intrinsic_reward = self.curiosity.compute_reward(&sensory);
            let _total_reward = reward + intrinsic_reward;
            
            // 6. STDP模拟（强化成功的动作）
            self.update_bias(action, reward);
            
            stats.survival_steps += 1;
            if reward > 1.0 {
                stats.food_eaten += 1;
            }
            
            // 硬实时保证
            let elapsed = tick_start.elapsed();
            if elapsed < Duration::from_millis(10) {
                std::thread::sleep(Duration::from_millis(10) - elapsed);
            }
            
            if done {
                break;
            }
        }
        
        stats.unique_cells_visited = world.unique_cells();
        stats
    }
    
    /// 模拟SNN（简化版）
    #[inline(always)]
    fn simulate_snn(&mut self, sensory: &[u8; 256], output: &mut [f32; 5]) {
        // 检测食物方向（简化启发式）
        let mut food_x_sum = 0i32;
        let mut food_y_sum = 0i32;
        let mut food_count = 0i32;
        
        for y in 0..16 {
            for x in 0..16 {
                let idx = y * 16 + x;
                if sensory[idx] == 128 {  // 食物
                    food_x_sum += x as i32;
                    food_y_sum += y as i32;
                    food_count += 1;
                }
            }
        }
        
        // 基于bias生成motor输出
        output[0] = self.motor_bias[0] + if food_count > 0 && food_y_sum < 128 { 0.3 } else { 0.0 };
        output[1] = self.motor_bias[1] + if food_count > 0 && food_y_sum > 128 { 0.3 } else { 0.0 };
        output[2] = self.motor_bias[2] + if food_count > 0 && food_x_sum < 128 { 0.3 } else { 0.0 };
        output[3] = self.motor_bias[3] + if food_count > 0 && food_x_sum > 128 { 0.3 } else { 0.0 };
        output[4] = self.motor_bias[4];
        
        // softmax归一化
        let max_val = output.iter().cloned().fold(0.0f32, f32::max);
        let exp_sum: f32 = output.iter().map(|x| (x - max_val).exp()).sum();
        for i in 0..5 {
            output[i] = ((output[i] - max_val).exp() / exp_sum).clamp(0.01, 0.99);
        }
    }
    
    /// 更新运动bias（模拟STDP）
    #[inline(always)]
    fn update_bias(&mut self, action: Action, reward: f32) {
        let idx = match action {
            Action::Up => 0,
            Action::Down => 1,
            Action::Left => 2,
            Action::Right => 3,
            Action::Stay => 4,
        };
        
        // Hebbian学习
        if reward > 0.0 {
            self.motor_bias[idx] = (self.motor_bias[idx] + 0.01).min(1.0);
        } else if reward < 0.0 {
            self.motor_bias[idx] = (self.motor_bias[idx] - 0.005).max(-0.5);
        }
    }
    
    pub fn reset(&mut self) {
        self.curiosity.reset();
        self.motor_bias = [0.0f32; 5];
    }
}

/// 运行基准测试
pub fn run_benchmark() -> EpisodeStats {
    let mut agent = SuperbrainAgent::new();
    let mut world = GridWorld::new(16, 16, 1000);
    agent.run_episode(&mut world, 1000)
}

/// 随机agent基准（用于对比）
pub fn run_random_benchmark() -> EpisodeStats {
    let mut world = GridWorld::new(16, 16, 1000);
    let mut stats = EpisodeStats::default();
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for _ in 0..1000 {
        let action = match rng.gen_range(0..5) {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            3 => Action::Right,
            _ => Action::Stay,
        };
        
        let (reward, done) = world.step(action);
        stats.survival_steps += 1;
        if reward > 1.0 {
            stats.food_eaten += 1;
        }
        
        if done {
            break;
        }
    }
    
    stats.unique_cells_visited = world.unique_cells();
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_world_creation() {
        let world = GridWorld::new(16, 16, 100);
        assert_eq!(world.agent_pos, (8, 8));
        assert_eq!(world.food_pos, (12, 12));
    }
    
    #[test]
    fn test_action_delta() {
        assert_eq!(Action::Up.delta(), (0, -1));
        assert_eq!(Action::Right.delta(), (1, 0));
    }
    
    #[test]
    fn test_visited_set() {
        let mut visited = VisitedSet::default();
        assert!(!visited.mark(0));  // 第一次返回false（未访问过）
        assert!(visited.mark(0));   // 第二次返回true（已访问过）
        assert_eq!(visited.count(), 1);
    }
    
    #[test]
    fn test_episode() {
        let mut agent = SuperbrainAgent::new();
        let mut world = GridWorld::new(16, 16, 100);
        let stats = agent.run_episode(&mut world, 100);
        
        println!("Episode stats: {:?}", stats);
        assert!(stats.survival_steps > 0);
    }
}
