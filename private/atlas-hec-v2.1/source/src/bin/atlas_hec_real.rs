use agl_mwe::hec_ffi::HecSuperbrain;
use agl_mwe::gridworld::{SuperbrainAgent, GridWorld};
use std::time::Instant;

fn main() {
    println!("⚡ Atlas-HEC v2.1 异构燃烧测试\n");
    
    // 配置
    let neurons = 10_000;
    let synapses = neurons * 1000;
    let gpu_id = 0;
    
    println!("配置:");
    println!("  神经元: {}", neurons);
    println!("  突触: {}", synapses);
    println!("  GPU: {}", gpu_id);
    println!();
    
    // 初始化异构系统
    let hec = match HecSuperbrain::new(neurons, synapses, gpu_id) {
        Ok(h) => {
            println!("✅ HEC初始化成功");
            println!("  状态: {}", h.status());
            h
        }
        Err(e) => {
            eprintln!("❌ HEC初始化失败: {}", e);
            eprintln!("回退到纯CPU模式...");
            
            // 回退到纯CPU
            run_cpu_baseline();
            return;
        }
    };
    
    // 运行10 episode测试
    println!("\n🔥 开始异构测试...");
    let start = Instant::now();
    
    for ep in 0..10 {
        let mut world = GridWorld::new(16, 16, 1000);
        let mut steps = 0u32;
        
        for _ in 0..1000 {
            // 感官编码
            let sensory = world.observe();
            
            // HEC异构计算（CPU → GPU → CPU）
            let motor = match hec.step(&sensory) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("HEC错误: {}", e);
                    break;
                }
            };
            
            // 解码动作（简化）
            let action_idx = motor.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(4);
            
            use agl_mwe::gridworld::Action;
            let action = match action_idx % 5 {
                0 => Action::Up, 1 => Action::Down,
                2 => Action::Left, 3 => Action::Right,
                _ => Action::Stay,
            };
            
            // 环境步进
            let (reward, done) = world.step(action);
            
            // 异步STDP（如果成功）
            let _ = hec.stdp_async(reward);
            
            steps += 1;
            if done { break; }
        }
        
        println!("Episode {}: {} steps", ep + 1, steps);
    }
    
    let elapsed = start.elapsed();
    println!("\n✅ 异构测试完成: {:?}", elapsed);
}

fn run_cpu_baseline() {
    println!("\n运行纯CPU基线...");
    let mut agent = SuperbrainAgent::new();
    let stats = agent.run_episode(&mut GridWorld::new(16, 16, 1000), 1000);
    println!("CPU基线: {} steps", stats.survival_steps);
}
