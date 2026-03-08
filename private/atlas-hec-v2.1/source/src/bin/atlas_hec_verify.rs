use agl_mwe::gridworld::{SuperbrainAgent, GridWorld, run_random_benchmark};
use std::time::Instant;

fn main() {
    let start = Instant::now();
    let episodes = 50;
    
    println!("⚡ Atlas-HEC v2.1 异构验证 - CPU行为基线\n");
    
    // Random基准
    let random_stats = run_random_benchmark();
    println!("[Random Baseline] {} steps", random_stats.survival_steps);
    
    // Superbrain测试
    let mut agent = SuperbrainAgent::new();
    let mut total = 0u32;
    let mut success = 0u32;
    
    for ep in 0..episodes {
        let mut world = GridWorld::new(16, 16, 1000);
        let stats = agent.run_episode(&mut world, 1000);
        agent.reset();
        
        total += stats.survival_steps;
        if stats.survival_steps > 100 {
            success += 1;
        }
        
        if ep % 10 == 9 {
            println!("Episode {:>2}: avg={:.0} steps, success={}/{}", 
                ep+1, total as f32 / (ep+1) as f32, success, ep+1);
        }
    }
    
    let avg = total as f32 / episodes as f32;
    let ratio = avg / random_stats.survival_steps as f32;
    
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("CPU行为验证结果");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Random:    {:.0} steps", random_stats.survival_steps);
    println!("Superbrain: {:.1} steps (ratio: {:.1}x)", avg, ratio);
    println!(">100步成功率: {:.0}%", success as f32 * 100.0 / episodes as f32);
    println!("总耗时: {:?}", start.elapsed());
    println!("═══════════════════════════════════════════════════════════════");
    
    if ratio > 1.5 {
        println!("✅ PASS: CPU行为逻辑正确 (>1.5x Random)");
        std::process::exit(0);
    } else {
        println!("❌ FAIL: 需要检查STDP信号");
        std::process::exit(1);
    }
}
