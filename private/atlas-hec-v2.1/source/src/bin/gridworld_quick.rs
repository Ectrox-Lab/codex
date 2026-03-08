//! GridWorld快速测试 - 架构验证

use agl_mwe::gridworld::{SuperbrainAgent, GridWorld, run_random_benchmark};
use std::time::Instant;

fn main() {
    println!("⚡ Atlas Superbrain - GridWorld Quick Test\n");
    
    // 快速测试：10 episodes
    let start = Instant::now();
    
    println!("[Random Agent 基准]");
    let random_stats = run_random_benchmark();
    println!("  生存步数: {}, 食物: {}, 探索: {}", 
        random_stats.survival_steps, 
        random_stats.food_eaten,
        random_stats.unique_cells_visited
    );
    
    println!("\n[Superbrain Agent 10 episodes]");
    let mut agent = SuperbrainAgent::new();
    let mut total_steps = 0u32;
    
    for ep in 0..10 {
        let mut world = GridWorld::new(16, 16, 1000);
        let stats = agent.run_episode(&mut world, 1000);
        agent.reset();
        total_steps += stats.survival_steps;
        println!("  Episode {}: {} steps, {} food, {} cells", 
            ep + 1, stats.survival_steps, stats.food_eaten, stats.unique_cells_visited);
    }
    
    let avg = total_steps as f32 / 10.0;
    let ratio = avg / random_stats.survival_steps as f32;
    
    println!("\n[结果]");
    println!("  平均生存: {:.1} 步 (vs random {:.1}x)", avg, ratio);
    println!("  耗时: {:?}", start.elapsed());
    
    if ratio > 2.0 {
        println!("  ✅ 基础验证通过 (>2x random)");
    } else {
        println!("  ⚠️  需要优化 (<2x random)");
    }
}
