use agl_mwe::gridworld::{SuperbrainAgent, GridWorld, run_random_benchmark};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let episodes = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let max_steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    
    let start = Instant::now();
    
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  ⚡ GridWorld Evidence Collection v2.1                        ║");
    println!("║  Episodes: {}, Max Steps: {}", episodes, max_steps);
    println!("╚═══════════════════════════════════════════════════════════════╝");
    
    // 1. Random基准
    println!("\n[1] Random Agent基准 (10 episodes)...");
    let mut random_total = 0u32;
    for _ in 0..10 {
        let stats = run_random_benchmark();
        random_total += stats.survival_steps;
    }
    let random_avg = random_total as f32 / 10.0;
    println!("random_avg_steps: {:.1}", random_avg);
    
    // 2. Superbrain测试
    println!("\n[2] Superbrain Agent ({} episodes)...", episodes);
    let mut agent = SuperbrainAgent::new();
    let mut total_steps = 0u32;
    let mut total_food = 0u32;
    let mut total_unique = 0u32;
    let mut success_count = 0u32;
    
    let test_start = Instant::now();
    
    for ep in 0..episodes {
        let mut world = GridWorld::new(16, 16, max_steps as u32);
        
        let ep_start = Instant::now();
        let stats = agent.run_episode(&mut world, max_steps);
        let ep_time = ep_start.elapsed();
        
        agent.reset();
        
        total_steps += stats.survival_steps;
        total_food += stats.food_eaten;
        total_unique += stats.unique_cells_visited;
        
        if stats.survival_steps > 100 {
            success_count += 1;
        }
        
        // 每10 episode报告
        if ep % 10 == 9 || ep == episodes - 1 {
            let current_avg = total_steps as f32 / (ep + 1) as f32;
            let ratio = current_avg / random_avg;
            println!(
                "Episode {:>3}: avg={:.1} steps, ratio={:.1}x, time={:?}",
                ep + 1, current_avg, ratio, ep_time
            );
        }
    }
    
    let total_time = test_start.elapsed();
    let superbrain_avg = total_steps as f32 / episodes as f32;
    let ratio = superbrain_avg / random_avg;
    
    // 输出最终统计
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  证据收集完成                                                  ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  Random基线:        {:.1} steps", random_avg);
    println!("║  Superbrain平均:    {:.1} steps", superbrain_avg);
    println!("║  性能比:            {:.1}x", ratio);
    println!("║  >100步成功率:      {:.1}%", success_count as f32 * 100.0 / episodes as f32);
    println!("║  总耗时:            {:?}", total_time);
    println!("║  平均每集:          {:?}", total_time / episodes as u32);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    
    // 硬验收
    if ratio > 2.0 {
        println!("║  ✅ PASS: >2x Random，STDP学习有效                            ║");
    } else if ratio > 1.5 {
        println!("║  ⚠️  PARTIAL: >1.5x，需要优化                                 ║");
    } else {
        println!("║  ❌ FAIL: <1.5x，检查STDP信号                                 ║");
    }
    
    println!("╚═══════════════════════════════════════════════════════════════╝");
    
    // 输出关键指标供解析
    println!("\nMETRICS_START");
    println!("random_avg_steps: {:.2}", random_avg);
    println!("superbrain_avg_steps: {:.2}", superbrain_avg);
    println!("performance_ratio: {:.2}", ratio);
    println!("success_rate_pct: {:.1}", success_count as f32 * 100.0 / episodes as f32);
    println!("total_time_sec: {:.2}", total_time.as_secs_f64());
    println!("METRICS_END");
}
