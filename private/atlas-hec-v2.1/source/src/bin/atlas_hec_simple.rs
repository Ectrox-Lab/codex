fn main() {
    println!("⚡ Atlas-HEC v2.1 简化燃烧测试\n");
    
    // 使用已有的GridWorld进行长时测试
    use agl_mwe::gridworld::{SuperbrainAgent, GridWorld};
    use std::time::Instant;
    
    let episodes = 100;
    let start = Instant::now();
    
    println!("运行{} episode基线测试...", episodes);
    
    let mut agent = SuperbrainAgent::new();
    let mut total = 0u32;
    
    for ep in 0..episodes {
        let mut world = GridWorld::new(16, 16, 1000);
        let stats = agent.run_episode(&mut world, 1000);
        agent.reset();
        total += stats.survival_steps;
        
        if ep % 10 == 9 {
            println!("Episode {}: avg={:.0} steps", ep+1, total as f32 / (ep+1) as f32);
        }
    }
    
    let elapsed = start.elapsed();
    println!("\n✅ 完成: {} episodes, {:?}", episodes, elapsed);
}
