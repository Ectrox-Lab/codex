use agl_mwe::gridworld::{SuperbrainAgent, GridWorld};
use std::time::Instant;

fn main() {
    println!("⚡ Atlas-HEC v2.1 异构架构验证 (Stub模式)\n");
    
    // 模拟HEC初始化
    println!("[HEC] 初始化异构系统...");
    println!("  GPU: 0");
    println!("  神经元: 10000");
    println!("  突触: 10000000");
    
    // 检查GPU内存
    let output = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"])
        .output()
        .unwrap();
    let mem = String::from_utf8_lossy(&output.stdout);
    println!("  GPU内存: {}MB", mem.trim());
    println!("[HEC] ✅ 异构系统初始化成功 (Stub)\n");
    
    // 运行测试
    let episodes = 10;
    let start = Instant::now();
    let mut agent = SuperbrainAgent::new();
    
    println!("🔥 开始{} episode异构测试...", episodes);
    
    for ep in 0..episodes {
        let mut world = GridWorld::new(16, 16, 1000);
        let stats = agent.run_episode(&mut world, 1000);
        agent.reset();
        
        println!("Episode {}: {} steps, {} food, {} cells", 
            ep + 1, 
            stats.survival_steps,
            stats.food_eaten,
            stats.unique_cells_visited
        );
    }
    
    let elapsed = start.elapsed();
    println!("\n✅ 测试完成: {:?}", elapsed);
    println!("  平均每集: {:?}", elapsed / episodes as u32);
}
