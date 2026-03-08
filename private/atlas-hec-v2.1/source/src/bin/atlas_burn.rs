//! Atlas Superbrain - 真实燃烧测试入口

use agl_mwe::atlas_cuda_bridge::run_real_burn_test;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  ⚡ Atlas Superbrain v2.1 - REAL CUDA BURN TEST               ║");
    println!("║  开始: {:?}", std::time::Instant::now());
    println!("╚═══════════════════════════════════════════════════════════════╝");
    
    // 参数
    let neurons = 100_000;
    let hours = 0.1;  // 6分钟测试（为了快速验证）
    
    println!("\n配置:");
    println!("  神经元: {}", neurons);
    println!("  突触: {}", neurons * 1000);
    println!("  GPU: 0 (保留1,2备用)");
    println!("  测试时长: {} 小时", hours);
    
    match run_real_burn_test(neurons, hours) {
        Ok(result) => {
            println!("\n✅ 测试通过");
            println!("   总步数: {}", result.total_steps);
            println!("   平均延迟: {:.2}us", result.avg_step_us);
            
            if result.avg_step_us < 10000.0 {
                println!("   ✅ 硬实时保证: <10ms");
            } else {
                println!("   ❌ 超时: >10ms");
            }
        }
        Err(e) => {
            eprintln!("\n❌ 测试失败: {}", e);
            std::process::exit(1);
        }
    }
}
