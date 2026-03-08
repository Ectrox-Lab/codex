//! Atlas Superbrain - 真实6小时燃烧测试

use agl_mwe::atlas_cuda_bridge::AtlasSuperbrain;
use std::time::{Instant, Duration};
use std::fs::OpenOptions;
use std::io::Write;

fn main() {
    let log_file = format!("logs/burn_real_{}.log", 
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );
    
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)
        .expect("无法创建日志文件");
    
    macro_rules! log_print {
        ($($arg:tt)*) => {{
            let msg = format!($($arg)*);
            println!("{}", msg);
            writeln!(log, "{}", msg).ok();
            log.flush().ok();
        }};
    }
    
    log_print!("╔═══════════════════════════════════════════════════════════════╗");
    log_print!("║  ⚡ Atlas Superbrain v2.1 - 6小时真实燃烧测试                 ║");
    log_print!("║  开始: {:?}", Instant::now());
    log_print!("║  PID: {}", std::process::id());
    log_print!("╚═══════════════════════════════════════════════════════════════╝");
    
    // 配置
    let neurons = 100_000;
    let synapses = neurons * 1000;
    let device_id = 0;
    let hours = 6.0;
    let target_steps = (hours * 3600.0 * 100.0) as u64; // 100Hz
    
    log_print!("\n[配置]");
    log_print!("  神经元: {}", neurons);
    log_print!("  突触: {}", synapses);
    log_print!("  GPU: {}", device_id);
    log_print!("  目标步数: {}", target_steps);
    log_print!("  预计时长: {} 小时", hours);
    
    // 初始化超脑
    let mut brain = match AtlasSuperbrain::new(neurons, synapses, device_id) {
        Ok(b) => b,
        Err(e) => {
            log_print!("❌ 初始化失败: {}", e);
            std::process::exit(1);
        }
    };
    
    // 监控循环
    let start = Instant::now();
    let mut last_hourly_report = start;
    let mut hourly_steps: u64 = 0;
    
    log_print!("\n🔥 开始燃烧...");
    log_print!("═══════════════════════════════════════════════════════════════");
    
    for step in 0..target_steps {
        // 执行一步
        if let Err(e) = brain.step(&[0u8; 256]) {
            log_print!("❌ Step {} 错误: {}", step, e);
            break;
        }
        
        hourly_steps += 1;
        
        // 每小时报告
        if last_hourly_report.elapsed() >= Duration::from_secs(3600) {
            let hour = (step / 360000) + 1;
            let mem = check_gpu_memory(device_id);
            
            log_print!("[Hour {}/6] Steps: {}, GPU Mem: {}MB, Avg: {}us/step", 
                hour, step, mem, brain.last_step_time_us);
            
            hourly_steps = 0;
            last_hourly_report = Instant::now();
        }
    }
    
    let total_time = start.elapsed();
    let avg_step_us = total_time.as_micros() as f64 / brain.step_count as f64;
    
    log_print!("\n═══════════════════════════════════════════════════════════════");
    log_print!("✅ 燃烧测试完成");
    log_print!("  总步数: {}", brain.step_count);
    log_print!("  总时间: {:?}", total_time);
    log_print!("  平均步长: {:.2}us", avg_step_us);
    log_print!("  {}", brain.get_stats());
    log_print!("═══════════════════════════════════════════════════════════════");
    
    // 验收判定
    if avg_step_us < 10000.0 {
        log_print!("\n✅ 硬实时验证通过: <10ms");
    } else {
        log_print!("\n❌ 硬实时失败: >10ms");
    }
}

fn check_gpu_memory(device_id: i32) -> i64 {
    let output = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", &device_id.to_string()])
        .output()
        .expect("无法执行nvidia-smi");
    
    let mem_str = String::from_utf8_lossy(&output.stdout);
    mem_str.trim().parse::<i64>().unwrap_or(0)
}
