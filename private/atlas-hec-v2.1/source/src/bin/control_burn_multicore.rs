use atlas_hec_l6_living::neuron::{IzhikevichNeuron, NeuronState};
use atlas_hec_l6_living::burn_test::{BurnTestConfig, write_control_csv};
use std::time::{Instant, Duration};
use std::fs::{OpenOptions, create_dir_all};
use std::io::Write;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("🧠 Atlas-HEC L6 Burn Test - CPU MULTICORE VERSION");
    println!("═══════════════════════════════════════════════════════════");
    
    // Rayon初始化 - 使用全部核心
    let num_threads = rayon::current_num_threads();
    println!("⚡ Rayon并行池: {} 线程", num_threads);
    
    let config = BurnTestConfig::control();
    let mut neurons: Vec<IzhikevichNeuron> = (0..config.neuron_count)
        .map(|i| {
            let mut n = IzhikevichNeuron::new(
                config.a_base + (i as f64 % 0.02),
                config.b_base + (i as f64 % 0.25),
                config.c_base + (i as f64 % 60.0),
                config.d_base + (i as f64 % 8.0),
            );
            // 创建小型world连接
            for j in 0..config.connection_count {
                let target = (i + j + 1) % config.neuron_count;
                let weight = 5.0 + ((i * j) as f64 % 10.0);
                n.add_connection(target, weight);
            }
            n
        })
        .collect();
    
    println!("🧠 神经元数量: {}", neurons.len());
    println!("🔗 每个神经元连接数: {}", config.connection_count);
    println!("📊 目标频率: {} Hz", config.target_hz);
    println!("⏱️  目标运行时间: {} 小时", config.duration_hours);
    println!("🎯 目标步数: {}万步", config.target_steps / 10_000);
    println!("");
    println!("🔥 开始多核Burn Test（神经元级并行）...");
    println!("═══════════════════════════════════════════════════════════");
    
    // 确保日志目录存在
    create_dir_all("logs").ok();
    
    let start = Instant::now();
    let mut step: u64 = 0;
    let target_steps = config.target_steps;
    let interval = Duration::from_millis(10); // 100Hz
    
    // 预分配spike记录向量
    let mut spike_records: Vec<(u64, u64)> = Vec::with_capacity(1000);
    let last_spike_count = AtomicU64::new(0);
    
    while step < target_steps {
        let loop_start = Instant::now();
        
        // ========== 核心并行计算 ==========
        // 使用Rayon并行更新所有神经元（同一时间步内无依赖）
        let spike_count = AtomicU64::new(0);
        
        neurons.par_iter_mut().for_each(|neuron| {
            let current = neuron.current;
            let (spiked, _) = neuron.update(current + 5.0); // 基础刺激
            if spiked {
                spike_count.fetch_add(1, Ordering::Relaxed);
            }
        });
        
        // 顺序处理spike传播（有依赖，必须串行）
        let total_spikes = spike_count.load(Ordering::Relaxed);
        last_spike_count.store(total_spikes, Ordering::Relaxed);
        
        // 记录spike数据
        if total_spikes > 0 {
            spike_records.push((step, total_spikes));
        }
        
        step += 1;
        
        // 每100步统计一次
        if step % 100 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let hz = step as f64 / elapsed;
            let progress = (step as f64 / target_steps as f64) * 100.0;
            let remaining_steps = target_steps - step;
            let eta_seconds = remaining_steps as f64 / hz;
            
            print!("\r⚡ Step: {:>8} | Hz: {:>5.1} | 进度: {:>5.2}% | 放电: {:>4} | ETA: {:>5}m",
                step, hz, progress, total_spikes, (eta_seconds / 60.0) as u64);
            std::io::stdout().flush().unwrap();
            
            // 定期写入CSV
            if step % 1000 == 0 {
                write_control_csv(step, hz, progress, &spike_records);
                spike_records.clear();
            }
        }
        
        // 频率控制（保持100Hz）
        let elapsed = loop_start.elapsed();
        if elapsed < interval {
            std::thread::sleep(interval - elapsed);
        }
    }
    
    // 最终总结
    let total_time = start.elapsed();
    let avg_hz = step as f64 / total_time.as_secs_f64();
    
    println!("\n\n═══════════════════════════════════════════════════════════");
    println!("✅ 多核Burn Test 完成！");
    println!("═══════════════════════════════════════════════════════════");
    println!("📊 统计:");
    println!("   总步数: {}", step);
    println!("   总时间: {:.2} 小时", total_time.as_secs_f64() / 3600.0);
    println!("   平均频率: {:.2} Hz", avg_hz);
    println!("   并行线程: {}", num_threads);
    println!("═══════════════════════════════════════════════════════════");
}
