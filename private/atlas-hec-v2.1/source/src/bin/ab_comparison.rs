//! A/B对照测试入口

use std::time::{Instant, Duration};

/// 对照组: 原生256B感知
mod control {
    use std::time::{Instant, Duration};
    
    pub struct ControlGroup;
    
    impl ControlGroup {
        pub fn new() -> Self { Self }
        
        pub fn run_episode(&mut self, max_steps: usize) -> ControlResult {
            let start = Instant::now();
            let mut v = [-65.0f32; 10000];
            let mut u = [-13.0f32; 10000];
            
            for _ in 0..max_steps {
                // 256字节输入
                let input = [0u8; 256];
                
                // Izhikevich计算
                for i in 0..10000 {
                    let i_inj = if i < 256 { input[i] as f32 * 0.1 } else { 0.0 };
                    let vi = v[i];
                    let ui = u[i];
                    
                    let v_new = vi + 0.1 * (0.04 * vi * vi + 5.0 * vi + 140.0 - ui + i_inj);
                    let u_new = ui + 0.1 * (0.02 * (0.2 * vi - ui));
                    
                    if v_new >= 30.0 {
                        v[i] = -65.0;
                        u[i] = ui + 8.0;
                    } else {
                        v[i] = v_new;
                        u[i] = u_new;
                    }
                }
            }
            
            ControlResult {
                steps: max_steps,
                time: start.elapsed(),
                mode: "Control-256B",
            }
        }
    }
    
    pub struct ControlResult {
        pub steps: usize,
        pub time: Duration,
        pub mode: &'static str,
    }
}

/// 实验组: 24B压缩+CTMC
mod treatment {
    use std::time::{Instant, Duration};
    
    pub struct TreatmentGroup {
        energy: f32,
    }
    
    impl TreatmentGroup {
        pub fn new() -> Self { Self { energy: 1.0 } }
        
        pub fn run_episode(&mut self, max_steps: usize) -> TreatmentResult {
            let start = Instant::now();
            let mut v = [-65.0f32; 10000];
            let mut u = [-13.0f32; 10000];
            
            for step in 0..max_steps {
                // 24字节压缩（模拟）
                let _compressed = [0u8; 24];
                
                // CTMC节律更新
                self.energy = 0.95 + 0.05 * (step as f32 / max_steps as f32).sin();
                let active = (10000.0 * self.energy) as usize;
                
                // 适应性计算
                for i in 0..active.max(1000) {
                    let vi = v[i];
                    let ui = u[i];
                    
                    let v_new = vi + 0.1 * (0.04 * vi * vi + 5.0 * vi + 140.0 - ui);
                    
                    if v_new >= 30.0 {
                        v[i] = -65.0;
                        u[i] = ui + 8.0;
                    } else {
                        v[i] = v_new;
                        u[i] = ui + 0.1 * (0.02 * (0.2 * vi - ui));
                    }
                }
            }
            
            TreatmentResult {
                steps: max_steps,
                time: start.elapsed(),
                final_energy: self.energy,
                mode: "Treatment-24B-CTMC",
            }
        }
    }
    
    pub struct TreatmentResult {
        pub steps: usize,
        pub time: Duration,
        pub final_energy: f32,
        pub mode: &'static str,
    }
}

fn main() {
    println!("⚡ Atlas-HEC A/B对照测试\n");
    
    let episodes = 10;
    let steps = 1000;
    
    // 对照组
    println!("[对照组] Atlas-HEC原生 (256B感知)...");
    let mut ctrl_total = Duration::ZERO;
    for _ in 0..episodes {
        let mut g = control::ControlGroup::new();
        let r = g.run_episode(steps);
        ctrl_total += r.time;
    }
    let ctrl_avg = ctrl_total / episodes as u32;
    
    // 实验组
    println!("[实验组] MiniGravity-inspired (24B+CTMC)...");
    let mut treat_total = Duration::ZERO;
    for _ in 0..episodes {
        let mut g = treatment::TreatmentGroup::new();
        let r = g.run_episode(steps);
        treat_total += r.time;
    }
    let treat_avg = treat_total / episodes as u32;
    
    // 结果
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("A/B测试结果");
    println!("═══════════════════════════════════════════════════════════════");
    println!("对照组 (256B原生): {:?} / {} steps", ctrl_avg, steps);
    println!("实验组 (24B+CTMC): {:?} / {} steps", treat_avg, steps);
    
    let speedup = ctrl_avg.as_micros() as f32 / treat_avg.as_micros() as f32;
    println!("加速比: {:.2}x", speedup);
    
    if treat_avg < ctrl_avg {
        println!("✅ 实验组胜 ({}%更快)", ((1.0 - 1.0/speedup) * 100.0) as i32);
    } else {
        println!("✅ 对照组胜");
    }
}
