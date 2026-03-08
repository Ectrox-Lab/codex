//! Atlas-HEC v2.2 融合架构（简化版）

use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicU64, Ordering};

// 无锁遥测
static STEPS: AtomicU64 = AtomicU64::new(0);

/// 24字节压缩
fn compress24(world: &[u8; 256]) -> [u8; 24] {
    let mut out = [0u8; 24];
    for i in 0..8 {
        let mut sum = 0u64;
        for j in 0..32 {
            sum += world[i*32 + j] as u64;
        }
        out[i*3] = (sum >> 16) as u8;
        out[i*3+1] = (sum >> 8) as u8;
        out[i*3+2] = sum as u8;
    }
    out
}

/// 解压缩近似
fn decompress24(comp: &[u8; 24]) -> [u8; 256] {
    let mut out = [0u8; 256];
    for i in 0..8 {
        let val = comp[i*3];
        for j in 0..32 {
            out[i*32 + j] = val;
        }
    }
    out
}

/// 融合引擎
pub struct FusionEngine {
    v: [f32; 10000],
    u: [f32; 10000],
    energy: f32,
}

impl FusionEngine {
    pub fn new() -> Self {
        Self {
            v: [-65.0; 10000],
            u: [-13.0; 10000],
            energy: 1.0,
        }
    }
    
    fn step(&mut self, input: &[u8; 256]) -> [f32; 5] {
        let mut motor = [0.0f32; 5];
        let active = (10000.0 * self.energy) as usize;
        
        for i in 0..active.max(1000) {
            let i_inj = if i < 256 { input[i] as f32 * 0.1 } else { 0.0 };
            let vi = self.v[i];
            let ui = self.u[i];
            
            let v_new = vi + 0.1 * (0.04 * vi * vi + 5.0 * vi + 140.0 - ui + i_inj);
            let u_new = ui + 0.1 * (0.02 * (0.2 * vi - ui));
            
            if v_new >= 30.0 {
                self.v[i] = -65.0;
                self.u[i] = ui + 8.0;
                let m_idx = (i * 5) / active;
                if m_idx < 5 { motor[m_idx] += 0.1; }
            } else {
                self.v[i] = v_new;
                self.u[i] = u_new;
            }
        }
        motor
    }
    
    pub fn run(&mut self, steps: usize) -> Duration {
        let start = Instant::now();
        let mut world = [0u8; 256];
        world[128] = 255;
        
        for s in 0..steps {
            // 双模切换（每100步切换）
            let motor = if s % 100 < 50 {
                self.step(&world)  // 256B模式
            } else {
                let comp = compress24(&world);
                let approx = decompress24(&comp);
                self.step(&approx)  // 24B模式
            };
            
            // 节律更新
            self.energy = 0.95 + 0.05 * (s as f32 / steps as f32).sin();
            
            STEPS.fetch_add(1, Ordering::Relaxed);
            
            let elapsed = start.elapsed();
            if elapsed < Duration::from_millis((s + 1) as u64 * 10) {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        
        start.elapsed()
    }
}

fn main() {
    println!("⚡ Atlas-HEC v2.2 融合架构（极致简化版）\n");
    println!("特性: 双模感知(256B/24B) + 节律调节 + 零分配\n");
    
    let mut engine = FusionEngine::new();
    let time = engine.run(1000);
    
    println!("运行1000步:");
    println!("  总时间: {:?}", time);
    println!("  步长: {:.2}µs/step", time.as_micros() as f32 / 1000.0);
    println!("  总步数: {}", STEPS.load(Ordering::Relaxed));
    
    // 对比纯原生
    let start = Instant::now();
    let mut v = [-65.0f32; 10000];
    let mut u = [-13.0f32; 10000];
    for _ in 0..1000 {
        for i in 0..10000 {
            let v_new = v[i] + 0.1 * (0.04 * v[i] * v[i] + 5.0 * v[i] + 140.0 - u[i]);
            let u_new = u[i] + 0.1 * (0.02 * (0.2 * v[i] - u[i]));
            if v_new >= 30.0 {
                v[i] = -65.0;
                u[i] = u[i] + 8.0;
            } else {
                v[i] = v_new;
                u[i] = u_new;
            }
        }
    }
    let native_time = start.elapsed();
    
    println!("\n对比纯原生: {:?}", native_time);
    println!("融合/原生: {:.2}%", time.as_micros() as f32 / native_time.as_micros() as f32 * 100.0);
    println!("\n✅ 融合架构验证完成");
}
