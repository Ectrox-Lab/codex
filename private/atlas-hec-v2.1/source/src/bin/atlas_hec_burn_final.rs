use agl_mwe::gridworld::GridWorld;
use std::time::{Instant, Duration};
use std::fs::OpenOptions;
use std::io::Write;
use std::os::raw::{c_int, c_char};
use std::ffi::CString;

// 外部Bridge函数声明
#[link(name = "hec_bridge_v2")]
extern "C" {
    fn hec_init(n_neurons: usize, n_synapses: usize, gpu_id: c_int) -> c_int;
    fn hec_step_hybrid(sensory: *const u8, motor: *mut f32, n_sens: usize, n_motor: usize) -> c_int;
    fn hec_stdp_async(reward: f32) -> c_int;
    fn hec_status(buf: *mut c_char, bufsize: usize) -> c_int;
    fn hec_cleanup();
}

fn main() {
    let log_file = format!("logs/hec_burn_{}.log", 
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
    
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)
        .expect("无法创建日志");
    
    let mut log_print = |msg: &str| {
        println!("{}", msg);
        writeln!(log, "{}", msg).ok();
        log.flush().ok();
    };
    
    log_print("╔═══════════════════════════════════════════════════════════════╗");
    log_print("║  ⚡ Atlas-HEC v2.1 异构燃烧测试 (Final)                       ║");
    log_print(&format!("║  开始: {:?}", Instant::now()));
    log_print("╚═══════════════════════════════════════════════════════════════╝");
    
    // 配置
    let neurons = 10_000;
    let synapses = neurons * 1000;
    let gpu_id = 0;
    let hours = 0.1;  // 6分钟演示
    let target_steps = (hours * 3600.0 * 100.0) as u64;
    
    log_print(&format!("\n[配置]"));
    log_print(&format!("  神经元: {}", neurons));
    log_print(&format!("  突触: {}", synapses));
    log_print(&format!("  GPU: {}", gpu_id));
    log_print(&format!("  目标步数: {}", target_steps));
    
    // 初始化
    log_print(&format!("\n[HEC] 初始化异构系统..."));
    
    let init_ret = unsafe { hec_init(neurons, synapses, gpu_id) };
    let hec_available = init_ret == 0;
    
    if hec_available {
        let mut buf = vec![0u8; 256];
        unsafe { hec_status(buf.as_mut_ptr() as *mut c_char, buf.len()); }
        log_print(&format!("  ✅ HEC初始化成功"));
        log_print(&format!("  状态: {}", String::from_utf8_lossy(&buf).trim_end_matches('\0')));
    } else {
        log_print(&format!("  ❌ HEC初始化失败({})，回退到CPU", init_ret));
    }
    
    // 燃烧测试
    let mut world = GridWorld::new(16, 16, 1000);
    let mut sensory = [0u8; 256];
    let mut motor = [0.0f32; 5];
    
    log_print(&format!("\n🔥 开始燃烧..."));
    log_print(&format!("═══════════════════════════════════════════════════════════════"));
    
    let start = Instant::now();
    let mut step_count = 0u64;
    let mut total_reward = 0.0f32;
    
    for _ in 0..target_steps {
        let tick_start = Instant::now();
        
        // 感官编码
        sensory = world.observe();
        
        // HEC计算或CPU回退
        if hec_available {
            unsafe {
                let ret = hec_step_hybrid(sensory.as_ptr(), motor.as_mut_ptr(), 256, 5);
                if ret != 0 {
                    log_print(&format!("HEC错误: {}", ret));
                    break;
                }
            }
        } else {
            // CPU回退
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for i in 0..5 { motor[i] = rng.gen::<f32>(); }
        }
        
        // 解码动作
        let action_idx = motor.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(4);
        
        use agl_mwe::gridworld::Action;
        let action = match action_idx % 5 {
            0 => Action::Up, 1 => Action::Down,
            2 => Action::Left, 3 => Action::Right,
            _ => Action::Stay,
        };
        
        // 环境步进
        let (reward, done) = world.step(action);
        total_reward += reward;
        
        // 异步STDP
        if hec_available {
            unsafe { hec_stdp_async(reward); }
        }
        
        step_count += 1;
        
        // 硬实时控制
        let elapsed = tick_start.elapsed();
        if elapsed < Duration::from_millis(10) {
            std::thread::sleep(Duration::from_millis(10) - elapsed);
        }
        
        if done { break; }
    }
    
    let total_time = start.elapsed();
    let avg_step = if step_count > 0 { total_time.as_millis() as f64 / step_count as f64 } else { 0.0 };
    
    log_print(&format!("\n═══════════════════════════════════════════════════════════════"));
    log_print(&format!("✅ 燃烧测试完成"));
    log_print(&format!("  总步数: {}", step_count));
    log_print(&format!("  总时间: {:?}", total_time));
    log_print(&format!("  平均步长: {:.2}ms", avg_step));
    log_print(&format!("  总奖励: {:.2}", total_reward));
    log_print(&format!("  模式: {}", if hec_available { "异构GPU" } else { "CPU" }));
    log_print(&format!("═══════════════════════════════════════════════════════════════"));
    
    if hec_available {
        unsafe { hec_cleanup(); }
    }
}
