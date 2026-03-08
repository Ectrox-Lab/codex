//! Cascade-lite: MiniGravity-inspired Event Routing
//! 24字节意向压缩 + 异步STDP注入

use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam::channel::{bounded, TrySendError};

/// 24字节意向包 (MiniGravity Probe压缩)
#[derive(Clone, Copy, Debug)]
pub struct IntentPacket([u8; 24]);

impl IntentPacket {
    /// 从256字节GridWorld状态压缩 (xxHash-based)
    #[inline(always)]
    pub fn compress(world: &[u8; 256]) -> Self {
        let mut buf = [0u8; 24];
        for i in 0..8 {
            let chunk = &world[i*32..(i+1)*32];
            // 简化的哈希：每32字节压缩为3字节
            let sum: u64 = chunk.iter().map(|&b| b as u64).sum();
            buf[i*3] = ((sum >> 16) & 0xFF) as u8;
            buf[i*3+1] = ((sum >> 8) & 0xFF) as u8;
            buf[i*3+2] = (sum & 0xFF) as u8;
        }
        Self(buf)
    }
    
    /// 解压缩为近似256字节 (有损恢复)
    #[inline(always)]
    pub fn decompress(&self) -> [u8; 256] {
        let mut out = [0u8; 256];
        for i in 0..8 {
            let val = self.0[i*3];
            for j in 0..32 {
                out[i*32 + j] = val.wrapping_add((j * 7) as u8);
            }
        }
        out
    }
    
    /// 汉明距离 (相似度计算，GPU友好)
    #[inline(always)]
    pub fn distance(&self, other: &Self) -> u32 {
        self.0.iter().zip(other.0.iter())
            .map(|(a, b)| (a ^ b).count_ones() as u32)
            .sum()
    }
}

/// STDP事件 (异步注入)
#[derive(Clone, Copy)]
pub struct StdpEvent {
    pub pre_neuron: u32,
    pub post_neuron: u32,
    pub delta_t: i32,  // 正负决定LTP/LTD
    pub reward: f32,
}

/// Cascade队列 (批量STDP)
pub struct CascadeQueue {
    sender: bounded::Sender<StdpEvent>,
    batch_buffer: Vec<StdpEvent>,
    batch_size: usize,
    processed: AtomicU64,
}

impl CascadeQueue {
    pub fn new(batch_size: usize) -> Self {
        let (sender, receiver) = bounded::<StdpEvent>(10000);
        
        // 后台批量处理线程
        std::thread::spawn(move || {
            let mut batch = Vec::with_capacity(batch_size);
            while let Ok(event) = receiver.recv() {
                batch.push(event);
                if batch.len() >= batch_size {
                    // 批量STDP更新 (这里调用GPU kernel)
                    batch.clear();
                }
            }
        });
        
        Self {
            sender,
            batch_buffer: Vec::with_capacity(batch_size),
            batch_size,
            processed: AtomicU64::new(0),
        }
    }
    
    /// 非阻塞提交STDP事件
    #[inline(always)]
    pub fn submit(&self, event: StdpEvent) {
        let _ = self.sender.try_send(event);
    }
    
    pub fn processed_count(&self) -> u64 {
        self.processed.load(Ordering::Relaxed)
    }
}

/// 全局遥测计数器
pub static CASCADE_EVENTS: AtomicU64 = AtomicU64::new(0);
