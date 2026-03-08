//! 极致传输 - NVLink优化 + 零拷贝 + 批量压缩

use super::*;
use core::sync::atomic::{AtomicU64, AtomicU32, Ordering};

/// 压缩的脉冲包（极致紧凑）
/// 
/// 格式：
/// - neuron_id: 24bit（支持1600万神经元）
/// - timestamp_delta: 8bit（相对时间，支持25.6ms@100μs精度）
/// - 总大小：4字节
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CompressedSpike {
    /// 打包数据：[timestamp_delta:8][neuron_id:24]
    data: u32,
}

impl CompressedSpike {
    #[inline(always)]
    pub const fn new(neuron_id: u32, timestamp_delta: u8) -> Self {
        Self {
            data: ((timestamp_delta as u32) << 24) | (neuron_id & 0xFFFFFF),
        }
    }
    
    #[inline(always)]
    pub const fn neuron_id(&self) -> u32 {
        self.data & 0xFFFFFF
    }
    
    #[inline(always)]
    pub const fn timestamp_delta(&self) -> u8 {
        (self.data >> 24) as u8
    }
    
    #[inline(always)]
    pub const fn as_u32(&self) -> u32 {
        self.data
    }
    
    #[inline(always)]
    pub const fn from_u32(v: u32) -> Self {
        Self { data: v }
    }
}

/// 批量Spike包（缓存行对齐）
/// 
/// 优化：
/// - 页对齐（4096字节，适合DMA）
/// - 批量传输（减少NVLink调用）
/// - 无锁生产/消费（原子索引）
#[repr(align(4096))]
pub struct SpikeBatch {
    /// 写入索引（生产者）
    write_idx: AtomicU64,
    /// 读取索引（消费者）
    read_idx: AtomicU64,
    /// 对齐填充（分离w/r索引到不同缓存行）
    _pad: [u8; CACHE_LINE - 16],
    /// 数据存储
    data: [CompressedSpike; 1024],
}

const_assert!(core::mem::size_of::<SpikeBatch>() <= PAGE_SIZE);

impl SpikeBatch {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            write_idx: AtomicU64::new(0),
            read_idx: AtomicU64::new(0),
            _pad: [0; CACHE_LINE - 16],
            data: unsafe { core::mem::zeroed() },
        }
    }
    
    /// 生产者：写入spike（无锁，失败返回false表示满）
    #[inline(always)]
    pub fn push(&self, spike: CompressedSpike) -> bool {
        let w = self.write_idx.load(Ordering::Relaxed);
        let r = self.read_idx.load(Ordering::Acquire);
        
        // 检查满（保留一个槽位区分空/满）
        if cold!((w.wrapping_sub(r)) >= 1023) {
            return false;
        }
        
        let idx = (w & 1023) as usize;
        
        // 预取下一缓存行
        if hot!((idx & 0b1111000) == 0b1111000) {
            unsafe {
                prefetch_hot(self.data.as_ptr().add((idx + 8) & 1023));
            }
        }
        
        // 写入（Release确保可见）
        unsafe {
            core::ptr::write_volatile(&self.data[idx] as *const _ as *mut _, spike);
        }
        
        self.write_idx.store(w.wrapping_add(1), Ordering::Release);
        true
    }
    
    /// 消费者：批量读取（更高效）
    #[inline(always)]
    pub fn pop_batch<const N: usize>(&self, buf: &mut [CompressedSpike; N]) -> usize {
        let r = self.read_idx.load(Ordering::Relaxed);
        let w = self.write_idx.load(Ordering::Acquire);
        
        let available = w.wrapping_sub(r);
        let to_read = N.min(available as usize).min(1024);
        
        for i in 0..to_read {
            let idx = (r.wrapping_add(i as u64) & 1023) as usize;
            unsafe {
                buf[i] = core::ptr::read_volatile(&self.data[idx]);
            }
        }
        
        if to_read > 0 {
            self.read_idx.store(r.wrapping_add(to_read as u64), Ordering::Release);
        }
        
        to_read
    }
    
    /// 当前大小（估计值）
    #[inline(always)]
    pub fn len(&self) -> usize {
        let w = self.write_idx.load(Ordering::Relaxed);
        let r = self.read_idx.load(Ordering::Relaxed);
        w.wrapping_sub(r) as usize
    }
    
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.len() >= 1023
    }
}

/// GPU间通道（双边队列）
#[repr(align(4096))]
pub struct InterGPUChannel {
    /// GPU标识
    pub gpu_id: u32,
    /// 发送批次（本地写入，远端读取）
    pub tx_batch: SpikeBatch,
    /// 接收批次（远端写入，本地读取）
    pub rx_batch: SpikeBatch,
    /// 统计信息（分离到不同缓存行）
    stats: ChannelStats,
    _pad: [u8; PAGE_SIZE - 2 * core::mem::size_of::<SpikeBatch>() - core::mem::size_of::<ChannelStats>() - 4],
}

#[repr(align(64))]
struct ChannelStats {
    tx_count: AtomicU64,
    rx_count: AtomicU64,
    drop_count: AtomicU64,
    last_tx_time: AtomicU64, // TSC
}

impl InterGPUChannel {
    pub const fn new(gpu_id: u32) -> Self {
        Self {
            gpu_id,
            tx_batch: SpikeBatch::new(),
            rx_batch: SpikeBatch::new(),
            stats: ChannelStats {
                tx_count: AtomicU64::new(0),
                rx_count: AtomicU64::new(0),
                drop_count: AtomicU64::new(0),
                last_tx_time: AtomicU64::new(0),
            },
            _pad: [0; PAGE_SIZE - 2 * core::mem::size_of::<SpikeBatch>() - core::mem::size_of::<ChannelStats>() - 4],
        }
    }
    
    /// 发送spike到远端GPU
    #[inline(always)]
    pub fn send(&self, spike: CompressedSpike) -> bool {
        if hot!(self.tx_batch.push(spike)) {
            self.stats.tx_count.fetch_add(1, Ordering::Relaxed);
            self.stats.last_tx_time.store(rdtsc(), Ordering::Relaxed);
            true
        } else {
            self.stats.drop_count.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
    
    /// 接收批量spike
    #[inline(always)]
    pub fn recv_batch<const N: usize>(&self, buf: &mut [CompressedSpike; N]) -> usize {
        let n = self.rx_batch.pop_batch(buf);
        self.stats.rx_count.fetch_add(n as u64, Ordering::Relaxed);
        n
    }
    
    /// 触发传输（实际NVLink DMA）
    /// 
    /// 注意：实际实现需要CUDA/cudaMemcpyPeerAsync
    #[inline(always)]
    pub fn flush(&self) {
        // TODO: 触发NVLink DMA
        // cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, size, stream)
    }
}

/// 脉冲压缩器（Delta编码 + 突发检测）
pub struct SpikeCompressor {
    /// 上次时间戳
    last_time: u64,
    /// 临时缓冲区
    buffer: InlineVec<CompressedSpike, 128>,
}

impl SpikeCompressor {
    pub const fn new() -> Self {
        Self {
            last_time: 0,
            buffer: InlineVec::new(),
        }
    }
    
    /// 添加原始spike（neuron_id, timestamp）
    #[inline(always)]
    pub fn add(&mut self, neuron_id: u32, timestamp: u64) -> Option<CompressedSpike> {
        let delta = timestamp.saturating_sub(self.last_time);
        
        // 时间戳 delta 必须 fit in 8bit（255 * 100μs = 25.5ms）
        if delta > 255 {
            // 刷新当前缓冲区
            self.last_time = timestamp;
        }
        
        let spike = CompressedSpike::new(neuron_id, delta as u8);
        self.buffer.push(spike);
        
        // 缓冲区满，需要刷新
        if cold!(self.buffer.len() >= 128) {
            return self.flush();
        }
        
        None
    }
    
    /// 刷新缓冲区
    #[inline(always)]
    pub fn flush(&mut self) -> Option<CompressedSpike> {
        // 返回聚合的spike包（实际实现需要更复杂的编码）
        self.buffer.clear();
        None
    }
}

/// 多GPU传输管理器
pub struct MultiGPUTransport {
    /// 通道数组
    channels: Vec<InterGPUChannel>,
    /// 本地GPU ID
    local_gpu: u32,
}

impl MultiGPUTransport {
    pub fn new(n_gpus: u32, local_gpu: u32) -> Self {
        let mut channels = Vec::with_capacity(n_gpus as usize);
        for i in 0..n_gpus {
            channels.push(InterGPUChannel::new(i));
        }
        
        Self {
            channels,
            local_gpu,
        }
    }
    
    /// 广播spike到所有其他GPU
    #[inline(always)]
    pub fn broadcast(&self, spike: CompressedSpike) {
        for (i, ch) in self.channels.iter().enumerate() {
            if i as u32 != self.local_gpu {
                let _ = ch.send(spike);
            }
        }
    }
    
    /// 发送到特定GPU
    #[inline(always)]
    pub fn send_to(&self, gpu_id: u32, spike: CompressedSpike) -> bool {
        if let Some(ch) = self.channels.get(gpu_id as usize) {
            ch.send(spike)
        } else {
            false
        }
    }
    
    /// 轮询所有通道接收
    #[inline(always)]
    pub fn poll_all<const N: usize>(&self, buf: &mut [CompressedSpike; N]) -> usize {
        let mut total = 0;
        for ch in &self.channels {
            total += ch.recv_batch(buf);
        }
        total
    }
}

/// 网络统计（原子聚合）
#[repr(align(64))]
pub struct NetworkStats {
    pub total_tx: AtomicU64,
    pub total_rx: AtomicU64,
    pub total_drops: AtomicU64,
    pub bandwidth_bps: AtomicU64,
}

impl NetworkStats {
    pub const fn new() -> Self {
        Self {
            total_tx: AtomicU64::new(0),
            total_rx: AtomicU64::new(0),
            total_drops: AtomicU64::new(0),
            bandwidth_bps: AtomicU64::new(0),
        }
    }
    
    /// 聚合所有通道的统计
    pub fn aggregate(&self, channels: &[InterGPUChannel]) {
        let mut tx = 0u64;
        let mut rx = 0u64;
        let mut drops = 0u64;
        
        for ch in channels {
            tx += ch.stats.tx_count.load(Ordering::Relaxed);
            rx += ch.stats.rx_count.load(Ordering::Relaxed);
            drops += ch.stats.drop_count.load(Ordering::Relaxed);
        }
        
        self.total_tx.store(tx, Ordering::Relaxed);
        self.total_rx.store(rx, Ordering::Relaxed);
        self.total_drops.store(drops, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compressed_spike() {
        let spike = CompressedSpike::new(12345, 100);
        assert_eq!(spike.neuron_id(), 12345);
        assert_eq!(spike.timestamp_delta(), 100);
    }
    
    #[test]
    fn test_spike_batch() {
        static BATCH: SpikeBatch = SpikeBatch::new();
        
        // 生产者线程
        for i in 0..1000 {
            let spike = CompressedSpike::new(i % 1000000, (i % 256) as u8);
            if !BATCH.push(spike) {
                break; // 满
            }
        }
        
        // 消费者线程
        let mut buf = [CompressedSpike::default(); 128];
        let n = BATCH.pop_batch(&mut buf);
        
        assert!(n > 0);
        assert_eq!(BATCH.len(), 1000 - n);
    }
    
    #[test]
    fn test_inter_gpu_channel() {
        let ch = InterGPUChannel::new(0);
        
        let spike = CompressedSpike::new(42, 10);
        assert!(ch.send(spike));
        
        // 本地测试：tx和rx是分开的
        // 实际NVLink需要远端写入rx_batch
    }
}
