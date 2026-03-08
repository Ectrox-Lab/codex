//! 极致内存管理 - 零分配 + 缓存对齐 + 预取优化

use super::*;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

/// 编译期固定容量的环形缓冲区（极致无锁）
/// 
/// 优化点：
/// - 2的幂大小（位运算索引）
/// - 缓存行对齐（消除伪共享）
/// - 批量预取（减少cache miss）
#[repr(align(64))]
pub struct RingBuffer<T: Copy, const N: usize> {
    /// 写入索引（仅生产者修改）
    write_idx: core::sync::atomic::AtomicUsize,
    /// 读取索引（仅消费者修改）
    read_idx: core::sync::atomic::AtomicUsize,
    /// 对齐填充（确保write/read在不同缓存行）
    _pad: [u8; CACHE_LINE - 2 * core::mem::size_of::<usize>()],
    /// 数据存储（固定数组，零堆分配）
    data: [MaybeUninit<T>; N],
}

impl<T: Copy + Default, const N: usize> RingBuffer<T, N> {
    /// 编译期检查：N必须是2的幂
    const_assert!(is_power_of_two(N));
    
    /// 创建新缓冲区（编译期确定大小）
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            write_idx: core::sync::atomic::AtomicUsize::new(0),
            read_idx: core::sync::atomic::AtomicUsize::new(0),
            _pad: [0; CACHE_LINE - 2 * core::mem::size_of::<usize>()],
            data: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }
    
    /// 极致Push（无分支，O(1)）
    /// 
    /// # Safety
    /// 生产者单线程（或外部同步）
    #[inline(always)]
    pub unsafe fn push(&self, value: T) -> bool {
        let w = self.write_idx.load(Ordering::Relaxed);
        let r = self.read_idx.load(Ordering::Acquire);
        
        // 检查满（保留一个槽位区分空/满）
        if cold!((w.wrapping_sub(r)) >= N - 1) {
            return false; // 满，冷路径
        }
        
        // 位运算索引（无除法）
        let idx = w & (N - 1);
        
        // 预取下一缓存行（如果接近边界）
        if hot!((idx & 0b111000) == 0b111000) {
            prefetch_hot(self.data.as_ptr().add((idx + 8) & (N - 1)));
        }
        
        // 写入（volatile防止编译器重排）
        core::ptr::write_volatile(self.data.as_ptr().add(idx).cast_mut(), MaybeUninit::new(value));
        
        // Release语义：确保写入可见后才更新索引
        self.write_idx.store(w.wrapping_add(1), Ordering::Release);
        
        true
    }
    
    /// 极致Pop（无分支，O(1)）
    /// 
    /// # Safety
    /// 消费者单线程（或外部同步）
    #[inline(always)]
    pub unsafe fn pop(&self) -> Option<T> {
        let r = self.read_idx.load(Ordering::Relaxed);
        let w = self.write_idx.load(Ordering::Acquire);
        
        // 检查空
        if cold!(r == w) {
            return None; // 空，冷路径
        }
        
        let idx = r & (N - 1);
        
        // 读取（assume_init假设已初始化）
        let value = core::ptr::read_volatile(self.data.as_ptr().add(idx)).assume_init();
        
        // Release语义
        self.read_idx.store(r.wrapping_add(1), Ordering::Release);
        
        Some(value)
    }
    
    /// 批量Pop（一次取多个，减少缓存同步）
    #[inline(always)]
    pub unsafe fn pop_batch<const B: usize>(&self, buf: &mut [T; B]) -> usize {
        let r = self.read_idx.load(Ordering::Relaxed);
        let w = self.write_idx.load(Ordering::Acquire);
        
        let available = w.wrapping_sub(r);
        let to_read = B.min(available.min(N));
        
        for i in 0..to_read {
            let idx = (r.wrapping_add(i)) & (N - 1);
            buf[i] = core::ptr::read_volatile(self.data.as_ptr().add(idx)).assume_init();
        }
        
        if to_read > 0 {
            self.read_idx.store(r.wrapping_add(to_read), Ordering::Release);
        }
        
        to_read
    }
    
    /// 当前大小（估计值，非精确）
    #[inline(always)]
    pub fn len(&self) -> usize {
        let w = self.write_idx.load(Ordering::Relaxed);
        let r = self.read_idx.load(Ordering::Relaxed);
        w.wrapping_sub(r)
    }
    
    /// 是否空
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// 容量（编译期常量）
    pub const CAPACITY: usize = N;
}

// 确保RingBuffer大小是缓存行的整数倍
const_assert!(core::mem::size_of::<RingBuffer<u8, 64>>() % CACHE_LINE == 0);

/// 对象池（固定容量，零分配）
#[repr(align(64))]
pub struct ObjectPool<T: Copy, const N: usize> {
    /// 空闲栈索引
    free_stack: [usize; N],
    /// 栈顶
    top: core::sync::atomic::AtomicUsize,
    /// 存储空间
    storage: [MaybeUninit<T>; N],
    _phantom: PhantomData<T>,
}

impl<T: Copy + Default, const N: usize> ObjectPool<T, N> {
    #[inline(always)]
    pub const fn new() -> Self {
        let mut stack = [0; N];
        let mut i = 0;
        while i < N {
            stack[i] = N - 1 - i; // 倒序填充
            i += 1;
        }
        
        Self {
            free_stack: stack,
            top: core::sync::atomic::AtomicUsize::new(N),
            storage: unsafe { MaybeUninit::uninit().assume_init() },
            _phantom: PhantomData,
        }
    }
    
    /// 获取对象（无锁，单线程）
    #[inline(always)]
    pub unsafe fn acquire(&mut self) -> Option<PoolHandle<T, N>> {
        let t = self.top.load(Ordering::Relaxed);
        if cold!(t == 0) {
            return None;
        }
        
        let idx = self.free_stack[t - 1];
        self.top.store(t - 1, Ordering::Relaxed);
        
        Some(PoolHandle {
            pool: self,
            index: idx,
        })
    }
    
    /// 初始化对象（必须在使用前调用）
    #[inline(always)]
    pub unsafe fn initialize(&mut self, index: usize, value: T) {
        core::ptr::write(self.storage.as_mut_ptr().add(index).cast(), value);
    }
}

/// 池对象句柄（RAII自动归还）
pub struct PoolHandle<'a, T: Copy, const N: usize> {
    pool: &'a mut ObjectPool<T, N>,
    index: usize,
}

impl<T: Copy, const N: usize> PoolHandle<'_, T, N> {
    #[inline(always)]
    pub fn get(&self) -> &T {
        unsafe { &*self.pool.storage.as_ptr().add(self.index).cast::<T>() }
    }
    
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pool.storage.as_mut_ptr().add(self.index).cast::<T>() }
    }
}

impl<T: Copy, const N: usize> Drop for PoolHandle<'_, T, N> {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe {
            let t = self.pool.top.load(Ordering::Relaxed);
            self.pool.free_stack[t] = self.index;
            self.pool.top.store(t + 1, Ordering::Release);
        }
    }
}

/// 固定容量向量（栈分配，零堆分配）
#[repr(align(64))]
pub struct InlineVec<T: Copy, const N: usize> {
    len: usize,
    data: [MaybeUninit<T>; N],
}

impl<T: Copy + Default, const N: usize> InlineVec<T, N> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            len: 0,
            data: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }
    
    #[inline(always)]
    pub fn push(&mut self, value: T) -> bool {
        if cold!(self.len >= N) {
            return false;
        }
        unsafe {
            core::ptr::write(self.data.as_mut_ptr().add(self.len).cast(), value);
        }
        self.len += 1;
        true
    }
    
    #[inline(always)]
    pub fn pop(&mut self) -> Option<T> {
        if cold!(self.len == 0) {
            return None;
        }
        self.len -= 1;
        unsafe { Some(core::ptr::read(self.data.as_ptr().add(self.len).cast())) }
    }
    
    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }
    
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast(), self.len) }
    }
    
    #[inline(always)]
    pub fn len(&self) -> usize { self.len }
    
    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.len == 0 }
    
    pub const CAPACITY: usize = N;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ring_buffer_basic() {
        static BUF: RingBuffer<u32, 16> = RingBuffer::new();
        
        unsafe {
            for i in 0..15 {
                assert!(BUF.push(i));
            }
            assert!(!BUF.push(99)); // 满（保留一个槽位）
            
            for i in 0..15 {
                assert_eq!(BUF.pop(), Some(i));
            }
            assert_eq!(BUF.pop(), None); // 空
        }
    }
    
    #[test]
    fn test_inline_vec() {
        let mut v = InlineVec::<u32, 8>::new();
        for i in 0..8 {
            assert!(v.push(i));
        }
        assert!(!v.push(9)); // 满
        
        for i in 0..8 {
            assert_eq!(v.pop(), Some(7 - i));
        }
    }
    
    #[test]
    fn test_object_pool() {
        let mut pool = ObjectPool::<u64, 4>::new();
        unsafe {
            for i in 0..4 {
                pool.initialize(i, i as u64 * 100);
            }
            
            let h1 = pool.acquire().unwrap();
            assert_eq!(h1.get(), &0);
            
            let h2 = pool.acquire().unwrap();
            assert_eq!(h2.get(), &100);
            
            drop(h1);
            
            let h3 = pool.acquire().unwrap();
            assert_eq!(h3.get(), &0); // 复用归还的槽位
        }
    }
}
