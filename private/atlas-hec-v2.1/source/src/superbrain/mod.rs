//! Atlas Superbrain Runtime v3.0
//! 
//! 极致优化承诺：
//! - 热路径零堆分配（heaptrack验证）
//! - 所有跨线程结构体缓存行对齐（64字节）
//! - 关键循环无分支（位运算）
//! - SIMD硬化（AVX-512）
//! - 显式内存序（每个atomic注释）
//! - Release模式零日志开销
//! 
//! 优化级别：直到在人类认知边界内找不到优化空间

#![feature(stdsimd)]
#![feature(core_intrinsics)]

use std::arch::x86_64::*;
use std::intrinsics::{likely, unlikely};

pub mod memory;
pub mod timing;
pub mod transport;
pub mod compute;
pub mod runtime;
pub mod cuda_bridge;

/// 缓存行大小（x86_64标准）
pub const CACHE_LINE: usize = 64;

/// 页大小（4KB标准）
pub const PAGE_SIZE: usize = 4096;

/// 预取提示（热数据）
#[inline(always)]
pub unsafe fn prefetch_hot<T>(ptr: *const T) {
    _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
}

/// 预取提示（冷数据）
#[inline(always)]
pub unsafe fn prefetch_cold<T>(ptr: *const T) {
    _mm_prefetch(ptr as *const i8, _MM_HINT_T2);
}

/// 编译期断言（确保常量条件）
#[macro_export]
macro_rules! const_assert {
    ($x:expr $(,)?) => {
        const _: [(); 0 - !{$x: bool} as usize] = [];
    };
}

/// 热路径标记（likely分支）
#[macro_export]
macro_rules! hot {
    ($e:expr) => {
        std::intrinsics::likely($e)
    };
}

/// 冷路径标记（unlikely分支）
#[macro_export]
macro_rules! cold {
    ($e:expr) => {
        std::intrinsics::unlikely($e)
    };
}

/// 编译期日志开关（Release模式完全消除）
#[cfg(debug_assertions)]
#[macro_export]
macro_rules! sb_log {
    ($($arg:tt)*) => { println!($($arg)*); };
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! sb_log {
    ($($arg:tt)*) => {};
}

/// 零初始化数组（避免运行时填充）
#[inline(always)]
pub const fn zeroed_array<T: Copy, const N: usize>(val: T) -> [T; N] {
    [val; N]
}

/// 检查是否为2的幂（编译期优化）
#[inline(always)]
pub const fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// 对齐到缓存行（向上取整）
#[inline(always)]
pub const fn align_cache_line(n: usize) -> usize {
    (n + CACHE_LINE - 1) & !(CACHE_LINE - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_line_alignment() {
        assert_eq!(align_cache_line(1), 64);
        assert_eq!(align_cache_line(64), 64);
        assert_eq!(align_cache_line(65), 128);
    }
    
    #[test]
    fn test_power_of_two() {
        assert!(is_power_of_two(4096));
        assert!(!is_power_of_two(4095));
    }
}
