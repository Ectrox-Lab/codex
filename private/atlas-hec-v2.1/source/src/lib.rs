pub mod mnist {
    pub mod loader;
}
pub mod mlp;

// 重导出主要类型
pub use mnist::loader::MNISTDataset;
pub use mlp::MLPReadout;

// GridWorld模块
pub mod gridworld;
pub mod atlas_cuda_bridge;
pub mod hec_ffi;
pub mod sensory;
