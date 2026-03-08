//! 感知编码模块
//! 
//! 将外部输入（图像、声音等）转换为SNN可处理的神经编码

pub mod mnist_encoder;

pub use mnist_encoder::{MNISTEncoder, MNISTLoader, SNNInput};
