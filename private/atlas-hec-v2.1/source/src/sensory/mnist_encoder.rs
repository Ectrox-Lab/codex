//! MNIST → SNN输入编码（速率编码）
//! 
//! 将28x28像素图像转换为784个神经元的泊松发放率

use rand::Rng;

pub struct MNISTEncoder;

/// 编码后的SNN输入
#[derive(Clone, Debug)]
pub struct SNNInput {
    pub rates: Vec<f32>,      // 784个发放率 (0.0-1.0)
    pub label: u8,            // 真实标签 (0-9)
}

impl MNISTEncoder {
    /// 28x28像素 → 784个频率编码神经元
    /// 
    /// 使用泊松速率编码：像素值越高，发放率越高
    /// 黑色(0) → 0Hz, 白色(255) → 100Hz
    pub fn encode(image: &[u8; 784]) -> Vec<f32> {
        image.iter()
            .map(|&pixel| {
                // 归一化到 0.0-1.0，作为发放概率
                let normalized = pixel as f32 / 255.0;
                // 应用非线性增强对比度
                let enhanced = normalized.powf(0.5); // gamma校正
                enhanced
            })
            .collect()
    }
    
    /// 生成泊松发放事件（用于SNN输入）
    /// 
    /// 在一个时间步内，根据发放率决定是否发放
    pub fn generate_spikes(rates: &[f32], dt_ms: f32) -> Vec<bool> {
        let mut rng = rand::thread_rng();
        rates.iter()
            .map(|&rate| {
                // 泊松过程：P(spike) = rate * dt
                let prob = rate * (dt_ms / 1000.0) * 100.0; // 缩放使峰值约100Hz
                rng.gen::<f32>() < prob.min(1.0)
            })
            .collect()
    }
    
    /// 时间维度编码：将单张图片编码为时间序列
    /// 
    /// 模拟视网膜持续观察的过程
    pub fn encode_temporal(image: &[u8; 784], timesteps: usize) -> Vec<Vec<f32>> {
        let base_rates = Self::encode(image);
        let mut sequence = Vec::with_capacity(timesteps);
        
        for t in 0..timesteps {
            // 添加时间噪声模拟真实神经波动
            let noisy: Vec<f32> = base_rates.iter()
                .map(|&r| {
                    let noise = (t as f32 * 0.1).sin() * 0.05; // 微小振荡
                    (r + noise).clamp(0.0, 1.0)
                })
                .collect();
            sequence.push(noisy);
        }
        
        sequence
    }
}

/// MNIST数据集加载器
pub struct MNISTLoader {
    pub train_images: Vec<[u8; 784]>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<[u8; 784]>,
    pub test_labels: Vec<u8>,
}

impl MNISTLoader {
    /// 从标准MNIST文件加载
    /// 
    /// 文件格式: http://yann.lecun.com/exdb/mnist/
    pub fn load_from_files(
        train_images_path: &str,
        train_labels_path: &str,
        test_images_path: &str,
        test_labels_path: &str,
    ) -> Result<Self, std::io::Error> {
        // 简化版本：尝试从numpy或预设路径加载
        // 实际实现需要解析MNIST二进制格式
        todo!("MNIST文件解析实现")
    }
    
    /// 快速测试：生成合成MNIST数据（用于架构验证）
    pub fn generate_synthetic(count: usize) -> Self {
        let mut train_images = Vec::with_capacity(count);
        let mut train_labels = Vec::with_capacity(count);
        
        for i in 0..count {
            let mut image = [0u8; 784];
            let label = (i % 10) as u8;
            
            // 为每个数字创建简单模式
            for y in 0..28 {
                for x in 0..28 {
                    let idx = y * 28 + x;
                    // 根据数字创建不同模式
                    image[idx] = match label {
                        0 => if (x as i32 - 14).abs() + (y as i32 - 14).abs() < 10 { 200 } else { 0 },
                        1 => if x == 14 { 200 } else { 0 },
                        _ => ((x * y) % 256) as u8,
                    };
                }
            }
            
            train_images.push(image);
            train_labels.push(label);
        }
        
        Self {
            train_images: train_images.clone(),
            train_labels: train_labels.clone(),
            test_images: train_images,
            test_labels: train_labels,
        }
    }
    
    /// 获取批处理数据
    pub fn get_batch(&self, start: usize, size: usize) -> Vec<SNNInput> {
        self.train_images.iter()
            .skip(start)
            .take(size)
            .zip(self.train_labels.iter().skip(start).take(size))
            .map(|(img, &label)| SNNInput {
                rates: MNISTEncoder::encode(img),
                label,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encode_dimensions() {
        let image = [128u8; 784];
        let rates = MNISTEncoder::encode(&image);
        assert_eq!(rates.len(), 784);
        assert!(rates.iter().all(|&r| r >= 0.0 && r <= 1.0));
    }
    
    #[test]
    fn test_spike_generation() {
        let rates = vec![0.5; 784];
        let spikes = MNISTEncoder::generate_spikes(&rates, 10.0);
        assert_eq!(spikes.len(), 784);
        // 大约50%应该发放（统计验证）
        let spike_count = spikes.iter().filter(|&&s| s).count();
        assert!(spike_count > 300 && spike_count < 500, "发放率应该在40-60%之间");
    }
}
