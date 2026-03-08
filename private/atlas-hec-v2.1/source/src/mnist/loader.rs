//! MNIST数据加载器（idx3/idx1-ubyte格式）

use std::fs::File;
use std::io::{Read, BufReader};
use std::path::Path;

/// MNIST数据集
pub struct MNISTDataset {
    pub images: Vec<Vec<f32>>,  // [n][784], 归一化到[0,1]
    pub labels: Vec<u8>,        // [n], 0-9
}

impl MNISTDataset {
    /// 加载训练集
    pub fn load_train(data_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let images_path = Path::new(data_dir).join("train-images-idx3-ubyte");
        let labels_path = Path::new(data_dir).join("train-labels-idx1-ubyte");
        Self::load(&images_path, &labels_path)
    }
    
    /// 加载测试集
    pub fn load_test(data_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let images_path = Path::new(data_dir).join("t10k-images-idx3-ubyte");
        let labels_path = Path::new(data_dir).join("t10k-labels-idx1-ubyte");
        Self::load(&images_path, &labels_path)
    }
    
    fn load(images_path: &Path, labels_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let images = Self::load_images(images_path)?;
        let labels = Self::load_labels(labels_path)?;
        
        assert_eq!(images.len(), labels.len(), 
            "Images and labels count mismatch: {} vs {}", images.len(), labels.len());
        
        println!("[MNIST] Loaded {} samples from {:?}", images.len(), images_path);
        Ok(MNISTDataset { images, labels })
    }
    
    fn load_images(path: &Path) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // 读取头部
        let mut header = [0u8; 16];
        reader.read_exact(&mut header)?;
        
        // 验证magic number (0x00000803 = 2051)
        let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        assert_eq!(magic, 2051, "Invalid MNIST images magic number: {}", magic);
        
        let n_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let n_rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]) as usize;
        let n_cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]) as usize;
        
        println!("[MNIST] Images: {}x{}x{}", n_images, n_rows, n_cols);
        
        // 读取像素数据
        let mut images = Vec::with_capacity(n_images);
        let mut buffer = vec![0u8; n_rows * n_cols];
        
        for _ in 0..n_images {
            reader.read_exact(&mut buffer)?;
            let pixels: Vec<f32> = buffer.iter()
                .map(|&p| p as f32 / 255.0)  // 归一化到[0,1]
                .collect();
            images.push(pixels);
        }
        
        Ok(images)
    }
    
    fn load_labels(path: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // 读取头部
        let mut header = [0u8; 8];
        reader.read_exact(&mut header)?;
        
        // 验证magic number (0x00000801 = 2049)
        let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        assert_eq!(magic, 2049, "Invalid MNIST labels magic number: {}", magic);
        
        let n_labels = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
        
        // 读取标签
        let mut labels = vec![0u8; n_labels];
        reader.read_exact(&mut labels)?;
        
        Ok(labels)
    }
    
    /// 获取batch
    pub fn get_batch(&self, start: usize, size: usize) -> (Vec<&[f32]>, &[u8]) {
        let end = (start + size).min(self.images.len());
        let images: Vec<&[f32]> = self.images[start..end].iter()
            .map(|v| v.as_slice())
            .collect();
        let labels = &self.labels[start..end];
        (images, labels)
    }
}
