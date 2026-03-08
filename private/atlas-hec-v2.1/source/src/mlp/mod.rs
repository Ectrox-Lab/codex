//! MLP读出层（CPU实现，可后续迁移到GPU）

use rand::Rng;

/// 两层MLP: input -> hidden -> output
pub struct MLPReadout {
    pub w1: Vec<Vec<f32>>,  // [input_dim, hidden_dim]
    pub b1: Vec<f32>,       // [hidden_dim]
    pub w2: Vec<Vec<f32>>,  // [hidden_dim, output_dim]
    pub b2: Vec<f32>,       // [output_dim]
    
    // 缓存前向传播值（用于反向传播）
    pub hidden: Vec<f32>,
    pub hidden_activated: Vec<f32>,
    pub output: Vec<f32>,
}

impl MLPReadout {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        // Xavier初始化
        let w1: Vec<Vec<f32>> = (0..input_dim)
            .map(|_| (0..hidden_dim)
                .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                .map(|v| v * (2.0 / (input_dim + hidden_dim) as f32).sqrt())
                .collect())
            .collect();
        
        let b1 = vec![0.0f32; hidden_dim];
        
        let w2: Vec<Vec<f32>> = (0..hidden_dim)
            .map(|_| (0..output_dim)
                .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                .map(|v| v * (2.0 / (hidden_dim + output_dim) as f32).sqrt())
                .collect())
            .collect();
        
        let b2 = vec![0.0f32; output_dim];
        
        MLPReadout {
            w1, b1, w2, b2,
            hidden: vec![0.0; hidden_dim],
            hidden_activated: vec![0.0; hidden_dim],
            output: vec![0.0; output_dim],
        }
    }
    
    /// 前向传播
    pub fn forward(&mut self, input: &[f32]) -> &[f32] {
        let hidden_dim = self.b1.len();
        let output_dim = self.b2.len();
        
        // hidden = input @ w1 + b1
        for j in 0..hidden_dim {
            let mut sum = self.b1[j];
            for (i, &x) in input.iter().enumerate() {
                sum += x * self.w1[i][j];
            }
            self.hidden[j] = sum;
            self.hidden_activated[j] = relu(sum);
        }
        
        // output = hidden_activated @ w2 + b2
        for k in 0..output_dim {
            let mut sum = self.b2[k];
            for (j, &h) in self.hidden_activated.iter().enumerate() {
                sum += h * self.w2[j][k];
            }
            self.output[k] = sum;
        }
        
        // Softmax
        softmax_inplace(&mut self.output);
        
        &self.output
    }
    
    /// 训练一步（反向传播）
    pub fn train_step(&mut self, input: &[f32], label: usize, lr: f32) -> f32 {
        let hidden_dim = self.b1.len();
        let output_dim = self.b2.len();
        
        // 前向
        self.forward(input);
        
        // 计算loss和输出层梯度
        let mut d_output = vec![0.0f32; output_dim];
        let mut loss = 0.0f32;
        
        for k in 0..output_dim {
            let target = if k == label { 1.0 } else { 0.0 };
            let pred = self.output[k];
            d_output[k] = pred - target;  // Softmax + CrossEntropy 梯度
            
            // Cross entropy loss: -log(pred[label])
            if k == label {
                loss = -pred.ln();
            }
        }
        
        // 反向传播到hidden层
        let mut d_hidden = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            let mut sum = 0.0f32;
            for k in 0..output_dim {
                sum += d_output[k] * self.w2[j][k];
            }
            d_hidden[j] = sum * relu_derivative(self.hidden[j]);
        }
        
        // 更新w2, b2
        for j in 0..hidden_dim {
            for k in 0..output_dim {
                self.w2[j][k] -= lr * self.hidden_activated[j] * d_output[k];
            }
        }
        for k in 0..output_dim {
            self.b2[k] -= lr * d_output[k];
        }
        
        // 更新w1, b1
        for (i, &x) in input.iter().enumerate() {
            for j in 0..hidden_dim {
                self.w1[i][j] -= lr * x * d_hidden[j];
            }
        }
        for j in 0..hidden_dim {
            self.b1[j] -= lr * d_hidden[j];
        }
        
        loss
    }
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = x.iter().map(|&v| (v - max).exp()).sum();
    for v in x.iter_mut() {
        *v = (*v - max).exp() / exp_sum;
    }
}
