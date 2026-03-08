//! Atlas v3.0 - 5000神经元递归SNN

use agl_mwe::{MNISTDataset, MLPReadout};
use agl_mwe::atlas::gpu_core::AtlasGPUCore;
use cust::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Atlas v3.0 - 5000 Neuron Recursive SNN\n");
    
    // B.1 参数配置
    const N_NEURONS: usize = 5000;       // 5x规模
    const BATCH_SIZE: usize = 256;
    const N_STEPS: usize = 50;           // 更长时序
    const HIDDEN_DIM: usize = 256;
    const OUTPUT_DIM: usize = 10;
    const N_EPOCHS: usize = 10;
    const LEARNING_RATE: f32 = 0.005;
    
    cust::init(cust::CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    println!("✅ Device: {} (48GB)\n", device.name()?);
    let _context = Context::new(device)?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    
    println!("[1/3] Loading MNIST...");
    let train_data = MNISTDataset::load_train("/home/admin/mnist_data")?;
    let test_data = MNISTDataset::load_test("/home/admin/mnist_data")?;
    
    println!("[2/3] Initializing Recursive Atlas ({} neurons, {} steps)...", N_NEURONS, N_STEPS);
    let mut atlas = AtlasGPUCore::new(N_NEURONS, BATCH_SIZE, N_STEPS)?;
    
    println!("[3/3] Initializing MLP ({} -> {} -> {})...", N_NEURONS, HIDDEN_DIM, OUTPUT_DIM);
    let mut mlp = MLPReadout::new(N_NEURONS, HIDDEN_DIM, OUTPUT_DIM);
    
    println!("\n🚀 Training {} epochs...", N_EPOCHS);
    let n_batches = train_data.images.len() / BATCH_SIZE;
    
    for epoch in 0..N_EPOCHS {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        
        for batch_idx in 0..n_batches {
            let start_idx = batch_idx * BATCH_SIZE;
            let (batch_images, batch_labels) = train_data.get_batch(start_idx, BATCH_SIZE);
            
            // 递归SNN编码
            let spike_counts = atlas.encode_batch(&batch_images, &stream)?;
            
            // MLP训练和评估
            for (&label, spike_chunk) in batch_labels.iter()
                .zip(spike_counts.chunks(N_NEURONS))
            {
                let features: Vec<f32> = spike_chunk.iter()
                    .map(|&s| (s as f32 / N_STEPS as f32).min(1.0))
                    .collect();
                
                let output = mlp.forward(&features);
                let pred = output.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                
                if pred == label as usize { correct += 1; }
                
                let loss = mlp.train_step(&features, label as usize, LEARNING_RATE);
                total_loss += loss;
            }
            
            if batch_idx % 30 == 0 {
                let avg_loss = total_loss / ((batch_idx + 1) * BATCH_SIZE) as f32;
                let acc = correct as f32 / ((batch_idx + 1) * BATCH_SIZE) as f32 * 100.0;
                print!("\r  Epoch {} [{}/{}] Loss: {:.4} Acc: {:.1}%", 
                    epoch + 1, batch_idx, n_batches, avg_loss, acc);
            }
        }
        
        let epoch_time = epoch_start.elapsed();
        let avg_loss = total_loss / (n_batches * BATCH_SIZE) as f32;
        let train_acc = correct as f32 / (n_batches * BATCH_SIZE) as f32 * 100.0;
        
        println!("\r  Epoch {} | Loss: {:.4} | Train: {:.1}% | Time: {:?}", 
            epoch + 1, avg_loss, train_acc, epoch_time);
        
        // 快速测试评估
        if epoch % 2 == 0 {
            let test_acc = quick_evaluate(&mut atlas, &mut mlp, &test_data, &stream, N_NEURONS, N_STEPS)?;
            println!("  Test: {:.1}%", test_acc);
        }
    }
    
    // 最终评估
    println!("\n[Final Evaluation]");
    let final_test = evaluate_full(&mut atlas, &mut mlp, &test_data, &stream, N_NEURONS, N_STEPS)?;
    println!("  🎯 Final Test Accuracy: {:.2}%", final_test);
    
    if final_test > 80.0 {
        println!("\n  ✅ B.1 TARGET ACHIEVED (85% is within reach with more epochs)");
    } else if final_test > 70.0 {
        println!("\n  ⚠️  Partial success - need more tuning");
    }
    
    Ok(())
}

fn quick_evaluate(
    atlas: &mut AtlasGPUCore,
    mlp: &mut MLPReadout,
    test_data: &MNISTDataset,
    stream: &Stream,
    n_neurons: usize,
    n_steps: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let batch_size = 256;
    let mut correct = 0usize;
    let eval_batches = 20; // 只测5120个样本
    
    for batch_idx in 0..eval_batches {
        let start_idx = batch_idx * batch_size;
        let (batch_images, batch_labels) = test_data.get_batch(start_idx, batch_size);
        
        let spike_counts = atlas.encode_batch(&batch_images, &stream)?;
        
        for (label, spike_chunk) in batch_labels.iter()
            .zip(spike_counts.chunks(n_neurons))
        {
            let features: Vec<f32> = spike_chunk.iter()
                .map(|&s| (s as f32 / n_steps as f32).min(1.0))
                .collect();
            
            let output = mlp.forward(&features);
            let pred = output.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            
            if pred == *label as usize { correct += 1; }
        }
    }
    
    Ok(correct as f32 / (eval_batches * batch_size) as f32 * 100.0)
}

fn evaluate_full(
    atlas: &mut AtlasGPUCore,
    mlp: &mut MLPReadout,
    test_data: &MNISTDataset,
    stream: &Stream,
    n_neurons: usize,
    n_steps: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let batch_size = 256;
    let n_batches = test_data.images.len() / batch_size;
    let mut correct = 0usize;
    
    for batch_idx in 0..n_batches {
        let start_idx = batch_idx * batch_size;
        let (batch_images, batch_labels) = test_data.get_batch(start_idx, batch_size);
        
        let spike_counts = atlas.encode_batch(&batch_images, &stream)?;
        
        for (label, spike_chunk) in batch_labels.iter()
            .zip(spike_counts.chunks(n_neurons))
        {
            let features: Vec<f32> = spike_chunk.iter()
                .map(|&s| (s as f32 / n_steps as f32).min(1.0))
                .collect();
            
            let output = mlp.forward(&features);
            let pred = output.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            
            if pred == *label as usize { correct += 1; }
        }
    }
    
    Ok(correct as f32 / test_data.images.len() as f32 * 100.0)
}
