//! Atlas调试版 - 检查特征统计

use agl_mwe::atlas::gpu_core::AtlasGPUCore;
use agl_mwe::MNISTDataset;
use cust::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Atlas Debug - Feature Analysis\n");
    
    cust::init(cust::CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    println!("✅ Device: {}", device.name()?);
    let _context = Context::new(device)?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    
    // 加载少量数据
    let train_data = MNISTDataset::load_train("/home/admin/mnist_data")?;
    
    let n_neurons = 1000;
    let batch_size = 256;
    
    println!("\n[1] Initializing Atlas...");
    let mut atlas = AtlasGPUCore::new(n_neurons, batch_size)?;
    
    println!("\n[2] Processing first batch...");
    let (batch_images, batch_labels) = train_data.get_batch(0, batch_size);
    
    // 检查输入数据
    println!("  First image range: [{:.3}, {:.3}]", 
        batch_images[0].iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        batch_images[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    println!("  First image sum: {:.3}", batch_images[0].iter().sum::<f32>());
    
    // Atlas处理
    let spikes = atlas.process_batch(&batch_images, &stream)?;
    
    // 统计spike特征
    let total_spikes: u32 = spikes.iter().sum();
    let spike_rate = total_spikes as f32 / (batch_size * n_neurons) as f32 * 100.0;
    
    println!("\n[3] Spike Statistics:");
    println!("  Total spikes: {}/{} ({:.2}%)", 
        total_spikes, batch_size * n_neurons, spike_rate);
    
    // 每个样本的spike数
    let mut per_sample_spikes: Vec<u32> = spikes.chunks(n_neurons)
        .map(|chunk| chunk.iter().sum())
        .collect();
    per_sample_spikes.sort_unstable();
    
    println!("  Spike count per sample (sorted):");
    println!("    Min: {}, Max: {}, Median: {}", 
        per_sample_spikes[0],
        per_sample_spikes[batch_size-1],
        per_sample_spikes[batch_size/2]);
    
    // 按标签分组统计
    println!("\n[4] Spike rate by digit:");
    for digit in 0..10 {
        let digit_indices: Vec<usize> = batch_labels.iter()
            .enumerate()
            .filter(|(_, &label)| label == digit)
            .map(|(i, _)| i)
            .collect();
        
        if !digit_indices.is_empty() {
            let digit_spikes: u32 = digit_indices.iter()
                .map(|&i| spikes[i*n_neurons..(i+1)*n_neurons].iter().sum::<u32>())
                .sum();
            let digit_rate = digit_spikes as f32 / (digit_indices.len() * n_neurons) as f32 * 100.0;
            println!("  Digit {}: {:.2}% (n={})", digit, digit_rate, digit_indices.len());
        }
    }
    
    // 检查特征区分度
    println!("\n[5] Feature discriminability check:");
    let first_sample: Vec<f32> = spikes[0..n_neurons].iter().map(|&s| s as f32).collect();
    let second_sample: Vec<f32> = spikes[n_neurons..2*n_neurons].iter().map(|&s| s as f32).collect();
    
    let diff: f32 = first_sample.iter().zip(second_sample.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("  L1 distance between sample 0 and 1: {:.1}", diff);
    
    println!("\n✅ Debug complete");
    
    if spike_rate < 1.0 {
        println!("\n⚠️ WARNING: Spike rate too low (<1%), kernel may not be working");
    } else if spike_rate > 50.0 {
        println!("\n⚠️ WARNING: Spike rate too high (>50%), may be saturated");
    } else {
        println!("\n✅ Spike rate in reasonable range (1-50%)");
    }
    
    Ok(())
}
