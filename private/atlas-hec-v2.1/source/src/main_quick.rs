use agl_mwe::{MNISTDataset, MLPReadout};
use agl_mwe::atlas::gpu_core::AtlasGPUCore;
use cust::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Atlas v3.0 Quick Test (5000 neurons, 1 epoch)\n");
    
    const N_NEURONS: usize = 5000;
    const BATCH_SIZE: usize = 256;
    const N_STEPS: usize = 50;
    
    cust::init(cust::CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    println!("✅ Device: {}\n", device.name()?);
    let _context = Context::new(device)?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    
    let train_data = MNISTDataset::load_train("/home/admin/mnist_data")?;
    let mut atlas = AtlasGPUCore::new(N_NEURONS, BATCH_SIZE, N_STEPS)?;
    let mut mlp = MLPReadout::new(N_NEURONS, 256, 10);
    
    let n_batches = 50; // 只跑50个batch（约5%数据）
    let mut correct = 0usize;
    
    println!("Training on {} batches...", n_batches);
    let start = Instant::now();
    
    for batch_idx in 0..n_batches {
        let (batch_images, batch_labels) = train_data.get_batch(batch_idx * BATCH_SIZE, BATCH_SIZE);
        let spike_counts = atlas.encode_batch(&batch_images, &stream)?;
        
        for (&label, spike_chunk) in batch_labels.iter().zip(spike_counts.chunks(N_NEURONS)) {
            let features: Vec<f32> = spike_chunk.iter().map(|&s| s as f32 / N_STEPS as f32).collect();
            let output = mlp.forward(&features);
            let pred = output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
            if pred == label as usize { correct += 1; }
            mlp.train_step(&features, label as usize, 0.01);
        }
        
        if batch_idx % 10 == 0 {
            let acc = correct as f32 / ((batch_idx + 1) * BATCH_SIZE) as f32 * 100.0;
            print!("\r  Batch {}/{} Acc: {:.1}%", batch_idx, n_batches, acc);
        }
    }
    
    let elapsed = start.elapsed();
    let final_acc = correct as f32 / (n_batches * BATCH_SIZE) as f32 * 100.0;
    
    println!("\r  ✅ 50 batches complete | Acc: {:.1}% | Time: {:?}", final_acc, elapsed);
    
    let time_per_batch = elapsed.as_secs_f32() / n_batches as f32;
    let est_full_epoch = time_per_batch * 234.0 / 60.0;
    
    println!("\n📊 Performance:");
    println!("  Time/batch: {:.2}s", time_per_batch);
    println!("  Est. full epoch: {:.1} min", est_full_epoch);
    println!("  Est. 10 epochs: {:.1} min", est_full_epoch * 10.0);
    
    if final_acc > 15.0 {
        println!("\n✅ Architecture validated (acc > random 10%)");
    }
    
    Ok(())
}
