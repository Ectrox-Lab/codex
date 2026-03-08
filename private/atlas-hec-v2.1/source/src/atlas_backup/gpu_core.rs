//! Atlas GPU核心 v3.0 - 全递归SNN

use cust::prelude::*;
use cust::memory::DeviceBuffer;
use rand::Rng;
use std::fs;

pub struct AtlasGPUCore {
    n_neurons: usize,
    batch_size: usize,
    n_steps: usize,
    
    v_mem: DeviceBuffer<f32>,
    spike_acc: DeviceBuffer<u32>,
    w_input: DeviceBuffer<f32>,
    w_rec: DeviceBuffer<f32>,
    
    module: Module,
}

impl AtlasGPUCore {
    pub fn new(n_neurons: usize, batch_size: usize, n_steps: usize) -> Result<Self, Box<dyn std::error::Error>> {
        println!("[AtlasGPU v3.0] {} recursive neurons", n_neurons);
        
        let mem_mb = ((784 * n_neurons + n_neurons * n_neurons + batch_size * n_neurons) * 4) / (1024 * 1024);
        println!("[AtlasGPU] GPU memory: ~{} MB", mem_mb);
        
        let v_mem = DeviceBuffer::<f32>::zeroed(batch_size * n_neurons)?;
        let spike_acc = DeviceBuffer::<u32>::zeroed(batch_size * n_neurons)?;
        
        println!("[AtlasGPU] W_input [784 x {}]...", n_neurons);
        let mut w_input_host = vec![0.0f32; 784 * n_neurons];
        for i in 0..(784 * n_neurons) {
            w_input_host[i] = (rand::random::<f32>() * 2.0 - 1.0) * 0.1;
        }
        let w_input = DeviceBuffer::from_slice(&w_input_host)?;
        
        println!("[AtlasGPU] W_rec [{0} x {0}, sparse]...", n_neurons);
        let mut w_rec_host = vec![0.0f32; n_neurons * n_neurons];
        let mut rng = rand::thread_rng();
        for i in 0..n_neurons {
            for _ in 0..(n_neurons / 10) {
                let j = rng.gen_range(0..n_neurons);
                w_rec_host[i * n_neurons + j] = (rng.gen::<f32>() * 2.0 - 1.0) * 0.05;
            }
        }
        let w_rec = DeviceBuffer::from_slice(&w_rec_host)?;
        
        let ptx_str = fs::read_to_string("cuda/atlas_kernel_v3.ptx")?;
        let module = Module::from_ptx(&ptx_str, &[])?;
        
        println!("[AtlasGPU] Ready");
        
        Ok(AtlasGPUCore {
            n_neurons, batch_size, n_steps,
            v_mem, spike_acc, w_input, w_rec,
            module,
        })
    }
    
    pub fn encode_batch(
        &mut self,
        images: &[&[f32]],
        stream: &Stream,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        assert_eq!(images.len(), self.batch_size);
        
        self.spike_acc = DeviceBuffer::<u32>::zeroed(self.batch_size * self.n_neurons)?;
        
        let mut flat_images = Vec::with_capacity(self.batch_size * 784);
        for img in images {
            flat_images.extend_from_slice(img);
        }
        let d_images = DeviceBuffer::from_slice(&flat_images)?;
        
        let total = self.batch_size * self.n_neurons;
        let block_size = 256u32;
        let grid_size = ((total + 255) / 256) as u32;
        
        let module = &self.module;
        let w_input = &self.w_input;
        let w_rec = &self.w_rec;
        let v_mem = &self.v_mem;
        let spike_acc = &self.spike_acc;
        let n_neurons = self.n_neurons as i32;
        let batch_size = self.batch_size as i32;
        let n_steps = self.n_steps as i32;
        
        unsafe {
            launch!(module.atlas_recursive_snn<<<grid_size, block_size, 0, stream>>>(
                d_images.as_device_ptr(),
                w_input.as_device_ptr(),
                w_rec.as_device_ptr(),
                v_mem.as_device_ptr(),
                spike_acc.as_device_ptr(),
                batch_size,
                n_neurons,
                n_steps,
                0.1f32,
                0.5f32
            ))?;
        }
        
        stream.synchronize()?;
        
        let mut host_spikes = vec![0u32; self.batch_size * self.n_neurons];
        self.spike_acc.copy_to(&mut host_spikes)?;
        
        Ok(host_spikes)
    }
}
