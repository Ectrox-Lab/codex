#include <cuda_runtime.h>
#include <cuda.h>

// Izhikevich kernel
__global__ void izhikevich_kernel(
    float* v, float* u, float* i_syn,
    const uint8_t* sensory, float* motor,
    int n_neurons, int n_sensory, int n_motor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_neurons) return;
    
    float vi = v[idx];
    float ui = u[idx];
    float ii = i_syn[idx];
    
    // 应用感官输入
    if (idx < n_sensory) {
        ii += sensory[idx] * 0.1f;
    }
    
    // Izhikevich模型
    float dt = 0.1f;
    float v_new = vi + dt * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + ii);
    float u_new = ui + dt * (0.02f * (0.2f * vi - ui));
    
    // Spike检测
    if (v_new >= 30.0f) {
        v_new = -65.0f;
        u_new += 8.0f;
        
        // 累加到运动输出
        int motor_idx = (idx * n_motor) / n_neurons;
        if (motor_idx < n_motor) {
            atomicAdd(&motor[motor_idx], 0.01f);
        }
    }
    
    v[idx] = v_new;
    u[idx] = u_new;
    i_syn[idx] = 0.0f;
}

// STDP kernel
__global__ void stdp_kernel(
    float* weights, float reward, int n_weights
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_weights) return;
    
    float dw = reward * 0.001f;
    weights[idx] += dw;
    
    if (weights[idx] > 1.0f) weights[idx] = 1.0f;
    if (weights[idx] < 0.0f) weights[idx] = 0.0f;
}

// 包装函数供外部调用
extern "C" {

void launch_izhikevich(
    float* v, float* u, float* i_syn,
    const uint8_t* sensory, float* motor,
    int n_neurons, int n_sensory, int n_motor,
    int blocks, int threads
) {
    izhikevich_kernel<<<blocks, threads>>>(
        v, u, i_syn, sensory, motor,
        n_neurons, n_sensory, n_motor
    );
}

void launch_stdp(
    float* weights, float reward, int n_weights,
    int blocks, int threads, cudaStream_t stream
) {
    stdp_kernel<<<blocks, threads, 0, stream>>>(
        weights, reward, n_weights
    );
}

} // extern "C"
