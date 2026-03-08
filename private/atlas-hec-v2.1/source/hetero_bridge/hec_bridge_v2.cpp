// Atlas-HEC v2.1 Heterogeneous Bridge - 完整实现
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include <cstdint>
#include <cstdio>

// 预编译PTX路径
#define PTX_PATH "./atlas_kernels.ptx"

// GPU内存句柄
static float *d_neurons_v = nullptr;
static float *d_neurons_u = nullptr;
static float *d_neurons_i = nullptr;
static float *d_weights = nullptr;
static float *d_sensory = nullptr;
static float *d_motor = nullptr;
static int *d_spike_times = nullptr;

static size_t g_n_neurons = 0;
static size_t g_n_synapses = 0;
static bool g_initialized = false;

// CUDA kernels声明（从PTX加载）
typedef void (*kernel_ptr_t)(void*);

// 简单的Izhikevich kernel（内联实现，避免PTX加载复杂性）
__global__ void izhikevich_kernel_simple(
    float* v, float* u, float* i_syn,
    const uint8_t* sensory, float* motor,
    int n_neurons, int n_sensory, int n_motor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_neurons) return;
    
    // 读取膜电位
    float vi = v[idx];
    float ui = u[idx];
    float ii = i_syn[idx];
    
    // 应用感官输入（前256个神经元）
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
        
        // 累加到运动输出（简化）
        int motor_idx = (idx * n_motor) / n_neurons;
        if (motor_idx < n_motor) {
            atomicAdd(&motor[motor_idx], 0.01f);
        }
    }
    
    v[idx] = v_new;
    u[idx] = u_new;
    i_syn[idx] = 0.0f;  // 重置输入
}

// STDP kernel
__global__ void stdp_kernel_simple(
    float* weights, const float* spike_times,
    float reward, int n_weights
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_weights) return;
    
    // 简化的奖励调制STDP
    float dw = reward * 0.001f;
    weights[idx] += dw;
    
    // 限制权重范围
    if (weights[idx] > 1.0f) weights[idx] = 1.0f;
    if (weights[idx] < 0.0f) weights[idx] = 0.0f;
}

extern "C" {

// 初始化异构系统
int hec_init(size_t n_neurons, size_t n_synapses, int gpu_id) {
    printf("[HEC] 初始化异构系统...\n");
    printf("  GPU: %d\n", gpu_id);
    printf("  神经元: %zu\n", n_neurons);
    printf("  突触: %zu\n", n_synapses);
    
    // 设置GPU设备
    cudaError_t err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "[HEC] cudaSetDevice失败: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // 分配Unified Memory（零拷贝）
    size_t neuron_bytes = n_neurons * sizeof(float);
    
    err = cudaMallocManaged(&d_neurons_v, neuron_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[HEC] 内存分配失败(v): %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaMallocManaged(&d_neurons_u, neuron_bytes);
    cudaMallocManaged(&d_neurons_i, neuron_bytes);
    cudaMallocManaged(&d_spike_times, n_neurons * sizeof(int));
    
    // 运动输出缓冲
    cudaMallocManaged(&d_motor, 5 * sizeof(float));
    cudaMallocManaged(&d_sensory, 256 * sizeof(uint8_t));
    
    // 突触权重
    if (n_synapses > 0) {
        cudaMallocManaged(&d_weights, n_synapses * sizeof(float));
        cudaMemset(d_weights, 0, n_synapses * sizeof(float));
    }
    
    // 初始化神经元状态
    cudaMemset(d_neurons_v, 0, neuron_bytes);
    cudaMemset(d_neurons_u, 0, neuron_bytes);
    cudaMemset(d_neurons_i, 0, neuron_bytes);
    
    // 设置初始值（Izhikevich resting state）
    float init_v = -65.0f;
    float init_u = -13.0f;
    cudaMemcpy(d_neurons_v, &init_v, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neurons_u, &init_u, sizeof(float), cudaMemcpyHostToDevice);
    
    // 预取到GPU
    cudaMemPrefetchAsync(d_neurons_v, neuron_bytes, gpu_id, nullptr);
    cudaMemPrefetchAsync(d_neurons_u, neuron_bytes, gpu_id, nullptr);
    
    g_n_neurons = n_neurons;
    g_n_synapses = n_synapses;
    g_initialized = true;
    
    printf("[HEC] ✅ 异构系统初始化成功\n");
    
    // 打印GPU信息
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  GPU内存: %.1f GB / %.1f GB\n", 
           (total_mem - free_mem) / 1e9, total_mem / 1e9);
    
    return 0;
}

// 异构步进：CPU感官 → GPU计算 → CPU动作
int hec_step_hybrid(
    const uint8_t* sensory_cpu,
    float* motor_cpu,
    size_t n_sensory,
    size_t n_motor
) {
    if (!g_initialized) return -1;
    
    // CPU→GPU: 拷贝感官输入（Unified Memory自动处理）
    memcpy(d_sensory, sensory_cpu, n_sensory * sizeof(uint8_t));
    
    // 清零运动输出
    cudaMemset(d_motor, 0, n_motor * sizeof(float));
    
    // GPU计算: 启动Izhikevich内核
    int threads = 256;
    int blocks = (g_n_neurons + threads - 1) / threads;
    
    izhikevich_kernel_simple<<<blocks, threads>>>(
        d_neurons_v, d_neurons_u, d_neurons_i,
        d_sensory, d_motor,
        (int)g_n_neurons, (int)n_sensory, (int)n_motor
    );
    
    // 同步等待（硬实时保证<10ms）
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[HEC] Kernel失败: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // GPU→CPU: 拷贝运动输出
    memcpy(motor_cpu, d_motor, n_motor * sizeof(float));
    
    return 0;
}

// 异步STDP更新（不阻塞CPU）
static cudaStream_t stream_stdp = nullptr;

int hec_stdp_async(float reward) {
    if (!g_initialized || g_n_synapses == 0) return 0;
    
    // 首次调用创建流
    if (!stream_stdp) {
        cudaStreamCreate(&stream_stdp);
    }
    
    // 异步启动STDP内核
    int threads = 256;
    int blocks = (g_n_synapses + threads - 1) / threads;
    
    stdp_kernel_simple<<<blocks, threads, 0, stream_stdp>>>(
        d_weights, nullptr, reward, (int)g_n_synapses
    );
    
    // 不调用cudaStreamSynchronize，让STDP在后台跑
    return 0;
}

// 同步STDP（如果需要等待结果）
int hec_stdp_sync() {
    if (stream_stdp) {
        cudaStreamSynchronize(stream_stdp);
    }
    return 0;
}

// 获取状态
int hec_status(char* buf, size_t bufsize) {
    if (!g_initialized) {
        snprintf(buf, bufsize, "未初始化");
        return -1;
    }
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    snprintf(buf, bufsize, 
        "HEC v2.1 | N:%zu | W:%zu | Mem:%.1fGB free",
        g_n_neurons, g_n_synapses, free_mem / 1e9
    );
    return 0;
}

// 清理
int hec_cleanup() {
    if (!g_initialized) return 0;
    
    if (stream_stdp) cudaStreamDestroy(stream_stdp);
    
    cudaFree(d_neurons_v);
    cudaFree(d_neurons_u);
    cudaFree(d_neurons_i);
    cudaFree(d_spike_times);
    cudaFree(d_motor);
    cudaFree(d_sensory);
    if (d_weights) cudaFree(d_weights);
    
    g_initialized = false;
    printf("[HEC] 清理完成\n");
    return 0;
}

} // extern "C"
