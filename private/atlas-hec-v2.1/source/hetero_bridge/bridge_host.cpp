#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

// 外部kernel启动函数
extern "C" {
void launch_izhikevich(
    float* v, float* u, float* i_syn,
    const uint8_t* sensory, float* motor,
    int n_neurons, int n_sensory, int n_motor,
    int blocks, int threads
);

void launch_stdp(
    float* weights, float reward, int n_weights,
    int blocks, int threads, cudaStream_t stream
);
}

// GPU内存句柄
static float *d_neurons_v = nullptr;
static float *d_neurons_u = nullptr;
static float *d_neurons_i = nullptr;
static float *d_weights = nullptr;
static float *d_motor = nullptr;
static uint8_t *d_sensory = nullptr;

static size_t g_n_neurons = 0;
static size_t g_n_synapses = 0;
static bool g_initialized = false;
static cudaStream_t stream_stdp = nullptr;

extern "C" {

int hec_init(size_t n_neurons, size_t n_synapses, int gpu_id) {
    printf("[HEC] 初始化异构系统...\n");
    printf("  GPU: %d, 神经元: %zu, 突触: %zu\n", gpu_id, n_neurons, n_synapses);
    
    cudaError_t err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "[HEC] cudaSetDevice失败: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // 分配Unified Memory
    size_t neuron_bytes = n_neurons * sizeof(float);
    
    cudaMallocManaged(&d_neurons_v, neuron_bytes);
    cudaMallocManaged(&d_neurons_u, neuron_bytes);
    cudaMallocManaged(&d_neurons_i, neuron_bytes);
    cudaMallocManaged(&d_motor, 5 * sizeof(float));
    cudaMallocManaged(&d_sensory, 256 * sizeof(uint8_t));
    
    if (n_synapses > 0) {
        cudaMallocManaged(&d_weights, n_synapses * sizeof(float));
        cudaMemset(d_weights, 0, n_synapses * sizeof(float));
    }
    
    // 初始化
    cudaMemset(d_neurons_v, 0, neuron_bytes);
    cudaMemset(d_neurons_u, 0, neuron_bytes);
    cudaMemset(d_neurons_i, 0, neuron_bytes);
    
    // 预取到GPU
    cudaMemPrefetchAsync(d_neurons_v, neuron_bytes, gpu_id, nullptr);
    cudaMemPrefetchAsync(d_neurons_u, neuron_bytes, gpu_id, nullptr);
    
    cudaStreamCreate(&stream_stdp);
    
    g_n_neurons = n_neurons;
    g_n_synapses = n_synapses;
    g_initialized = true;
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[HEC] ✅ 初始化成功, GPU内存: %.1fGB / %.1fGB\n",
           (total_mem - free_mem) / 1e9, total_mem / 1e9);
    
    return 0;
}

int hec_step_hybrid(
    const uint8_t* sensory_cpu,
    float* motor_cpu,
    size_t n_sensory,
    size_t n_motor
) {
    if (!g_initialized) return -1;
    
    // 拷贝输入到Unified Memory
    memcpy(d_sensory, sensory_cpu, n_sensory * sizeof(uint8_t));
    cudaMemset(d_motor, 0, n_motor * sizeof(float));
    
    // 启动kernel
    int threads = 256;
    int blocks = (g_n_neurons + threads - 1) / threads;
    
    launch_izhikevich(
        d_neurons_v, d_neurons_u, d_neurons_i,
        d_sensory, d_motor,
        (int)g_n_neurons, (int)n_sensory, (int)n_motor,
        blocks, threads
    );
    
    // 同步
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[HEC] kernel失败: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // 拷贝输出
    memcpy(motor_cpu, d_motor, n_motor * sizeof(float));
    
    return 0;
}

int hec_stdp_async(float reward) {
    if (!g_initialized || g_n_synapses == 0) return 0;
    
    int threads = 256;
    int blocks = (g_n_synapses + threads - 1) / threads;
    
    launch_stdp(d_weights, reward, (int)g_n_synapses, blocks, threads, stream_stdp);
    
    return 0;
}

int hec_cleanup() {
    if (!g_initialized) return 0;
    
    if (stream_stdp) cudaStreamDestroy(stream_stdp);
    
    cudaFree(d_neurons_v);
    cudaFree(d_neurons_u);
    cudaFree(d_neurons_i);
    cudaFree(d_motor);
    cudaFree(d_sensory);
    if (d_weights) cudaFree(d_weights);
    
    g_initialized = false;
    printf("[HEC] 清理完成\n");
    return 0;
}

int hec_status(char* buf, size_t bufsize) {
    if (!g_initialized) {
        snprintf(buf, bufsize, "未初始化");
        return -1;
    }
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    snprintf(buf, bufsize,
        "HEC v2.1 | N:%zu | Mem:%.1fGB",
        g_n_neurons, free_mem / 1e9
    );
    return 0;
}

} // extern "C"
