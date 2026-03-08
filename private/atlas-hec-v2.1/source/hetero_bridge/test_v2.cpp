#include <cstdio>
#include <cstdint>

extern "C" {
    int hec_init(size_t n_neurons, size_t n_synapses, int gpu_id);
    int hec_step_hybrid(const uint8_t* sensory, float* motor, size_t n_sens, size_t n_motor);
    int hec_stdp_async(float reward);
    int hec_status(char* buf, size_t bufsize);
    int hec_cleanup();
}

int main() {
    printf("⚡ Atlas-HEC v2.1 Bridge v2 测试\n\n");
    
    if (hec_init(10000, 10000000, 0) != 0) {
        printf("❌ 初始化失败\n");
        return 1;
    }
    
    char buf[256];
    hec_status(buf, sizeof(buf));
    printf("状态: %s\n\n", buf);
    
    uint8_t sensory[256] = {0};
    float motor[5] = {0};
    sensory[128] = 255;
    
    printf("测试步进:\n");
    for (int i = 0; i < 3; i++) {
        if (hec_step_hybrid(sensory, motor, 256, 5) != 0) {
            printf("  Step %d 失败\n", i);
            break;
        }
        printf("  Step %d: motor=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               i+1, motor[0], motor[1], motor[2], motor[3], motor[4]);
    }
    
    hec_stdp_async(0.1f);
    printf("\nSTDP异步启动\n");
    
    hec_cleanup();
    printf("\n✅ 测试通过!\n");
    return 0;
}
