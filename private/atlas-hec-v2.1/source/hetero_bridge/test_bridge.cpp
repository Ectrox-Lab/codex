#include <iostream>
#include <cstring>

extern "C" {
    int hec_init(size_t n_neurons, size_t n_synapses, int gpu_id);
    int hec_cleanup();
    int hec_status(char* buf, size_t bufsize);
}

int main() {
    std::cout << "⚡ Atlas-HEC Bridge Test\n" << std::endl;
    
    // 测试初始化
    int ret = hec_init(10000, 10000000, 0);
    if (ret != 0) {
        std::cerr << "初始化失败: " << ret << std::endl;
        return 1;
    }
    
    // 获取状态
    char buf[256];
    hec_status(buf, sizeof(buf));
    std::cout << "状态: " << buf << std::endl;
    
    // 清理
    hec_cleanup();
    
    std::cout << "\n✅ Bridge测试通过" << std::endl;
    return 0;
}
