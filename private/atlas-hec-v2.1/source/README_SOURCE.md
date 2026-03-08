# 🔐 Atlas-HEC v2.1 核心源代码

> ⚠️ **机密**: 核心算法和实现细节

---

## 📁 文件结构

```
source/
├── Cargo.toml              # Rust项目配置
├── Cargo.lock              # 依赖锁定
├── README_SOURCE.md        # 本文件
├── src/                    # Rust源代码
│   ├── lib.rs              # 库入口
│   ├── main_5k.rs          # 5K神经元版本主程序
│   ├── main_debug.rs       # 调试版本
│   ├── main_quick.rs       # 快速测试版本
│   ├── hec_ffi.rs          # FFI绑定
│   ├── hec_bridge_dl.rs    # 动态加载桥接
│   ├── atlas_cuda_bridge.rs # CUDA桥接
│   ├── atlas_backup/       # 备份模块
│   │   ├── mod.rs
│   │   └── gpu_core.rs
│   ├── gridworld/          # GridWorld环境
│   │   └── mod.rs
│   ├── mlp/                # 多层感知机
│   │   └── mod.rs
│   ├── sensory/            # 感知编码
│   │   ├── mod.rs
│   │   └── mnist_encoder.rs
│   ├── mnist/              # MNIST相关
│   │   └── loader.rs
│   ├── circadian/          # 昼夜节律
│   │   └── ctmc.rs
│   ├── biomimetic/         # 生物启发模块
│   │   └── metabolism.rs
│   ├── telemetry/          # 遥测
│   │   └── reflector.rs
│   └── bin/                # 可执行文件
│       ├── atlas_burn.rs
│       ├── atlas_burn_real.rs
│       ├── gridworld_fast_check.rs
│       └── mnist_certification.rs
├── hetero_bridge/          # CUDA C++桥接库
│   ├── Makefile
│   ├── hec_bridge_v2.cpp   # 桥接实现
│   ├── bridge_host.cpp     # 主机端代码
│   ├── kernels.cu          # CUDA内核
│   ├── atlas_kernels.ptx   # PTX汇编
│   ├── libhec_bridge_v2.so # 编译后的共享库
│   └── test_v2.cpp         # 测试代码
└── scripts/                # 构建和测试脚本
    ├── burn_test_real.sh
    ├── run_6hour_hec_burn.sh
    ├── stress_test_v2.1.sh
    ├── run_control_burn_multicore.sh
    └── compare_single_vs_multicore.sh
```

---

## 🔑 核心组件

### 1. Izhikevich神经元 (src/lib.rs)
```rust
pub struct IzhikevichNeuron {
    pub v: f64,  // 膜电位
    pub u: f64,  // 恢复变量
    pub a: f64,  // 时间尺度
    pub b: f64,  // 恢复敏感度
    pub c: f64,  // 重置电位
    pub d: f64,  // 恢复偏移
}
```

### 2. CUDA桥接 (hetero_bridge/)
- `hec_bridge_v2.cpp`: Rust-FFI接口
- `kernels.cu`: Izhikevich内核, STDP内核
- `libhec_bridge_v2.so`: 编译后的共享库

### 3. GridWorld环境 (src/gridworld/)
- 零分配实现
- 1000步生存验证
- 与SNN闭环交互

### 4. STDP学习
- 时序依赖可塑性
- 奖励调节
- 在线学习

---

## 🚀 构建说明

```bash
# 1. 构建CUDA桥接
cd hetero_bridge
make clean && make

# 2. 设置库路径
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH

# 3. 构建Rust项目
cargo build --release

# 4. 运行测试
./target/release/atlas_burn
```

---

## 📊 关键参数

| 参数 | 值 | 说明 |
|------|------|------|
| 神经元数量 | 10,000 | 隐藏层 |
| 连接数 | 100/神经元 | 稀疏连接 |
| 时间步长 | 10ms | 100Hz更新 |
| 学习率 | 0.001 | STDP |
| CUDA版本 | 11.5 | sm_86 |

---

*Atlas-HEC Core Source - Ectrox Lab Confidential*
