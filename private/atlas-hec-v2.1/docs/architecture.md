# 🏗️ Atlas-HEC v2.1 架构设计

## 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Atlas-HEC v2.1                           │
├─────────────────────────────────────────────────────────────┤
│  CPU Layer (控制逻辑)                                        │
│  ├── GridWorld Environment                                   │
│  ├── Curiosity Engine (探索奖励)                             │
│  └── CTMC Rhythm (昼夜节律)                                  │
├─────────────────────────────────────────────────────────────┤
│  GPU Layer (脉冲神经网络)                                     │
│  ├── Izhikevich Neurons (10K)                               │
│  ├── STDP Learning (时序可塑性)                              │
│  └── CUDA Kernels (izhikevich_kernel, stdp_kernel)          │
├─────────────────────────────────────────────────────────────┤
│  Bridge Layer (FFI通信)                                      │
│  ├── libhec_bridge_v2.so                                     │
│  └── Unified Memory                                          │
└─────────────────────────────────────────────────────────────┘
```

## 神经元模型

### Izhikevich模型
```rust
v' = 0.04v² + 5v + 140 - u + I
u' = a(bv - u)

if v ≥ 30mV:
    v = c
    u = u + d
```

参数:
- a: 时间尺度 (0.02)
- b: 恢复敏感度 (0.2)
- c: 重置电位 (-65mV)
- d: 恢复偏移 (8)

## 学习规则

### STDP (Spike-Timing Dependent Plasticity)
```
Δw = A⁺ exp(-Δt/τ⁺)  if Δt > 0 (pre在post前)
Δw = A⁻ exp(Δt/τ⁻)   if Δt < 0 (pre在post后)
```

### 奖励调节
```rust
reward = if prediction == target { 1.0 } else { -0.5 }
weight += learning_rate * reward * STDP_window
```

## 异构调度

```
时间步 (10ms):
┌──────────┐    ┌──────────┐    ┌──────────┐
│ CPU:     │ →  │ GPU:     │ →  │ CPU:     │
│ 感知环境  │    │ SNN计算  │    │ 决策执行  │
│ GridWorld│    │ Izhikevich│   │ 动作输出  │
│ 好奇心计算│    │ STDP更新 │    │ 奖励反馈  │
└──────────┘    └──────────┘    └──────────┘
      ↑                              │
      └──────────────────────────────┘
              (闭环反馈)
```

## 内存布局

```
GPU Memory (386MiB):
├── 神经元状态: 10K × (v, u, I) × f64
├── 突触权重: 10K × 100 connections × f32
├── STDP窗口: 10K × time_window
└── CUDA上下文: ~100MiB
```

## 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 单步延迟 | 2.3ms | 远低于10ms预算 |
| 频率 | 97-99Hz | 接近100Hz目标 |
| GPU利用率 | 15% | 轻量负载，预留扩展 |
| 内存带宽 | 2GB/s | 远低于HBM极限 |

---

*Architecture v2.1 - Stable*
