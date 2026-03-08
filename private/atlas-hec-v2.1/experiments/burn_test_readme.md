# 燃烧测试脚本

## 启动命令

```bash
# A组: GPU纯加速
./run_real_burn.sh

# B组: CPU单核基线
./run_control_burn.sh

# C组: 异构CPU+GPU
./run_hetero_burn.sh

# D组: CPU多核64线程（待启动）
RAYON_NUM_THREADS=64 ./run_control_burn_multicore.sh
```

## 监控命令

```bash
# 查看实时日志
tail -f logs/*BURN*.log

# 查看GPU状态
nvidia-smi -l 1

# 查看进程
ps aux | grep burn
```

## 停止测试

```bash
kill $(cat logs/*BURN.pid)
```
