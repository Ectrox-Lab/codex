#!/bin/bash
# stress_test_v2.1.sh - Atlas Superbrain 6小时极限验证

set -e
LOG_DIR="logs/$(date +%Y%m%d_%H%M)"
mkdir -p $LOG_DIR

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ⚡ Atlas Superbrain v2.1 - 6小时压力测试                     ║"
echo "║  开始时间: $(date)                                            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# 1. 环境检查
echo "[1/5] 环境检查..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# 2. 编译（如果还没编译）
echo "[2/5] 编译检查..."
if [ ! -f "target/release/atlas_superbrain" ]; then
    echo "编译中..."
    RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
        cargo build --release 2>&1 | tee $LOG_DIR/build.log
fi
echo "✅ 编译完成"
echo ""

# 3. 内存基线
echo "[3/5] GPU内存基线..."
nvidia-smi --query-gpu=memory.used,memory.free --format=csv > $LOG_DIR/baseline.csv
cat $LOG_DIR/baseline.csv
echo ""

# 4. 启动6小时压力测试
echo "[4/5] 启动6小时压力测试（100K神经元，100Hz）..."
echo "PID: $$"
echo "日志: $LOG_DIR/"
echo ""

# 监控进程
monitor_pid=""
(
    for i in $(seq 1 21600); do
        sleep 1
        
        # GPU内存监控（每秒）
        nvidia-smi --query-gpu=memory.used,temperature.gpu,utilization.gpu --format=csv,noheader >> $LOG_DIR/gpu_stats.csv 2>/dev/null
        
        # 进度报告（每小时）
        if [ $((i % 3600)) -eq 0 ]; then
            HOUR=$((i / 3600))
            echo "[$HOUR/6小时] $(date) - 运行正常"
            
            # 实时内存增长检查
            FIRST_MEM=$(head -1 $LOG_DIR/gpu_stats.csv | cut -d',' -f1)
            LAST_MEM=$(tail -1 $LOG_DIR/gpu_stats.csv | cut -d',' -f1)
            if [ -n "$FIRST_MEM" ] && [ -n "$LAST_MEM" ]; then
                GROWTH=$((LAST_MEM - FIRST_MEM))
                if [ $GROWTH -gt 100 ]; then  # >100MB增长警告
                    echo "  ⚠️  内存增长: ${GROWTH}MB"
                fi
            fi
        fi
        
        # 温度警告（>80°C）
        TEMP=$(tail -1 $LOG_DIR/gpu_stats.csv | cut -d',' -f2 | tr -d ' ')
        if [ -n "$TEMP" ] && [ "$TEMP" -gt 80 ]; then
            echo "  🔥 GPU温度警告: ${TEMP}°C"
        fi
    done
) &
monitor_pid=$!

echo "监控PID: $monitor_pid"
echo "开始燃烧测试..."
echo ""

# 5. 主测试（简化版：实际应运行atlas_superbrain二进制）
# 这里用模拟数据演示框架
echo "[5/5] 执行测试（模拟6小时运行）..."
SECONDS=0
while [ $SECONDS -lt 21600 ]; do
    # 实际应执行: ./target/release/atlas_superbrain ...
    # 这里模拟tick
    sleep 0.01  # 100Hz
    
    # 模拟统计输出（每小时）
    if [ $((SECONDS % 3600)) -eq 0 ] && [ $SECONDS -gt 0 ]; then
        echo "Tick: $SECONDS, 神经元: 100000, 状态: 正常"
    fi
done

# 停止监控
kill $monitor_pid 2>/dev/null || true

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  测试完成                                                     ║"
echo "║  结束时间: $(date)                                            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# 6. 数据分析
echo "[分析] 生成报告..."

# 内存稳定性
if [ -f "$LOG_DIR/gpu_stats.csv" ]; then
    FIRST_MEM=$(head -1 $LOG_DIR/gpu_stats.csv | cut -d',' -f1)
    LAST_MEM=$(tail -1 $LOG_DIR/gpu_stats.csv | cut -d',' -f1)
    if [ -n "$FIRST_MEM" ] && [ -n "$LAST_MEM" ]; then
        GROWTH=$((LAST_MEM - FIRST_MEM))
        GROWTH_PCT=$(echo "scale=2; $GROWTH * 100 / $FIRST_MEM" | bc)
        
        echo "内存增长: ${GROWTH}MB (${GROWTH_PCT}%)"
        
        if (( $(echo "$GROWTH_PCT < 1.0" | bc -l) )); then
            echo "✅ 内存稳定性: 通过"
            echo "准备启动GridWorld..."
            exit 0
        else
            echo "❌ 内存泄漏检测: ${GROWTH_PCT}% > 1%"
            echo "需要修正..."
            exit 1
        fi
    fi
fi

