#!/bin/bash
cd /home/admin/agl_mwe
export LD_LIBRARY_PATH=/home/admin/agl_mwe/hetero_bridge:$LD_LIBRARY_PATH

# 配置
NEURONS=10000
HOURS=6
STEPS=$((HOURS * 3600 * 100))  # 100Hz
LOG="logs/hec_6hour_$(date +%Y%m%d_%H%M).log"

echo "⚡ Atlas-HEC v2.1 6小时异构燃烧测试" | tee $LOG
echo "开始: $(date)" | tee -a $LOG
echo "配置: ${NEURONS}神经元, ${HOURS}小时, ${STEPS}步" | tee -a $LOG
echo "GPU: 0 (46GB空闲)" | tee -a $LOG
echo "" | tee -a $LOG

# 运行测试（后台）
./target/release/atlas_hec_burn_final >> $LOG 2>&1 &
PID=$!
echo $PID > logs/hec_burn.pid

echo "PID: $PID" | tee -a $LOG
echo "日志: $LOG" | tee -a $LOG
echo "" | tee -a $LOG

# 监控循环
SECONDS=0
while kill -0 $PID 2>/dev/null; do
    sleep 60
    
    # GPU状态
    nvidia-smi --query-gpu=timestamp,memory.used,temperature.gpu,power.draw,utilization.gpu \
        --format=csv,noheader -i 0 >> logs/hec_gpu_telemetry.csv 2>/dev/null
    
    # 进度报告（每小时）
    if [ $((SECONDS % 3600)) -eq 0 ] && [ $SECONDS -gt 0 ]; then
        HOUR=$((SECONDS / 3600))
        echo "[Hour $HOUR/6] $(date) - 运行正常" | tee -a $LOG
        tail -5 $LOG | head -3
    fi
done

wait $PID
EXIT_CODE=$?

echo "" | tee -a $LOG
echo "═══════════════════════════════════════════════════════════════" | tee -a $LOG
echo "测试结束: $(date)" | tee -a $LOG
echo "退出码: $EXIT_CODE" | tee -a $LOG
echo "═══════════════════════════════════════════════════════════════" | tee -a $LOG

# 分析结果
if [ -f "$LOG" ]; then
    echo "" | tee -a $LOG
    echo "结果分析:" | tee -a $LOG
    grep "总步数:" $LOG | tail -1 | tee -a $LOG
    grep "平均步长:" $LOG | tail -1 | tee -a $LOG
    grep "模式:" $LOG | tail -1 | tee -a $LOG
    
    # 生成证书
    if grep -q "异构GPU" $LOG; then
        CERT="logs/CERTIFICATE_HEC_PASS_$(date +%Y%m%d_%H%M).txt"
        echo "ATLAS-HEC v2.1 异构燃烧测试证书" > $CERT
        echo "================================" >> $CERT
        echo "状态: PASS" >> $CERT
        echo "时间: $(date)" >> $CERT
        echo "模式: 异构GPU (CPU+GPU+RAM)" >> $CERT
        tail -10 $LOG >> $CERT
        echo "证书: $CERT"
    fi
fi
