#!/bin/bash
# 多核CPU Burn Test启动脚本

echo "═══════════════════════════════════════════════════════════════"
echo "🧠 Atlas-HEC L6 Burn Test - 多核CPU版本"
echo "═══════════════════════════════════════════════════════════════"

# 设置Rayon线程池
export RAYON_NUM_THREADS=64  # 使用64核（留一半给系统和其他进程）
export RUST_BACKTRACE=1

cd /home/admin/agl_mwe

# 编译多核版本
echo "🔨 编译多核版本..."
cargo build --release --bin control_burn_multicore 2>&1 | tee logs/build_multicore.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ 编译失败，查看 logs/build_multicore.log"
    exit 1
fi

echo "✅ 编译成功"
echo ""
echo "⚡ 启动多核Burn Test..."
echo "   线程数: $RAYON_NUM_THREADS"
echo ""

# 创建带时间戳的日志
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/control_burn_multicore_${TIMESTAMP}.log"
CSVFILE="logs/control_burn_multicore_${TIMESTAMP}.csv"

echo "timestamp,step,hz,progress,mem_mb,spikes" > "$CSVFILE"

# 后台运行
timeout 7h ./target/release/control_burn_multicore 2>&1 | tee "$LOGFILE" &
PID=$!
echo $PID > logs/CONTROL_MULTICORE.pid

echo "✅ 多核测试已启动"
echo "   PID: $PID"
echo "   日志: $LOGFILE"
echo "   CSV:  $CSVFILE"
echo ""
echo "查看状态: tail -f $LOGFILE"
echo "停止测试: kill \$(cat logs/CONTROL_MULTICORE.pid)"
