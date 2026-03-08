#!/bin/bash
# 明晨自动化对比脚本: 单核 vs 多核

echo "═══════════════════════════════════════════════════════════════"
echo "📊 单核 vs 多核 性能对比分析"
echo "═══════════════════════════════════════════════════════════════"
echo ""

cd /home/admin/agl_mwe

# 停止当前单核B组（假设已完成6小时测试）
echo "停止当前B组..."
kill $(cat logs/CONTROL_BURN.pid 2>/dev/null) 2>/dev/null
sleep 2

echo "启动多核D组（64线程）..."
./run_control_burn_multicore.sh
sleep 10

echo ""
echo "✅ 多核测试已启动"
echo "   运行时间: 1小时（快速对比）"
echo "   对比指标: 频率稳定性、CPU占用、吞吐量"
echo ""
echo "预计结果:"
echo "   单核: ~0.2% CPU, 100Hz（10ms周期内）"
echo "   多核: ~50% CPU (64核), 100Hz（轻松达到）"
echo "   理论吞吐量提升: 64x（如果瓶颈在计算）"
echo "   实际提升: 取决于内存带宽"
