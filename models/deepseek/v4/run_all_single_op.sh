#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTO_LIB_DIR="$(cd "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/run_results"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/run_results.txt"
echo "=== DeepSeek V4 单算子运行结果 ===" > "$LOG_FILE"
echo "运行时间: $(date)" >> "$LOG_FILE"
echo "PYTHONPATH: $PTO_LIB_DIR" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

declare -A RESULTS
declare -A ERROR_MESSAGES

SKIP_FILES=(
    "config.py"
    "decode_fwd.py"
    "decode_layer.py"
    "lm_head.py"
    "moe.py"
    "prefill_fwd.py"
    "prefill_layer.py"
    "prefill_mtp.py"
)

PY_FILES=$(ls "$SCRIPT_DIR"/*.py)

TOTAL=0
SUCCESS=0
FAILED=0

for FILE in $PY_FILES; do
    BASENAME=$(basename "$FILE")
    
    SKIP=0
    for SKIP_FILE in "${SKIP_FILES[@]}"; do
        if [ "$BASENAME" = "$SKIP_FILE" ]; then
            SKIP=1
            break
        fi
    done
    
    if [ $SKIP -eq 1 ]; then
        echo "[跳过] $BASENAME (分布式/多卡)" | tee -a "$LOG_FILE"
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    echo "" >> "$LOG_FILE"
    echo "--- 算子: $BASENAME ---" >> "$LOG_FILE"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_LOG="$OUTPUT_DIR/${BASENAME%.py}_${TIMESTAMP}.log"
    
    echo "[运行中] $BASENAME"
    echo "执行命令: task-submit --timeout 600 --max-time 600 --device 0 --run \"cd $SCRIPT_DIR && PYTHONPATH=$PTO_LIB_DIR python $BASENAME -p a5 -d 0\"" >> "$LOG_FILE"
    
    task-submit --timeout 600 --max-time 600 --device 0 --run "cd $SCRIPT_DIR && PYTHONPATH=$PTO_LIB_DIR python $BASENAME -p a5 -d 0" > "$OUTPUT_LOG" 2>&1
    
    if [ $? -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
        RESULTS["$BASENAME"]="成功"
        echo "[成功] $BASENAME" | tee -a "$LOG_FILE"
        echo "结果: 成功" >> "$LOG_FILE"
        tail -10 "$OUTPUT_LOG" >> "$LOG_FILE"
    else
        FAILED=$((FAILED + 1))
        RESULTS["$BASENAME"]="失败"
        ERROR_MSG=$(cat "$OUTPUT_LOG" | tail -30)
        ERROR_MESSAGES["$BASENAME"]="$ERROR_MSG"
        echo "[失败] $BASENAME" | tee -a "$LOG_FILE"
        echo "结果: 失败" >> "$LOG_FILE"
        echo "错误信息:" >> "$LOG_FILE"
        echo "$ERROR_MSG" >> "$LOG_FILE"
    fi
done

echo "" >> "$LOG_FILE"
echo "=== 运行总结 ===" >> "$LOG_FILE"
echo "总算子数: $TOTAL" >> "$LOG_FILE"
echo "成功: $SUCCESS" >> "$LOG_FILE"
echo "失败: $FAILED" >> "$LOG_FILE"

echo "" >> "$LOG_FILE"
echo "=== 详细结果列表 ===" >> "$LOG_FILE"
for FILE in "${!RESULTS[@]}"; do
    echo "$FILE: ${RESULTS[$FILE]}" >> "$LOG_FILE"
done

echo ""
echo "=== 运行完成 ==="
echo "总算子数: $TOTAL"
echo "成功: $SUCCESS"
echo "失败: $FAILED"
echo "详细结果已保存至: $LOG_FILE"

exit 0