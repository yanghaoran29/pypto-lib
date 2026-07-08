#!/bin/bash

# 判断是否传入脚本文件名参数
if [ $# -ne 1 ]; then
    echo "用法：$0 xxx.py"
    echo "示例：$0 test.py"
    exit 1
fi

# 提取传入的py文件名
PY_FILE="$1"

# 固定路径
BASE_DIR="/home/pyptouser/yanghaoran/Desktop/pypto-lib/models/deepseek/v4/MXFP8-MXFP4"
FULL_PY_PATH="${BASE_DIR}/${PY_FILE}"

# 校验文件是否存在
if [ ! -f "${FULL_PY_PATH}" ]; then
    echo "错误：文件不存在 ${FULL_PY_PATH}"
    exit 1
fi

# 执行提交命令
task-submit --device 0 --run "python ${FULL_PY_PATH} -p a5"