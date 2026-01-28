#!/bin/bash
# 消融实验启动脚本
# 功能：自动停止旧进程，启动新的并行实验

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
PARALLEL=${PARALLEL:-4}
GROUPS=${GROUPS:-"A B C D E"}
TIMESTEPS=${TIMESTEPS:-""}
FORCE_RESTART=${FORCE_RESTART:-false}

# 显示帮助
show_help() {
    echo "消融实验启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -p, --parallel N    并行组数 (默认: 4)"
    echo "  -g, --groups GROUPS 运行的组 (默认: A B C D E)"
    echo "  -t, --timesteps N   训练步数"
    echo "  -f, --force         强制重启 (停止现有进程)"
    echo "  -h, --help          显示帮助"
    echo ""
    echo "示例:"
    echo "  $0                          # 默认运行所有组，4路并行"
    echo "  $0 -p 5 -g 'A B'            # 5路并行，只运行A、B组"
    echo "  $0 -f                       # 强制停止现有进程并重启"
    echo "  PARALLEL=2 $0               # 环境变量方式设置并行度"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        -g|--groups)
            GROUPS="$2"
            shift 2
            ;;
        -t|--timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_RESTART=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 切换到项目根目录
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}消融实验启动脚本${NC}"
echo -e "${GREEN}================================${NC}"
echo "项目目录: $PROJECT_ROOT"
echo "并行度: $PARALLEL"
echo "实验组: $GROUPS"

# 检查是否有正在运行的消融实验
check_running() {
    pgrep -f "run_ablation.py" > /dev/null 2>&1
    return $?
}

# 停止现有进程
stop_existing() {
    echo -e "${YELLOW}检测到正在运行的消融实验，正在停止...${NC}"

    # 获取主进程PID
    PIDS=$(pgrep -f "run_ablation.py" || true)

    if [ -n "$PIDS" ]; then
        echo "停止进程: $PIDS"
        # 发送SIGTERM，优雅停止
        kill $PIDS 2>/dev/null || true

        # 等待最多10秒
        for i in {1..10}; do
            if ! check_running; then
                echo -e "${GREEN}进程已停止${NC}"
                return 0
            fi
            sleep 1
        done

        # 如果还在运行，强制杀死
        echo -e "${YELLOW}进程未响应，强制终止...${NC}"
        kill -9 $PIDS 2>/dev/null || true
        sleep 1
    fi
}

# 检查并处理现有进程
if check_running; then
    if [ "$FORCE_RESTART" = true ]; then
        stop_existing
    else
        echo -e "${YELLOW}警告: 检测到正在运行的消融实验${NC}"
        echo ""
        ps aux | grep "run_ablation.py" | grep -v grep || true
        echo ""
        echo -e "使用 ${GREEN}-f${NC} 或 ${GREEN}--force${NC} 选项强制重启"
        echo -e "或运行 ${GREEN}pkill -f run_ablation.py${NC} 手动停止"
        exit 1
    fi
fi

# 创建日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ablation_${TIMESTAMP}.log"

# 构建命令
CMD="python examples/run_ablation.py --parallel $PARALLEL"

if [ -n "$GROUPS" ]; then
    CMD="$CMD --groups $GROUPS"
fi

if [ -n "$TIMESTEPS" ]; then
    CMD="$CMD --timesteps $TIMESTEPS"
fi

echo ""
echo -e "${GREEN}启动命令:${NC} $CMD"
echo -e "${GREEN}日志文件:${NC} $LOG_FILE"
echo ""

# 运行实验
echo -e "${GREEN}开始运行消融实验...${NC}"
echo "按 Ctrl+C 可中断实验"
echo ""

# 前台运行，输出同时保存到日志
$CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo -e "${GREEN}实验完成!${NC}"
echo -e "日志保存在: $LOG_FILE"
echo -e "结果目录: outputs/ablation/"
