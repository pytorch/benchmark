debug_arg=""
if [ $# -gt 1 ]; then
        if [ "$1" == "--debug" ]; then
                debug_arg="--debug $2"
        fi
fi
python3.8 train.py --data coco128.data --img 416 --batch 8 --nosave --notest --epochs 10 --weights '' $debug_arg
