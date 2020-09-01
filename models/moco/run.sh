debug_arg=""
if [ $# -gt 1 ]; then
        if [ "$1" == "--debug" ]; then
                debug_arg="-d $2"
        fi
fi
CUDA_VISIBLE_DEVICES=0 python main_moco.py   -a resnet50   --lr 0.03   --batch-size 32   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  --fake_data --epochs 2 --seed 1058467 $debug_arg dummy
