debug_arg=""
if [ $# -gt 1 ]; then
  if [ "$1" == "--debug" ]; then
    debug_arg="--debug $2"
  fi
fi
python main.py --mode train --dataset CelebA --image_size 128 --c_dim 2 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results --selected_attrs Male Young  --use_tensorboard False --num_iters 30 --should_script True --deterministic True $debug_arg
