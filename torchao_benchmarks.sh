python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int8dynamic --inductor-compile-mode max-autotune --tag int8dynamic-epi --filter BERT_pytorch --batch_size 128
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --inductor-compile-mode max-autotune --tag baseline --filter BERT_pytorch --batch_size 128


# echo "start dynamic"
# python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int8dynamic --inductor-compile-mode max-autotune --custom_find_batchsize --tag int8dynamic
# echo "start baseline"
# python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --inductor-compile-mode max-autotune --custom_find_batchsize --tag baseline

# echo "start int8 weight only batchsize 1"
# python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int8weightonly --inductor-compile-mode max-autotune --batch_size 1 --tag int8weightonly-bs1
# echo "start int4 weight only batchsize 1"
# python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int4weightonly --inductor-compile-mode max-autotune --batch_size 1 --tag int4weightonly-bs1
# echo "start baseline batchsize 1"
# python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --inductor-compile-mode max-autotune --batch_size 1 --tag baseline-bs1

# echo "start accuracy"
# python run_benchmark.py dynamo --bfloat16 --inductor --inference --inductor-compile-mode max-autotune --batch_size 1 --tag acc --accuracy
