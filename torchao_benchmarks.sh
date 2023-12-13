echo "start dynamic"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int8dynamic --inductor-compile-mode max-autotune --tag int8dynamic
echo "start int8 weight only"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int8weightonly --inductor-compile-mode max-autotune --tag int8weightonly
echo "start int4 weight only"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int4weightonly --inductor-compile-mode max-autotune --tag int4weightonly
echo "start baseline"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --inductor-compile-mode max-autotune --tag baseline

echo "start int8 weight only batchsize 1"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int8weightonly --inductor-compile-mode max-autotune --batch_size 1 --tag int8weightonly-bs1
echo "start int4 weight only batchsize 1"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int4weightonly --inductor-compile-mode max-autotune --batch_size 1 --tag int4weightonly-bs1
echo "start baseline batchsize 1"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --inductor-compile-mode max-autotune --batch_size 1 --tag baseline-bs1

echo "start dynamic batchsize 32"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int8dynamic --inductor-compile-mode max-autotune --batch_size 32 --tag int8dynamic-bs32
echo "start baseline batchsize 32"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --inductor-compile-mode max-autotune --batch_size 32 --tag baseline-bs32

echo "start accuracy"
python run_benchmark.py dynamo --bfloat16 --inductor --inference --quantization int8dynamic --inductor-compile-mode max-autotune --batch_size 1 --tag int8dynamic-bs1-acc --accuracy
python run_benchmark.py dynamo --bfloat16 --inductor --inference --quantization int8weightonly --inductor-compile-mode max-autotune --batch_size 1 --tag int8weightonly-bs1-acc --accuracy
python run_benchmark.py dynamo --bfloat16 --inductor --inference --quantization int4weightonly --inductor-compile-mode max-autotune --batch_size 1 --tag int4weightonly-bs1-acc --accuracy
