echo "start dynamic"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int8dynamic --inductor-compile-mode max-autotune
echo "start int8 weight only"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int8weightonly --inductor-compile-mode max-autotune
echo "start int4 weight only"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int4weightonly --inductor-compile-mode max-autotune
echo "start baseline"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --inductor-compile-mode max-autotune


echo "start int4 weight only batchsize 1"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int4weightonly --inductor-compile-mode max-autotune --batch_size 1
echo "start baseline batchsize 1"
python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --inductor-compile-mode max-autotune --batch_size 1
