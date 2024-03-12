echo "start dynamic"
python run_benchmark.py group_bench -c /home/cdhernandez/local/benchmark/userbenchmark/group_bench/configs/torch_ao.yaml

# echo "start int8 weight only"
# python run_benchmark.py dynamo --bfloat16 --inductor --performance --inference --quantization int8weightonly --inductor-compile-mode max-autotune --tag int8weightonly

# one python run.py BERT_pytorch -d cuda --precision bf16 --torchdynamo inductor --inductor-compile-mode max-autotun
