CUDA_VISIBLE_DEVICES=0,1 bert -c data/corpus.small -v data/vocab.small -o bert.model $@
