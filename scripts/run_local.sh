#!/bin/bash

pytest test_bench.py -k "test_train[alexnet-mps-eager or \
                         test_train[dcgan-mps-eager or \
                         test_train[hf_Bert-mps-eager or \
                         test_train[maml-mps-eager or \
                         test_train[mnasnet1_0-mps-eager or \
                         test_train[mobilenet_v2-mps-eager or \
                         test_train[pytorch_unet-mps-eager or \
                         test_train[resnet18-mps-eager or \
                         test_train[resnet50-mps-eager or \
                         test_train[resnext50_32x4d-mps-eager or \
                         test_train[shufflenet_v2_x1_0-mps-eager or \
                         test_train[timm_efficientnet-mps-eager or \
                         test_train[timm_nfnet-mps-eager or \
                         test_train[timm_regnet-mps-eager or \
                         test_train[timm_resnest-mps-eager or \
                         test_train[timm_vision_transformer-mps-eager or \
                         test_train[timm_vovnet-mps-eager or \
                         test_train[soft_actor_critic-mps-eager or \
                         test_train[hf_DistilBert-mps-eager or \
                         test_train[hf_Bart-mps-eager or \
                         test_train[hf_Albert-mps-eager or \
                         test_train[hf_GPT2-mps-eager or \
                         test_train[lennard_jones-mps-eager or \
                         test_train[pytorch_stargan-mps-eager or \
                         test_train[pytorch_struct-mps-eager or \
                         test_train[timm_vision_transformer_large-mps-eager or \
                         test_train[functorch_dp_cifar10-mps-eager or \
                         test_train[squeezenet1_1-mps-eager or \
                         test_train[hf_T5_base-mps-eager or \
                         test_train[hf_T5_large-mps-eager or \
                         test_train[densenet121-mps-eager or \
                         test_train[vgg16-mps-eager " --ignore_machine_config --sweep_bs 'all' --benchmark-save="."
