# PyTorch Benchmark Score V1

This file describes how we generate the PyTorch Benchmark Score Version 1. The
goal is to help users and developers understand the score and be able to
reproduce it.

V1 uses the same hardware environment as [V0](../v0/config-v0.md), but it covers
far more models and test configurations.

## Requirements

The V1 benchmark suite uses an experimental JIT feature,
[optimize_for_inference](https://github.com/pytorch/pytorch/pull/58193),
introduced on May 22, 2021. Therefore, it can't run on earlier versions of
PyTorch.

## Coverage

The V1 suite covers 50 models from popular machine learning domains.
The complete list of models is as follows:

| Model name                             | Category                |
|----------------------------------------|-------------------------|
| BERT\_pytorch                          | NLP                     |
| Background\_Matting                    | COMPUTER VISION         |
| LearningToPaint                        | REINFORCEMENT LEARNING  |
| alexnet                                | COMPUTER VISION         |
| attention\_is\_all\_you\_need\_pytorch | NLP                     |
| demucs                                 | OTHER                   |
| densenet121                            | COMPUTER VISION         |
| dlrm                                   | RECOMMENDATION          |
| drq                                    | REINFORCEMENT LEARNING  |
| fastNLP                                | NLP                     |
| hf\_Albert                             | NLP                     |
| hf\_Bert                               | NLP                     |
| hf\_BigBird                            | NLP                     |
| hf\_DistilBert                         | NLP                     |
| hf\_GPT2                               | NLP                     |
| hf\_Longformer                         | NLP                     |
| hf\_T5                                 | NLP                     |
| maml                                   | OTHER                   |
| maml\_omniglot                         | OTHER                   |
| mnasnet1\_0                            | COMPUTER VISION         |
| mobilenet\_v2                          | COMPUTER VISION         |
| mobilenet\_v3\_large                   | COMPUTER VISION         |
| moco                                   | OTHER                   |
| nvidia\_deeprecommender                | RECOMMENDATION          |
| opacus\_cifar10                        | OTHER                   |
| pyhpc\_equation\_of\_state             | OTHER                   |
| pyhpc\_isoneutral\_mixing              | OTHER                   |
| pytorch\_CycleGAN\_and\_pix2pix        | COMPUTER VISION         |
| pytorch\_stargan                       | COMPUTER VISION         |
| pytorch\_struct                        | OTHER                   |
| resnet18                               | COMPUTER VISION         |
| resnet50                               | COMPUTER VISION         |
| resnet50\_quantized\_qat               | COMPUTER VISION         |
| resnext50\_32x4d                       | COMPUTER VISION         |
| shufflenet\_v2\_x1\_0                  | COMPUTER VISION         |
| soft\_actor\_critic                    | REINFORCEMENT LEAERNING |
| speech\_transformer                    | SPEECH                  |
| squeezenet1\_1                         | COMPUTER VISION         |
| timm\_efficientnet                     | COMPUTER VISION         |
| timm\_nfnet                            | COMPUTER VISION         |
| timm\_regnet                           | COMPUTER VISION         |
| timm\_resnest                          | COMPUTER VISION         |
| timm\_vision\_transformer              | COMPUTER VISION         |
| timm\_vovnet                           | COMPUTER VISION         |
| tts\_angular                           | SPEECH                  |
| vgg16                                  | COMPUTER VISION         |
| yolov3                                 | COMPUTER VISION         |

## Reference Config YAML

The reference config YAML file is stored [here](config-v1.yaml). It is generated
by repeated runs of the same benchmark setting on pytorch v1.10.0.dev20210612,
torchtext 0.10.0.dev20210612, and torchvision 0.11.0.dev20210612. We choose the
earliest PyTorch nightly version that has a stable implementation of the
`optimize_for_inference` feature. We then picked a random execution of the
repeated V1 benchmark runs as the reference execution, and summarize its
performance metrics in the reference config YAML.

We have also manually verified that the maximum variance of any single test in
the V1 suite is smaller than 7%. In the V1 nightly CI job, we raise signal if
any tests performance metric changes over the 7% threshold, or the overall score
number changes over 1% threshold.

We define the V1 score value of the referenece execution to be 1000. All other
V1 scores are relative to the performance of the reference execution. For
example, if another V1 benchmark execution's score is 900, it means the its
performance is 10% slower comparing to the reference execution.
