## Model Accuracy Regression 

To test model accuracy regression scripts please run 

```
./scripts/model_accuracy.py --data-dir <imagenet-folder with train and val folders>
```

Please make sure that you set [EXAMPLES_HOME](https://github.com/pytorch/examples) in your environment variables. `Tiny ImageNet` can be downloaded from [here](https://tiny-imagenet.herokuapp.com/) to test this script.

### Usage


```
usage: model_accuracy.py [-h] [--repeat REPEAT]
                         [--arch {alexnet,densenet121,densenet161,densenet169,densenet201,inception_v3,resnet101,resnet152,resnet18,resnet34,resnet50,squeezenet1_0,squeezenet1_1,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn}]
                         [--log-dir LOG_DIR] [--filename FILENAME] --data-dir
                         DATA_DIR

PyTorch model accuracy benchmark.

optional arguments:
  -h, --help            show this help message and exit
  --repeat REPEAT       Number of Runs
  --arch {alexnet,densenet121,densenet161,densenet169,densenet201,inception_v3,resnet101,resnet152,resnet18,resnet34,resnet50,squeezenet1_0,squeezenet1_1,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn}
                        model architectures: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | inception_v3
                        | resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: all)
  --log-dir LOG_DIR     the path on the file system to place the working log
                        directory at
  --filename FILENAME   name of the output file
  --data-dir DATA_DIR   path to imagenet dataset
``
