This repsitory provides scripts and tooling to evaluate pytorch along two major axes.
1. Model Accuracy
2. Model and Library Performance

Model Accuracy concerns itself with reproducing models on fixed datasets within certain accuracy ranges. For example, pytorch should not regress or always be able to train a convnet on MNIST with a certain accuracy given a particular model.

Model and Library Performance concerns itself with the speed of PyTorch. It answers questions such as: How does PyTorch compare to Numpy for the addition of two Tensors? How does PyTorch's CPU backend compare to Eigen for the pointwise application of tanh to a Tensor? The goal here is to provide reproducible timings across various platforms.

Please consider the folder `accuracy` for scripts regarding Model Accuracy and the model `performance` for scripts regarding Model and Library Performance.
