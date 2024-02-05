import torchbenchmark.models.densenet121

model, example_inputs = torchbenchmark.models.densenet121.Model(
    test="eval", device="cuda", batch_size=1
).get_module()
model(*example_inputs)
