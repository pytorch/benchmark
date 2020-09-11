import torch
import sys
a = torch.load(sys.argv[1])
b = torch.load(sys.argv[2])
torch.testing.assert_allclose(a,b, rtol=0.01, atol=0.01)
