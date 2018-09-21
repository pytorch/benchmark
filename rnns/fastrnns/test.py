import argparse
import torch

from .cells import lstm_cell
from .factory import pytorch_lstm_creator
from .runner import get_rnn_runners


def barf():
    import pdb
    pdb.set_trace()


def assertEqual(tensor, expected, threshold=0.001):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        for t, e in zip(tensor, expected):
            assertEqual(t, e)
    else:
        if (tensor - expected).max() > threshold:
            barf()


def filter_requires_grad(tensors):
    return [t for t in tensors if t.requires_grad]


def test_rnns(experim_creator, control_creator, check_grad=True, verbose=False,
              seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
              miniBatch=64, device='cuda', seed=17):
    creator_args = dict(seqLength=seqLength, numLayers=numLayers,
                        inputSize=inputSize, hiddenSize=hiddenSize,
                        miniBatch=miniBatch, device=device, seed=seed)

    print("Setting up...")
    control_rnn, control_inputs, control_params = control_creator(**creator_args)
    experim_rnn, experim_inputs, experim_params = experim_creator(**creator_args)

    print("Checking outputs...")
    control_outputs = control_rnn(*control_inputs)
    experim_outputs = experim_rnn(*experim_inputs)
    assertEqual(experim_outputs, control_outputs)

    print("Checking grads...")
    # NB: control_outputs[0] is always the output of a RNN and we are
    # backpropping against this.
    control_output = control_outputs[0]
    experim_output = experim_outputs[0]
    grad = torch.randn_like(control_output)
    control_grads = torch.autograd.grad([control_output], control_params, grad)
    experim_grads = torch.autograd.grad([experim_output], experim_params, grad)
    assertEqual(experim_grads, control_grads)

    if verbose:
        print(experim_rnn.graph_for(*experim_inputs))
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test lstm correctness')

    parser.add_argument('--seqLength', default='100', type=int)
    parser.add_argument('--numLayers', default='1', type=int)
    parser.add_argument('--inputSize', default='512', type=int)
    parser.add_argument('--hiddenSize', default='512', type=int)
    parser.add_argument('--miniBatch', default='64', type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--check_grad', default='True', type=bool)
    parser.add_argument('--seed', default='17', type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--rnns', nargs='*',
                        help='What to run. jit_flat, jit_premul, jit, etc')
    args = parser.parse_args()
    if args.rnns is None:
        args.rnns = ['jit_flat', 'jit_premul', 'jit']
    print(args)

    if 'cuda' in args.device:
        assert torch.cuda.is_available()

    rnn_runners = get_rnn_runners(*args.rnns)

    test_args = vars(args)
    del test_args['rnns']

    for name, creator, context in rnn_runners:
        with context():
            print('testing {}...'.format(name))
            test_rnns(creator, pytorch_lstm_creator, **test_args)
