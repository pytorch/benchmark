import torch

from .cells import lstm_cell


# list[list[T]] -> list[T]
def flatten_list(lst):
    result = []
    for inner in lst:
        result.extend(inner)
    return result

# Define a creator as a function: (sizes) -> (rnn, rnn_inputs, flat_rnn_params)
# rnn: function / graph executor / module
# rnn_inputs: the inputs to the returned 'rnn'
# flat_rnn_params: List[Tensor] all requires_grad=True parameters in a list
# One can call rnn(rnn_inputs) using the outputs of the creator.

# Full script-mode lstm. Returns (inputs, fn) so that one can fn(*inputs)
# The graph executor takes (Tensor, Tuple[Tensor, Tensor], List[Tensor]) args
def script_lstm_creator(seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
                        miniBatch=64, device='cuda',
                        seed=None):
    input_args = dict(seqLength=seqLength, numLayers=numLayers,
                      inputSize=inputSize, hiddenSize=hiddenSize,
                      miniBatch=miniBatch, device=device, seed=seed)
    input, hidden, params, _ = lstm_inputs(return_module=False, **input_args)
    inputs = [input, hidden, stack_weights(params)]
    return lstm_factory(lstm_cell), inputs, flatten_list(params)


# Script-mode lstm. This graph executor only takes tensors.
# I'm not sure if it matters that this only takes tensors,
# but it is here to be safe.
def script_lstm_flat_inputs_creator(seqLength=100, numLayers=1, inputSize=512,
                                    hiddenSize=512, miniBatch=64,
                                    device='cuda',
                                    seed=None):
    input_args = dict(seqLength=seqLength, numLayers=numLayers,
                      inputSize=inputSize, hiddenSize=hiddenSize,
                      miniBatch=miniBatch, device=device, seed=seed)
    input, hidden, params, _ = lstm_inputs(return_module=False, **input_args)
    wih, whh, bih, bhh = stack_weights(params)
    flat_args = [input, hidden[0], hidden[1], wih, whh, bih, bhh]
    return lstm_factory_flat(lstm_cell), flat_args, flatten_list(params)


def pytorch_lstm_creator(seqLength=100, numLayers=1, inputSize=512,
                         hiddenSize=512, miniBatch=64,
                         return_module=False, device='cuda',
                         seed=None):
    input_args = dict(seqLength=seqLength, numLayers=numLayers,
                      inputSize=inputSize, hiddenSize=hiddenSize,
                      miniBatch=miniBatch, device=device, seed=seed)
    input, hidden, _, module = lstm_inputs(return_module=True, **input_args)
    return module, [input, hidden], flatten_list(module.all_weights)


# input: lstm.all_weights format (wih, whh, bih, bhh = lstm.all_weights[layer])
# output: packed_weights with format
# packed_weights[0] is wih with size (layer, 4*hiddenSize, inputSize)
# packed_weights[1] is whh with size (layer, 4*hiddenSize, hiddenSize)
# packed_weights[2] is bih with size (layer, 4*hiddenSize)
# packed_weights[3] is bhh with size (layer, 4*hiddenSize)
def stack_weights(weights):
    def unzip_columns(mat):
        assert isinstance(mat, list)
        assert isinstance(mat[0], list)
        layers = len(mat)
        columns = len(mat[0])
        return [[mat[layer][col] for layer in range(layers)]
                for col in range(columns)]

    # XXX: script fns have problems indexing multidim lists, so we try to
    # avoid them by stacking tensors
    all_weights = weights
    packed_weights = [torch.stack(param)
                      for param in unzip_columns(all_weights)]
    return packed_weights


# returns: x, (hx, cx), all_weights, lstm module with all_weights as params
def lstm_inputs(seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
                miniBatch=64, return_module=False, device='cuda', seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(seqLength, miniBatch, inputSize, device=device)
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers)
    if 'cuda' in device:
        lstm = lstm.cuda()

    if return_module:
        return x, (hx, cx), lstm.all_weights, lstm
    else:
        # NB: lstm.all_weights format:
        # wih, whh, bih, bhh = lstm.all_weights[layer]
        return x, (hx, cx), lstm.all_weights, None


def lstm_factory(cell):
    @torch.jit.script
    def dynamic_rnn(input, state, params):
        # type: (Tensor, Tuple[Tensor, Tensor], List[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        output = []
        hx, cx = state
        num_layers = hx.size(0)
        seq_len = input.size(0)
        hy = hx  # for scoping
        cy = cx  # for scoping
        for layer in range(num_layers):
            hy = hx[layer]
            cy = cx[layer]
            wih = params[0][layer]
            whh = params[1][layer]
            bih = params[2][layer]
            bhh = params[3][layer]
            for seq_idx in range(seq_len):
                hy, cy = cell(input[seq_idx], (hy, cy), wih, whh, bih, bhh)
                output += [hy]
        return torch.stack(output), (hy.unsqueeze(0), cy.unsqueeze(0))

    return dynamic_rnn


def lstm_factory_flat(cell):
    @torch.jit.script
    def dynamic_rnn(input, hx, cx, lwih, lwhh, lbih, lbhh):
        output = []
        num_layers = hx.size(0)
        seq_len = input.size(0)
        hy = hx  # for scoping
        cy = cx  # for scoping
        for layer in range(num_layers):
            hy = hx[layer]
            cy = cx[layer]
            wih = lwih[layer]
            whh = lwhh[layer]
            bih = lbih[layer]
            bhh = lbhh[layer]
            for seq_idx in range(seq_len):
                hy, cy = cell(input[seq_idx], (hy, cy), wih, whh, bih, bhh)
                output += [hy]
        return torch.stack(output), (hy.unsqueeze(0), cy.unsqueeze(0))

    return dynamic_rnn


def rnn_factory(cell):
    @torch.jit.script
    def dynamic_rnn(input, state, params):
        # type: (Tensor, Tensor, List[Tensor]) -> Tuple[Tensor, Tensor]]
        output = []
        num_layers = state.size(0)
        seq_len = input.size(0)
        for layer in range(num_layers):
            hy = state[layer]
            wih = params[0][layer]
            whh = params[1][layer]
            bih = params[2][layer]
            bhh = params[3][layer]
            for seq_idx in range(seq_len):
                hy, cy = cell(input[seq_idx], state, wih, whh, bih, bhh)
                output += [hy]
        return torch.stack(output), hy.unsqueeze(0)

    return dynamic_rnn
