import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Based off of PyTorch's LSTM implementation

# This is slightly different to the most commonly used LSTM variant, where the output gate is
# applied after the hyperbolic tangent.


def KrauseLSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    # Terminology matchup:
    #   - This implementation uses the trick of having all gates concatenated
    #     together into a single tensor, so you can do one matrix multiply to
    #     compute all the gates.
    #   - Thus, w_ih holds W_hx, W_ix, W_ox, W_fx
    #       and w_hh holds W_hh, W_ih, W_oh, W_fh
    #   - Notice that the indices are swapped, because F.linear has swapped
    #     arguments.  "Cancelling" indices are always next to each other.
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    ingate, forgetgate, hiddengate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    outgate = F.sigmoid(outgate)
    forgetgate = F.sigmoid(forgetgate)

    cy = (forgetgate * cx) + (ingate * hiddengate)
    hy = F.tanh(cy * outgate)

    return hy, cy


def MultiplicativeLSTMCell(input, hidden, w_xm, w_hm, w_ih, w_mh, b_xm=None, b_hm=None, b_ih=None, b_mh=None):
    # w_ih holds W_hx, W_ix, W_ox, W_fx
    # w_mh holds W_hm, W_im, W_om, W_fm

    hx, cx = hidden

    # Key difference:
    m = F.linear(input, w_xm, b_xm) * F.linear(hx, w_hm, b_hm)
    gates = F.linear(input, w_ih, b_ih) + F.linear(m, w_mh, b_mh)

    ingate, forgetgate, hiddengate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    outgate = F.sigmoid(outgate)
    forgetgate = F.sigmoid(forgetgate)

    cy = (forgetgate * cx) + (ingate * hiddengate)
    hy = F.tanh(cy * outgate)

    return hy, cy
