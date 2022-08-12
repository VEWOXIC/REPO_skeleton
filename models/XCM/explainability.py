# File copied from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/explainability.py

__all__ = ['get_acts_and_grads', 'get_attribution_map']

# Cell
from fastai.callback.hook import *
from .imports import *
from .layers import *

warnings.filterwarnings("ignore", category=UserWarning)

# Cell
def get_acts_and_grads(model, modules, x, y=None, detach=True, cpu=False):
    r"""Returns activations and gradients for given modules in a model and a single input or a batch.
    Gradients require y value(s). If they rae not provided, it will use the predicttions. """
    if not is_listy(modules): modules = [modules]
    x = x[None, None] if x.ndim == 1 else x[None] if x.ndim == 2 else x
    with hook_outputs(modules, detach=detach, cpu=cpu) as h_act:
        with hook_outputs(modules, grad=True, detach=detach, cpu=cpu) as h_grad:
            preds = model.eval()(x)
            if y is None: preds.max(dim=-1).values.mean().backward()
            else:
                if preds.shape[0] == 1: preds[0, y].backward()
                else:
                    if y.ndim == 1: y = y.reshape(-1, 1)
                    torch_slice_by_dim(preds, y).mean().backward()
    if len(modules) == 1: return h_act.stored[0].data, h_grad.stored[0][0].data
    else: return [h.data for h in h_act.stored], [h[0].data for h in h_grad.stored]


def get_attribution_map(model, modules, x, y=None, detach=True, cpu=False, apply_relu=True):
    def _get_attribution_map(A_k, w_ck):
        dim = (0, 2, 3) if A_k.ndim == 4 else (0, 2)
        w_ck = w_ck.mean(dim, keepdim=True)
        L_c = (w_ck * A_k).sum(1)
        if apply_relu: L_c = nn.ReLU()(L_c)
        if L_c.ndim == 3:  return L_c.squeeze(0) if L_c.shape[0] == 1 else L_c
        else: return L_c.repeat(x.shape[1], 1) if L_c.shape[0] == 1 else L_c.unsqueeze(1).repeat(1, x.shape[1], 1)
    if x.ndim == 1: x = x[None, None]
    elif x.ndim == 2: x = x[None]
    A_k, w_ck = get_acts_and_grads(model, modules, x, y, detach=detach, cpu=cpu)
    if is_listy(A_k): return [_get_attribution_map(A_k[i], w_ck[i]) for i in range(len(A_k))]
    else: return _get_attribution_map(A_k, w_ck)

def torch_slice_by_dim(t, index, dim=-1, **kwargs):
    if not isinstance(index, torch.Tensor): index = torch.Tensor(index)
    assert t.ndim == index.ndim, "t and index must have the same ndim"
    index = index.long()
    return torch.gather(t, dim, index, **kwargs)