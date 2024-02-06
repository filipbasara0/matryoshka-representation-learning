import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mrl.utils import get_feature_size


class MatryoshkaProjector(nn.Module):
    def __init__(self, nesting_dims, out_dims, **kwargs):
        super(MatryoshkaProjector, self).__init__()
        self.nesting_dims = nesting_dims
        self.proj_hidden = nn.Linear(nesting_dims[-1], nesting_dims[-1], **kwargs)
        self.relu = nn.ReLU()
        self.proj_linear = nn.Linear(nesting_dims[-1], out_dims, **kwargs)

    def _apply_linear_layer(self, x, layer, nesting_dim):
        logits = torch.matmul(x[:, :nesting_dim], layer.weight[:, :nesting_dim].t())
        if layer.bias != None:
            logits += layer.bias
        return logits

    def forward(self, x):
        nesting_logits = []
        for nesting_dim in self.nesting_dims:
            logits = self._apply_linear_layer(x, self.proj_hidden, nesting_dim)
            logits = self.relu(logits)
            logits = self._apply_linear_layer(x, self.proj_linear, nesting_dim)
            nesting_logits.append(logits)
        return nesting_logits


def relic_loss(x, x_prime, temp, alpha, max_tau=5.0):
    """
    Parameters:
    x (torch.Tensor): Online projections [n, dim].
    x_prime (torch.Tensor): Target projections of shape [n, dim].
    temp (torch.Tensor): Learnable temperature parameter.
    alpha (float): KL divergence (regularization term) weight.
    """
    n = x.size(0)
    x, x_prime = F.normalize(x, p=2, dim=-1), F.normalize(x_prime, p=2, dim=-1)
    logits = torch.mm(x, x_prime.t()) * temp.exp().clamp(0, max_tau)

    # Instance discrimination loss
    labels = torch.arange(n).to(logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)

    # KL divergence loss
    p1 = torch.nn.functional.log_softmax(logits, dim=1)
    p2 = torch.nn.functional.softmax(logits, dim=0).t()
    invariance_loss = torch.nn.functional.kl_div(p1, p2, reduction="batchmean")

    loss = loss + alpha * invariance_loss

    return loss


class ReLIC(torch.nn.Module):

    def __init__(self,
                 encoder,
                 proj_out_dim=64,
                 nesting_dims=None,
                 proj_in_dim=None,
                 matryoshka_bias=False):
        super(ReLIC, self).__init__()

        if not proj_in_dim:
            proj_in_dim = get_feature_size(encoder)
        
        if not nesting_dims:
            nesting_dims = [2**i for i in range(3, int(math.log2(proj_in_dim)) + 1)]

        proj = MatryoshkaProjector(nesting_dims, proj_out_dim, bias=matryoshka_bias)

        self.online_encoder = torch.nn.Sequential(encoder, proj)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)

        self.t_prime = nn.Parameter(torch.zeros(1))

    @torch.inference_mode()
    def get_features(self, img):
        with torch.no_grad():
            return self.target_encoder[0](img)

    def forward(self, x1, x2):
        o1, o2 = self.online_encoder(x1), self.online_encoder(x2)
        with torch.no_grad():
            t1, t2 = self.target_encoder(x1), self.target_encoder(x2)
        t1 = [t_.detach() for t_ in t1]
        t2 = [t_.detach() for t_ in t2]
        return o1, o2, t1, t2
    
    @torch.inference_mode()
    def get_target_pred(self, x):
        with torch.no_grad():
            t = self.target_encoder(x)
        t = [t_.detach() for t_ in t]
        return t
    
    def get_online_pred(self, x):
        return self.online_encoder(x)

    def update_params(self, gamma):
        with torch.no_grad():
            valid_types = [torch.float, torch.float16]
            for o_param, t_param in self._get_params():
                if o_param.dtype in valid_types and t_param.dtype in valid_types:
                    t_param.data.lerp_(o_param.data, 1. - gamma)

            for o_buffer, t_buffer in self._get_buffers():
                if o_buffer.dtype in valid_types and t_buffer.dtype in valid_types:
                    t_buffer.data.lerp_(o_buffer.data, 1. - gamma)

    def copy_params(self):
        for o_param, t_param in self._get_params():
            t_param.data.copy_(o_param)

        for o_buffer, t_buffer in self._get_buffers():
            t_buffer.data.copy_(o_buffer)

    def save_encoder(self, path):
        torch.save(self.target_encoder[0].state_dict(), path)

    def _get_params(self):
        return zip(self.online_encoder.parameters(),
                   self.target_encoder.parameters())

    def _get_buffers(self):
        return zip(self.online_encoder.buffers(),
                   self.target_encoder.buffers())
