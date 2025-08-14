import math

import torch
from torch import nn

CUTOFF = -math.log(2)

def log1mexp(x, eps):
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = CUTOFF < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()+eps).log(),
        (-x.exp()+eps).log1p(),
    )

def negate_real(x, eps):
    return 1 - x


def unroll_ixs(ixs):
    deltas = torch.diff(ixs)
    ixs = torch.arange(len(deltas), dtype=torch.long, device=ixs.device)
    return ixs.repeat_interleave(repeats=deltas)


class KnowledgeModule(nn.Module):
    def __init__(self, ixs_in, ixs_out, semiring='real', probabilistic=False):
        super(KnowledgeModule, self).__init__()
        layers = []
        self.probabilistic = probabilistic
        sum_layer, prod_layer, self.zero, self.one, self.negate = get_semiring(semiring, probabilistic)
        for i, (ix_in, ix_out) in enumerate(zip(ixs_in, ixs_out)):
            ix_in = torch.as_tensor(ix_in, dtype=torch.long)
            ix_out = torch.as_tensor(ix_out, dtype=torch.long)
            ix_out = unroll_ixs(ix_out)
            if i % 2 == 0:
                layers.append(prod_layer(ix_in, ix_out))
            else:
                layers.append(sum_layer(ix_in, ix_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x_pos, x_neg=None, eps=0):
        x = self.encode_input(x_pos, x_neg, eps)
        return self.layers(x)

    def encode_input(self, pos, neg, eps):
        if neg is None:
            neg = self.negate(pos, eps)
        x = torch.stack([pos, neg], dim=1).flatten()
        units = torch.tensor([self.zero, self.one], dtype=torch.float32, device=pos.device)
        return torch.cat([units, x])

    def sparsity(self, nb_vars: int) -> float:
        sparse_params = sum(len(l.ix_out) for l in self.layers)
        layer_widths = [nb_vars] + [l.out_shape[0] for l in self.layers]
        dense_params  = sum(layer_widths[i] * layer_widths[i+1] for i in range(len(layer_widths) - 1))
        return sparse_params / dense_params

    def sample_pc(self):
        assert self.probabilistic
        y = torch.tensor([1])
        for layer in reversed(self.layers):
            y = layer.sample_pc(y)
        return y[2::2]

    def condition_pc(self, x_pos, x_neg):
        assert self.probabilistic
        x = self.encode_input(x_pos, x_neg, None)
        for layer in self.layers:
            x = layer.condition_pc(x) \
                if isinstance(layer, ProbabilisticKnowledgeLayer) \
                else layer(x)
        return x


class KnowledgeLayer(nn.Module):
    def __init__(self, ix_in, ix_out):
        super().__init__()
        self.register_buffer('ix_in', ix_in)
        self.register_buffer('ix_out', ix_out)
        self.out_shape = (self.ix_out[-1].item() + 1,)
        self.in_shape = (self.ix_in.max().item() + 1,)

    def _scatter_forward(self, x: torch.Tensor, reduce: str):
        output = torch.empty(self.out_shape, dtype=x.dtype, device=x.device)
        output = torch.scatter_reduce(output, 0, index=self.ix_out, src=x, reduce=reduce, include_self=False)
        return output

    def _scatter_backward(self, x: torch.Tensor, reduce: str):
        output = torch.zeros(self.in_shape, dtype=x.dtype, device=x.device)
        output = torch.scatter_reduce(output, 0, index=self.ix_in, src=x, reduce=reduce, include_self=False)
        return output

    def _safe_exp(self, x: torch.Tensor):
        with torch.no_grad():
            max_output = self._scatter_forward(x, "amax")
        x = x - max_output[self.ix_out]
        x.nan_to_num_(nan=0., posinf=float('inf'), neginf=float('-inf'))
        return torch.exp(x), max_output

    def _logsumexp_scatter(self, x: torch.Tensor, eps: float):
        x, max_output = self._safe_exp(x)
        output = torch.full(self.out_shape, eps, dtype=x.dtype, device=x.device)
        output = torch.scatter_add(output, 0, index=self.ix_out, src=x)
        output = torch.log(output) + max_output
        return output



class ProbabilisticKnowledgeLayer(KnowledgeLayer):
    def __init__(self, ix_in, ix_out):
        super().__init__(ix_in, ix_out)
        self.weights = nn.Parameter(torch.randn_like(ix_in, dtype=torch.float32))

    def get_edge_weights(self):
        exp_weights, _ = self._safe_exp(self.weights)
        norm = self._scatter_forward(exp_weights, "sum")
        return exp_weights / norm[self.ix_out]

    def renorm_weights(self, x):
        with torch.no_grad():
            self.weights.data = self.get_log_edge_weights(0) + x

    def get_log_edge_weights(self, eps):
        norm = self._logsumexp_scatter(self.weights, eps)
        return self.weights - norm[self.ix_out]

    def sample_pc(self, y, eps=10e-16):
        weights = self.get_log_edge_weights(eps)
        noise = -(-torch.log(torch.rand_like(weights) + eps) + eps).log()
        gumbels = weights + noise
        samples = self._scatter_forward(gumbels, "amax")
        samples = samples[self.ix_out] == gumbels
        samples &= y[self.ix_out].to(torch.bool)
        return self._scatter_backward(samples, "sum") > 0


class SumLayer(KnowledgeLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ix_in], "sum")

    def sample_pc(self, y):
        return self._scatter_backward(y[self.ix_out], "amax")


class ProdLayer(KnowledgeLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ix_in], "prod")

    def sample_pc(self, y):
        return self._scatter_backward(y[self.ix_out], "amax")


class MinLayer(KnowledgeLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ix_in], "amin")


class MaxLayer(KnowledgeLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ix_in], "amax")


class LogSumLayer(KnowledgeLayer):
    def forward(self, x, eps=10e-16):
        return self._logsumexp_scatter(x[self.ix_in], eps)


class ProbabilisticSumLayer(ProbabilisticKnowledgeLayer):
    def forward(self, x):
        x = self.get_edge_weights() * x[self.ix_in]
        return self._scatter_forward(x, "sum")

    def condition_pc(self, x):
        x2 = self.forward(x)
        self.renorm_weights(x[self.ix_in].log())
        return x2


class ProbabilisticLogSumLayer(ProbabilisticKnowledgeLayer):
    def forward(self, x, eps=10e-16):
        x = self.get_log_edge_weights(eps) + x[self.ix_in]
        return self._logsumexp_scatter(x, eps)

    def condition_pc(self, x):
        y = self.forward(x)
        self.renorm_weights(x[self.ix_in])
        return y

def get_semiring(name: str, probabilistic: bool):
    """
    For a given semiring, returns the sum and product layer,
    the zero and one elements, and a negation function.
    """
    if probabilistic:
        if name == "real":
            return ProbabilisticSumLayer, ProdLayer, 0, 1, negate_real
        if name == "log":
            return ProbabilisticLogSumLayer, SumLayer, float('-inf'), 0, log1mexp
        raise ValueError(f"Unknown probabilistic semiring {name}")
    else:
        if name == "real":
            return SumLayer, ProdLayer, 0, 1, negate_real
        elif name == "log":
            return LogSumLayer, SumLayer, float('-inf'), 0, log1mexp
        elif name == "mpe":
            return MaxLayer, ProdLayer, 0, 1, negate_real
        elif name == "godel":
            return MaxLayer, MinLayer, 0, 1, negate_real
        raise ValueError(f"Unknown semiring {name}")
