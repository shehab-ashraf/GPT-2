import torch
from torch.optim import Optimizer


# -----------------------------------------------------------------------------
# Newton-Schulz orthogonalization

@torch.compile
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + 1e-7)

    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(G.dtype)


# -----------------------------------------------------------------------------
# Muon optimizer

class Muon(Optimizer):

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr  = group["lr"]
            mom = group["momentum"]
            wd  = group["weight_decay"]
            ns  = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.float()

                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]

                buf.mul_(mom).add_(g)
                g = g + mom * buf

                if p.ndim == 2:
                    g = newton_schulz(g, steps=ns)
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5

                    if wd > 0:
                        p.mul_(1.0 - wd * lr)
                    p.add_(g.to(p.dtype) * scale, alpha=-lr)
                else:
                    p.add_(g.to(p.dtype), alpha=-lr)

        return loss