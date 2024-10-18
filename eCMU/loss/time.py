import torch
import torch.nn.functional as F
from itertools import combinations, chain


class TLoss(torch.nn.Module):
    r"""Base class for time domain loss modules.
    You can't use this module directly.
    Your loss should also subclass this class.
    Args:
        pred (Tensor): predict time signal tensor with size (batch, *, channels, samples), `*` is an optional multi targets dimension.
        gt (Tensor): target time signal tensor with size (batch, *, channels, samples), `*` is an optional multi targets dimension.
        mix (Tensor): mixture time signal tensor with size (batch, channels, samples)
    Returns:
        tuple: a length-2 tuple with the first element is the final loss tensor,
            and the second is a dict containing any intermediate loss value you want to monitor
    """

    def forward(self, *args, **kwargs):
        return self._core_loss(*args, **kwargs)

    def _core_loss(self, pred, gt, mix):
        raise NotImplementedError


class SDR(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expr = "bi,bi->b"

    def _batch_dot(self, x, y):
        return torch.einsum(self.expr, x, y)

    def forward(self, estimates, references):
        if estimates.dtype != references.dtype:
            estimates = estimates.to(references.dtype)
        length = min(references.shape[-1], estimates.shape[-1])
        references = references[..., :length].reshape(references.shape[0], -1)
        estimates = estimates[..., :length].reshape(estimates.shape[0], -1)

        delta = 1e-7  # avoid numerical errors
        num = self._batch_dot(references, references)
        den = (
            num
            + self._batch_dot(estimates, estimates)
            - 2 * self._batch_dot(estimates, references)
        )
        den = den.relu().add(delta).log10()
        num = num.add(delta).log10()
        return 10 * (num - den)


class NegativeSDR(TLoss):
    def __init__(self) -> None:
        super().__init__()
        self.sdr = SDR()

    def _core_loss(self, pred, gt, mix):
        return -self.sdr(pred, gt).mean(), {}


class CL1Loss(TLoss):
    def _core_loss(self, pred, gt, mix):
        gt = gt[..., : pred.shape[-1]]
        loss = []
        for c in chain(
            combinations(range(4), 1),
            combinations(range(4), 2),
            combinations(range(4), 3),
        ):
            x = sum([pred[:, i] for i in c])
            y = sum([gt[:, i] for i in c])
            loss.append(F.l1_loss(x, y))

        # All 14 Combination Losses (4C1 + 4C2 + 4C3)
        loss_l1 = sum(loss) / len(loss)
        return loss_l1, {}


class L1Loss(TLoss):
    def _core_loss(self, pred, gt, mix):
        return F.l1_loss(pred, gt[..., : pred.shape[-1]]), {}

if __name__ == "__main__":
    sdr = SDR()
    import torch
    # x1 = torch.rand(1,1,2,20)
    # x = torch.stack([x1,x1,x1,x1], axis=1)
    # y = torch.rand(1,1,2,20)
    # print(sdr(x.view(-1, *x.shape[-2:]), y.view(-1, *y.shape[-2:])))
    # print("=======")
    # x = x1
    # print(sdr(x.view(-1, *x.shape[-2:]), y.view(-1, *y.shape[-2:])))

    x = torch.rand(4,2)
    y = torch.rand(1,2)
    print(x)
    print(y)
    print(torch.einsum("bi,bi->b", x, y))