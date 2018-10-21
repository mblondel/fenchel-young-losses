# Author: Mathieu Blondel
# License: Simplified BSD

"""
PyTorch implementation of

Learning Classifiers with Fenchel-Young Losses:
    Generalized Entropies, Margins, and Algorithms.
Mathieu Blondel, André F. T. Martins, Vlad Niculae.
https://arxiv.org/abs/1805.09717
"""


import torch


class ConjugateFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, grad, Omega):
        ctx.save_for_backward(grad)
        return torch.sum(theta * grad, dim=1) - Omega(grad)

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad * grad_output.view(-1, 1), None, None


class FYLoss(torch.nn.Module):

    def __init__(self, weights="average"):
        self.weights = weights
        super(FYLoss, self).__init__()

    def forward(self, theta, y_true):
        self.y_pred = self.predict(theta)
        ret = ConjugateFunction.apply(theta, self.y_pred, self.Omega)

        if len(y_true.shape) == 2:
            # y_true contains label proportions
            ret += self.Omega(y_true)
            ret -= torch.sum(y_true * theta, dim=1)

        elif len(y_true.shape) == 1:
            # y_true contains label integers (0, ..., n_classes-1)

            if y_true.dtype != torch.long:
                raise ValueError("y_true should contains long integers.")

            all_rows = torch.arange(y_true.shape[0])
            ret -= theta[all_rows, y_true]

        else:
            raise ValueError("Invalid shape for y_true.")

        if self.weights == "average":
            return torch.mean(ret)
        else:
            return torch.sum(ret)


class SquaredLoss(FYLoss):

    def Omega(self, mu):
        return 0.5 * torch.sum((mu ** 2), dim=1)

    def predict(self, theta):
        return theta


class PerceptronLoss(FYLoss):

    def predict(self, theta):
        ret = torch.zeros_like(theta)
        all_rows = torch.arange(theta.shape[0])
        ret[all_rows, torch.argmax(theta, dim=1)] = 1
        return ret

    def Omega(self, theta):
        return 0


def Shannon_negentropy(p, dim):
    tmp = torch.zeros_like(p)
    mask = p > 0
    tmp[mask] = p[mask] * torch.log(p[mask])
    return torch.sum(tmp, dim)


class LogisticLoss(FYLoss):

    def predict(self, theta):
        return torch.nn.Softmax(dim=1)(theta)

    def Omega(self, p):
        return Shannon_negentropy(p, dim=1)


class Logistic_OVA_Loss(FYLoss):

    def predict(self, theta):
        return torch.nn.Sigmoid()(theta)

    def Omega(self, p):
        return Shannon_negentropy(p, dim=1) + Shannon_negentropy(1 - p, dim=1)


# begin: From OpenNMT-py
def threshold_and_support(z, dim=0):
    """
    z: any dimension
    dim: dimension along which to apply the sparsemax
    """
    sorted_z, _ = torch.sort(z, descending=True, dim=dim)
    z_sum = sorted_z.cumsum(dim) - 1  # sort of a misnomer
    k = torch.arange(1, sorted_z.size(dim) + 1, device=z.device).type(z.dtype).view(
        torch.Size([-1] + [1] * (z.dim() - 1))
    ).transpose(0, dim)
    support = k * sorted_z > z_sum

    k_z_indices = support.sum(dim=dim).unsqueeze(dim)
    k_z = k_z_indices.type(z.dtype)
    tau_z = z_sum.gather(dim, k_z_indices - 1) / k_z
    return tau_z, k_z


class SparsemaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dim=0):
        """
        input (FloatTensor): any shape
        returns (FloatTensor): same shape with sparsemax computed on given dim
        """
        ctx.dim = dim
        tau_z, k_z = threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau_z, min=0)
        ctx.save_for_backward(k_z, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        k_z, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = (grad_input.sum(dim=dim) / k_z.squeeze()).unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(torch.nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)
# end: From OpenNMT-py


class SparsemaxLoss(FYLoss):

    def predict(self, theta):
        return Sparsemax(dim=1)(theta)

    def Omega(self, p):
        return 0.5 * torch.sum((p ** 2), dim=1) - 0.5
