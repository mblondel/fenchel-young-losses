# Author: Mathieu Blondel
# License: Simplified BSD

import torch
from torch.autograd import gradcheck, Variable

from fyl_pytorch import SquaredLoss
from fyl_pytorch import PerceptronLoss
from fyl_pytorch import LogisticLoss
from fyl_pytorch import Logistic_OVA_Loss
from fyl_pytorch import SparsemaxLoss


def test_squared_loss():
    y_true = torch.tensor([[1.7], [2.76], [2.09]], dtype=torch.double)
    y_pred = torch.tensor([[0.7], [3.76], [1.09]], dtype=torch.double)
    y_pred = Variable(y_pred, requires_grad=True)
    loss = SquaredLoss()
    assert gradcheck(loss, (y_pred, y_true), eps=1e-4, atol=1e-3)


def test_classification_losses():
    for loss in (PerceptronLoss(), LogisticLoss(), Logistic_OVA_Loss(), SparsemaxLoss()):
        y_true = torch.tensor([0, 0, 1, 2], dtype=torch.long)
        theta = torch.tensor(torch.randn(y_true.shape[0], 3), dtype=torch.double)
        theta = Variable(theta, requires_grad=True)
        assert gradcheck(loss, (theta, y_true), eps=1e-4, atol=1e-3)


def test_classification_losses_multilabel():
    for loss in (LogisticLoss(), Logistic_OVA_Loss(), SparsemaxLoss()):
        y_true = torch.tensor([[1, 0, 0],
                               [1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]], dtype=torch.double)
        theta = torch.tensor(torch.randn(*y_true.shape), dtype=torch.double)
        theta = Variable(theta, requires_grad=True)
        assert gradcheck(loss, (theta, y_true), eps=1e-4, atol=1e-3)
