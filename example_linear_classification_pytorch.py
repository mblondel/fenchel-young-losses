# Author: Mathieu Blondel
# License: Simplified BSD

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from fyl_pytorch import PerceptronLoss
from fyl_pytorch import LogisticLoss
from fyl_pytorch import Logistic_OVA_Loss
from fyl_pytorch import SparsemaxLoss

# Hyper-parameters
num_epochs = 100
learning_rate = 0.001

# Toy dataset: we create 40 separable points.
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = np.array([0] * 20 + [1] * 20)

# Convert numpy arrays to torch tensors.
inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(y.astype(np.long))

# Linear model
model = torch.nn.Linear(in_features=X.shape[1], out_features=2, bias=True)

# Loss and optimizer
losses = {
    "logistic": LogisticLoss(),
    "logistic-ova": Logistic_OVA_Loss(),
    "sparsemax": SparsemaxLoss(),
    "perceptron": PerceptronLoss(),
}

if len(sys.argv) == 1:
    criterion = LogisticLoss()
elif sys.argv[1] in losses:
    criterion = losses[sys.argv[1]]
else:
    raise ValueError("Invalid loss.")

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

torch.manual_seed(0)

# Train the model.
for epoch in range(num_epochs):

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Get the separating hyperplane.
w = model.weight[1].detach().numpy() - model.weight[0].detach().numpy()
b = model.bias[1].detach().numpy() - model.bias[0].detach().numpy()
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - b / w[1]

plt.plot(xx, yy, 'k-')

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()
