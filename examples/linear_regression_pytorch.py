# Author: Mathieu Blondel
# License: Simplified BSD

import os, sys
currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(currentdir))

import torch
import matplotlib.pyplot as plt

from fyl_pytorch import SquaredLoss

# Hyper-parameters
num_epochs = 60
learning_rate = 0.001

# Toy dataset
X = torch.tensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]])

y = torch.tensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]])

# Linear model
model = torch.nn.Linear(in_features=1, out_features=1)

# Loss and optimizer
criterion = SquaredLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):

    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
y_pred = model(X).detach().numpy()
plt.plot(X.numpy(), y.numpy(), 'ro', label='Observations')
plt.plot(X.numpy(), y_pred, label='Fitted line')
plt.legend()
plt.show()
