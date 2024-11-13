import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic data (for demonstration purposes)
torch.manual_seed(40)
X = torch.randn(100, 1) * 10  # 100 data points with a variance of 10
y = 2 * X + 1 + torch.randn(100, 1) * 2  # Linear relationship with some noise


# Define the model (Linear Regression)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature, one output

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.011)  # Stochastic Gradient Descent

# Training the model
epochs = 100
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)

    # Compute the loss
    loss = criterion(y_pred, y)

    # Zero the gradients, perform backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Plot the results
predicted = model(X).detach()  # Get predictions from the trained model

plt.scatter(X.numpy(), y.numpy(), color='blue', label='Original data')
plt.plot(X.numpy(), predicted.numpy(), color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Print learned parameters
print(f'Learned parameters: weight = {model.linear.weight.item()}, bias = {model.linear.bias.item()}')
