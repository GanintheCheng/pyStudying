import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 11)

train_loss = np.array([
    1.20, 0.95, 0.78, 0.65, 0.54,
    0.45, 0.38, 0.32, 0.28, 0.24
])

valid_loss = np.array([
    1.25, 1.00, 0.82, 0.70, 0.62,
    0.58, 0.57, 0.59, 0.64, 0.72
])

plt.plot(epochs, train_loss, label="train loss", marker="o")
plt.plot(epochs, valid_loss, label="validation loss", marker="s")

plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training Curve")
plt.legend()
plt.grid(True)

plt.show()