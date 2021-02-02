from util import *
import numpy as np

"""
w = weights, b = bias, i = input, h1 = hidden1, h2 = hidden2, o = output, s = solve
e.g. w_i_h = weights from input layer to hidden layer
"""
mazes, solves = load_data("data.pickle")
w_i_h1 = np.random.uniform(-0.5, 0.5, (20, 2601))
w_h1_h2 = np.random.uniform(-0.5, 0.5, (20, 20))
w_h2_o = np.random.uniform(-0.5, 0.5, (2601, 20))
b_i_h1 = np.zeros((20, 1))
b_h1_h2 = np.zeros((20, 1))
b_h2_o = np.zeros((2601, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 3
for epoch in range(epochs):
    for img, s in zip(mazes, solves):
        img.shape += (1,)
        s.shape += (1,)
        # Forward propagation input -> hidden1
        h1_pre = b_i_h1 + w_i_h1 @ img
        h1 = 1 / (1 + np.exp(-h1_pre))
        # Forward propagation hidden1 -> hidden2
        h2_pre = b_h1_h2 + w_h1_h2 @ h1
        h2 = 1 / (1 + np.exp(-h2_pre))
        # Forward propagation hidden2 -> output
        o_pre = b_h2_o + w_h2_o @ h2
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        # e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(s))

        # Backpropagation output -> hidden2 (cost function derivative)
        delta_o = o - s
        w_h2_o += -learn_rate * delta_o @ np.transpose(h2)
        b_h2_o += -learn_rate * delta_o
        # Backpropagation hidden2 -> hidden1 (activation function derivative)
        delta_h2 = np.transpose(w_h2_o) @ delta_o * (h2 * (1 - h2))
        w_h1_h2 += -learn_rate * delta_h2 @ np.transpose(h1)
        b_h1_h2 += -learn_rate * delta_h2
        # Backpropagation hidden1 -> input (activation function derivative)
        delta_h1 = np.transpose(w_h1_h2) @ delta_h2 * (h1 * (1 - h1))
        w_i_h1 += -learn_rate * delta_h1 @ np.transpose(img)
        b_i_h1 += -learn_rate * delta_h1

    # Show accuracy for this epoch
    print(f"Epoch #{epoch} Acc: {(nr_correct / mazes.shape[0]) * 100}%")
    nr_correct = 0
