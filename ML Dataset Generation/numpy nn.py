from utils import *
import numpy as np
import sys


"""
w = weights, b = bias, i = input, h1 = hidden1, h2 = hidden2, o = output, s = solve
e.g. w_i_h = weights from input layer to hidden layer
"""
mazes, solves = load_data("data.pickle")
w_i_h = np.random.uniform(-0.5, 0.5, (1000, 2601))
w_h_o = np.random.uniform(-0.5, 0.5, (2601, 1000))
b_i_h = np.zeros((1000, 1))
b_h_o = np.zeros((2601, 1))

learn_rate = 0.01
acc_list = []
epochs = 3
for epoch in range(epochs):
    for img, s in zip(mazes, solves):
        img.shape += (1,)
        s.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img
        h = ReLu(h_pre)
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = sigmoid(o_pre)

        # Cost / Error calculation
        # e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        acc_list.append(hamming_score(s, o))

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = cross_entropy(o, s)
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Show accuracy for this epoch
    print(f"Epoch #{epoch} Acc: {np.mean(acc_list) * 100}%")
    acc_list = []
