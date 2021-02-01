from util import *
import numpy as np
import matplotlib.pyplot as plt


"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
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
epochs = 60
for epoch in range(epochs):
    for img, l in zip(mazes, solves):
        img.shape += (1,)
        l.shape += (1,)
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
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # Backpropagation output -> hidden2 (cost function derivative)
        delta_o = o - l
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
    print(f"Acc: {round((nr_correct / mazes.shape[0]) * 100, 2)}%")
    nr_correct = 0

'''
# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = mazes[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Subscribe if its a {o.argmax()} :)")
    plt.show()
'''