from utils import *
import numpy as np
import time
from numba import jit
import matplotlib.pyplot as plt

"""
w = weights, b = bias, i = input, h1 = hidden1, h2 = hidden2, o = output, s = solve
e.g. w_i_h = weights from input layer to hidden layer
"""
mazes, solves = load_data("data11.pickle")
w_i_h = np.random.uniform(-0.5, 0.5, (100, 121))
w_h_o = np.random.uniform(-0.5, 0.5, (121, 100))
b_i_h = np.zeros((100, 1))
b_h_o = np.zeros((121, 1))

learn_rate = 0.01
epochs = 40


# stuff = zip(mazes, solves)
def Epoch(img, s, w_i_h, b_i_h, b_h_o, w_h_o):
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img
    h = sigmoid(h_pre)
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = sigmoid(o_pre)

    # Cost / Error calculation
    # e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)

    # Backpropagation output -> hidden (cost function derivative)
    delta_o = o - s  # cross_entropy(s, o) * o
    w_h_o += -learn_rate * delta_o @ np.transpose(h)
    b_h_o += -learn_rate * delta_o

    # Backpropagation hidden -> input (activation function derivative)
    delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
    w_i_h += -learn_rate * delta_h @ np.transpose(img)
    b_i_h += -learn_rate * delta_h
    return w_i_h, b_i_h, b_h_o, w_h_o


start_time = time.perf_counter()
for epocho in range(epochs):
    for index in range(len(mazes) - 1):
        img, s = mazes[index], solves[index]
        img.shape += (1,)
        s.shape += (1,)
        w_i_h, b_i_h, b_h_o, w_h_o = Epoch(img, s, w_i_h, b_i_h, b_h_o, w_h_o)

    # Show accuracy for this epoch
    print(f"Epoch #{epocho} ")  # Mean Acc: {np.mean(ac_list) * 100}%  Highest Acc: {np.amax(ac_list) * 100}%"
    # f"  Lowest Acc: {np.amin(ac_list) * 100}%")

print(time.perf_counter() - start_time)

# Test with maze not in training

maze = mazes[-1]
solve = solves[-1]
maze.shape += (1,)
solve.shape += (1,)
# Forward propagation input -> hidden
h_pre = b_i_h + w_i_h @ maze
h = sigmoid(h_pre)
# Forward propagation hidden -> output
o_pre = b_h_o + w_h_o @ h
o = sigmoid(o_pre)

reshaped_solve = solve.reshape(11, 11) * 255
reshaped_maze = maze.reshape(11, 11) * 255
reshaped_o = o.reshape(11, 11) * 255
o_img = Image.fromarray(reshaped_o)
maze_img = Image.fromarray(reshaped_maze)
solve_img = Image.fromarray(reshaped_solve)
o_img.show()
solve_img.show()
maze_img.show()
