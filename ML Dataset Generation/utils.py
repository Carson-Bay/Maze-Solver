import pickle
import numpy as np
import math
from PIL import Image
from numba import jit
import pandas as pd
import collections


def store_data(folder, mazeSize, numMazes):
    mazes = np.empty((1, mazeSize ** 2), dtype="bool")
    solves = np.empty((1, mazeSize ** 2), dtype="bool")
    for i in range(0, numMazes):
        # Get mazes
        filename = "maze{}.png".format(i)
        image = Image.open(folder + "\\" + filename).convert('1')
        # convert image to numpy array
        array = np.asarray(image)
        Image.Image.close(image)
        array = array.reshape(1, mazeSize ** 2)
        mazes = np.append(mazes, array, axis=0)

        # Get Labels
        filename = "solve{}.png".format(i)
        image = Image.open(folder + "\\" + filename).convert('1')
        # convert image to numpy array
        array = np.asarray(image)
        Image.Image.close(image)
        array = array.reshape(1, mazeSize ** 2)
        solves = np.append(mazes, array, axis=0)

    mazes = np.delete(mazes, 0, axis=0)
    solves = np.delete(solves, 0, axis=0)
    mazes = mazes.astype(int)
    solves = solves.astype(int)

    pd.DataFrame(mazes).to_csv("mazes.csv")
    pd.DataFrame(solves).to_csv("solves.csv")

    with open("data.pickle", "wb") as fout:
        pickle.dump(mazes, fout)
        pickle.dump(solves, fout)


def load_data(filename):
    with open(filename, "rb") as fin:
        mazes = pickle.load(fin)
        solves = pickle.load(fin)
    return mazes, solves


def score(y_true, y_pred, size):
    # Find how many nodes the path takes
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    unique, counts = np.unique(y_true, return_counts=True)
    count_dict = dict(zip(unique, counts))
    len_path = count_dict[1]

    # turn y_pred into dict sorted by key (predicted value). Has values as index before being sorted

    y_dict = {}
    for i in range(len(y_pred)):
        y_dict[y_pred[i]] = i

    y_dict = collections.OrderedDict(sorted(y_dict.items()))  # might be wrong syntax but I'm pretty sure this works
    amount_correct = 0
    nodes_checked = 0
    keys = y_dict.keys()

    for i in keys:
        nodes_checked += 1
        if y_true[y_dict[i]] == 1:
            amount_correct += 1
        if amount_correct >= len_path:
            break

    # percent as decimal of score where 100 is amount_checked == y_path and slowly deteriorates after that

    l = np.linspace(0, 1, num=(size - len_path))[::-1]

    return l[(nodes_checked - len_path)-1]

    # ind = np.argpartition(y_pred, -len_path)[-len_path:]


def ReLu(x):
    for i in range(len(x)):
        if x[i] <= 0.0:
            x[i] = 0.01 * x[i]
    return x


@jit(nopython=True)
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    sig = np.minimum(sig, 0.9999)  # Set upper bound
    sig = np.maximum(sig, 0.0001)  # Set lower bound
    return sig


def cross_entropy(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    sum = 0
    for i in range(len(y_pred)):
        try:
            sum -= (y_true[i] * math.log10(y_pred[i])) + (1 - y_true[i]) * math.log10(1 - y_pred[i])
        except ValueError:
            print("Pred" + str(y_pred[i]))
            print("True" + str(y_true[i]))
    return sum


if __name__ == "__main__":
    mazes, solves = load_data("data11.pickle")
