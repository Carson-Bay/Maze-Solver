import pickle
import numpy as np
from PIL import Image
import pandas as pd


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


def score(y_true, y_pred):
    unique, counts = np.unique(y_true, return_counts=True)
    count_dict = dict(zip(unique, counts))
    len_path = count_dict[1]

    ind = np.argpartition(y_pred, -len_path)[-len_path:]


def ReLu(x):
    for i in range(len(x)):
        if x[i] <= 0.0:
            x[i] = 0.01 * x[i]
    return x


def sigmoid(x):
    return 1/(1 + np.exp(-x))


if __name__ == "__main__":
    load_data("data.pickle")
