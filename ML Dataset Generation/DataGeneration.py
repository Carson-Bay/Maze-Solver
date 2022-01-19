import MazeGen
import SolveDataGen as Solve
import numpy as np
import pickle
import time

numberOfMazes = int(input("How many mazes to generate and solve: "))
openTemp, mazeTemp, size = MazeGen.maze_template()
startTime = time.perf_counter()
mazes = np.empty((1, size ** 2), dtype="bool")
solves = np.empty((1, size ** 2), dtype="bool")

for i in range(0, numberOfMazes):

    maze = MazeGen.gen_maze(i, openTemp, mazeTemp, size)
    mazes = np.append(mazes, maze.astype(bool), axis=0)
    solve = Solve.solve(i, size)
    solves = np.append(solves, solve.astype(bool), axis=0)

mazes = mazes.astype(int)
solves = solves.astype(int)
mazes = np.delete(mazes, 0, axis=0)
solves = np.delete(solves, 0, axis=0)

with open("data{}.pickle".format(size), "wb") as fout:
    pickle.dump(mazes, fout)
    pickle.dump(solves, fout)

runTime = time.perf_counter() - startTime
print(runTime)

