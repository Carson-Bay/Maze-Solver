import MazeGen
import SolveDataGen as Solve
import time

numberOfMazes = int(input("How many mazes to generate and solve: "))
openTemp, mazeTemp, size = MazeGen.maze_template()
startTime = time.perf_counter()

for i in range(9672, numberOfMazes):

    MazeGen.gen_maze(i, openTemp, mazeTemp, size)
    Solve.solve(i, size)

runTime = time.perf_counter() - startTime
print(runTime)