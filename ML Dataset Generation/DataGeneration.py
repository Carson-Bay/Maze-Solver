import MazeGen
import SolveDataGen as Solve

numberOfMazes = int(input("How many mazes to generate and solve: "))
openTemp, mazeTemp, size = MazeGen.maze_template()


for i in range(0, numberOfMazes):

    MazeGen.gen_maze(i, openTemp, mazeTemp, size)
    Solve.solve(i, size)
