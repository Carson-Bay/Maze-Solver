from PIL import Image
import random
import numpy as np
import time

global size


class nodeData:
    def __init__(self, position, neighbors):
        self.p = position
        self.n = neighbors


def get_neighbors(coord):
    n = []
    x, y = coord

    for i in range(-2, 3, 4):
        if 0 < (x + i) < size:
            n.append((x + i, y))

    for j in range(-2, 3, 4):

        if 0 < (y + j) < size:
            n.append((x, y + j))

    return n


def maze_template():
    global size
    size = int(input("Size of Mazes: "))
    if size % 2 == 0:
        size += 1
    image = np.array(Image.new("RGB", (size, size), color=0))
    mazeTemplate = np.copy(image)
    mazeTemplate.setflags(write=True)
    openTemplate = []

    for i in range(1, size - 1, 2):
        for j in range(1, size - 1, 2):
            mazeTemplate[i, j] = (255, 255, 255)
            openTemplate.append(nodeData((i, j), None))

    return openTemplate, mazeTemplate, size


def gen_maze(mazeNumber, openTemplate, mazeTemplate, size):
    global open, closed, start, end
    open = openTemplate.copy()
    closed = []
    maze = np.copy(mazeTemplate)

    start = random.randrange(1, size - 1, 2)
    end = random.randrange(1, size - 1, 2)

    maze[0, start] = (255, 255, 255)
    maze[size - 1, end] = (255, 255, 255)

    current = random.choice(open)
    for node in open:
        if node.p == current.p:
            node.n = get_neighbors(node.p)
            open.pop(open.index(node))
            closed.append(node)
            break

    while True:
        current = random.choice(closed)
        while current.n == []:
            current = random.choice(closed)

        current_X, current_Y = current.p
        neighbors = get_neighbors(current.p)
        valid_Neighbors = []

        for neighbor in neighbors:
            for node in open:
                if node.p == neighbor:
                    valid_Neighbors.append(neighbor)
                    break
        if not valid_Neighbors:
            continue

        current.n = valid_Neighbors
        neighbor = random.choice(current.n)
        neighbor_X, neighbor_Y = neighbor

        for node in open:
            if node.p == neighbor:
                open.pop(open.index(node))
                closed.append(node)
                break

        wall_X = int((current_X + neighbor_X) / 2)
        wall_Y = int((current_Y + neighbor_Y) / 2)
        maze[wall_X, wall_Y] = (255, 255, 255)

        if not open:
            break

    img = Image.fromarray(maze, "RGB")
    # img.save(r"C:\Users\cgbma\Documents\Untracked Files\Generated Mazes\11x11 100,000\maze{}.png".format(mazeNumber))
    img.save("portfolio.png")

    maze = np.int64(np.all(maze[:, :, :3] == 0, axis=2))

    return maze.reshape(1, size ** 2)


if __name__ == "__main__":

    openTemp, mazeTemp, size = maze_template()

    numberOfMazes = input("Number of Mazes to Gen: ")

    startTime = time.perf_counter()

    for i in range(0, int(numberOfMazes)):
        gen_maze(i, openTemp, mazeTemp, size)

    runTime = time.perf_counter() - startTime
    print(runTime)
