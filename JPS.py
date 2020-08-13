from PIL import Image
import numpy as np

global yCord, xCord
global target
cardinalCost = 10
diagonalCost = 14

maze = str(input("Image Name: "))
fileSave = str(input("Set New File Name: "))

# load the image
image = Image.open(maze).convert('1')
# convert image to numpy array
array = np.asarray(image)

Image.Image.close(image)


class NodeData:
    def __init__(self, position, direction, g_cost, parent):
        self.p = position
        # nodeData object of parent
        self.parent = parent
        self.d = direction
        self.g = g_cost
        self.f = g_cost

    # Find H cost aka: distance to target disregarding "walls"
    def find_h_cost(self):
        x_initial, y_initial = self.p
        x_final, y_final = target
        x_change = x_final - x_initial
        y_change = y_final - y_initial

        self.f += cardinalCost * (x_change + y_change)


# returns first white pixel in image
def find_start():
    for index, line in enumerate(array):
        for j, pixel in enumerate(line):
            if bool(pixel) is True:
                return index, j


# returns last white pixel in image
def find_target():
    global yCord, xCord
    for index, line in enumerate(array):
        for j, pixel in enumerate(line):
            if bool(pixel) is True:
                xCord = index
                yCord = j
    return xCord, yCord


def search_hor(pos, hor_dir, dist):
    """ Search in horizontal direction, return the newly added open nodes
    @param pos: Start position of the horizontal scan.
    @param hor_dir: Horizontal direction (+1 or -1).
    @param dist: Distance traveled so far.
    @return: New jump point nodes (which need a parent). """

    x0, y0 = pos

    while True:
        x1 = x0 + hor_dir

        try:
            array[x1][y0]
        except(ValueError, IndexError):
            return []

            # Off-map, done.
        if array[x1][y0] == False:
            return []

        # Done.
        if (x1, y0) == target:
            return [NodeData((x1, y0), None, dist + cardinalCost, None)]

            # Open space at (x1, y0).
        dist = dist + cardinalCost
        x2 = x1 + hor_dir
        nodes = []

        if array[x1][y0 - 1] == False and array[x2][y0 - 1] == True:
            nodes.append(NodeData((x1, y0), (hor_dir, -1), dist, None))

        if array[x1][y0 + 1] == False and array[x2][y0 + 1] == True:
            nodes.append(NodeData((x1, y0), (hor_dir, 1), dist, None))

        if len(nodes) > 0:
            nodes.append(NodeData((x1, y0), (hor_dir, 0), dist, None))

            return nodes  # Process next tile. x0 = x1


def search_vert(pos, vert_dir, dist):
    """ Search in vertical direction, return the newly added open nodes
    @param pos: Start position of the vertical scan.
    @param vert_dir: vertical direction (+1 or -1).
    @param dist: Distance traveled so far.
    @return: New jump point nodes (which need a parent). """

    x0, y0 = pos

    while True:
        y1 = y0 + vert_dir

        try:
            array[x0][y1]
        except(ValueError, IndexError):
            return []

            # Off-map, done.
        if array[x0][y1] == False:
            return []

        # Done.
        if (x0, y1) == target:
            return [NodeData((x0, y1), None, dist + cardinalCost, None)]

            # Open space at (x1, y0).
        dist = dist + cardinalCost
        y2 = y1 + vert_dir
        nodes = []

        if array[x0 - 1][y1] == False and array[x0 - 1][y2] == True:
            nodes.append(NodeData((x0, y1), (vert_dir, -1), dist, None))

        if array[x0 + 1][y1] == False and array[x0 + 1][y2] == True:
            nodes.append(NodeData((x0, y1), (vert_dir, 1), dist, None))

        if len(nodes) > 0:
            nodes.append(NodeData((x0, y1), (vert_dir, 0), dist, None).find_h_cost())

        return nodes  # Process next tile. y0 = y1


def search_diagonal(pos, hor_dir, vert_dir, dist):
    """ Search diagonally, spawning horizontal and vertical searches. Returns newly added open nodes.
    @param pos: Start position.
    @param hor_dir: Horizontal search direction (+1 or -1).
    @param vert_dir: Vertical search direction (+1 or -1).
    @param dist: Distance traveled so far.
    @return: Jump points created during this scan (which need to get a parent jump point). """
    x0, y0 = pos
    while True:
        sub_nodes = []
        hor_done = False
        vert_done = False
        x1, y1 = x0 + hor_dir, y0 + vert_dir

        try:
            array[x1][y1]
        except(ValueError, IndexError):
            return []

            # Off-map, done.
        g = array[x1][y1]
        if g == False:
            return []

        if (x1, y1) == target:
            return [NodeData((x1, y1), None, dist + diagonalCost, None)]

        # Open space at (x1, y1)
        dist = dist + diagonalCost
        x2, y2 = x1 + hor_dir, y1 + vert_dir
        nodes = []

        if array[x0][y1] == False and array[x0][y2] == True:
            nodes.append(NodeData((x1, y1), (-hor_dir, vert_dir), dist, None))

        if array[x1][y0] == False and array[x2][y0] == True:
            nodes.append(NodeData((x1, y1), (hor_dir, -vert_dir), dist, None))
            hor_done, vert_done = False, False

        if len(nodes) == 0:
            sub_nodes = search_hor((x1, y1), hor_dir, dist)
            hor_done = True

        if len(sub_nodes) > 0:
            # Horizontal search ended with a jump point.
            for close in Closed:
                if close.p == (x1, y1):
                    pd = close
                    break

            for sub in sub_nodes:
                sub.parent = pd

            nodes.append(pd)

            if len(nodes) == 0:
                sub_nodes = search_vert((x1, y1), vert_dir, dist)
                vert_done = True

                if len(sub_nodes) > 0:
                    # Vertical search ended with a jump point.
                    for close in Closed:
                        if close.p == (x1, y1):
                            pd = close
                            break

                    for sub in sub_nodes:
                        sub.parent = pd
                        nodes.append(pd)

                    if len(nodes) > 0:
                        if not hor_done:
                            nodes.append(NodeData((x1, y1), (hor_dir, 0), dist, None))

                            if not vert_done:
                                nodes.append(NodeData((x1, y1), (0, vert_dir), dist, None))

                                nodes.append(NodeData((x1, y1), (hor_dir, vert_dir), dist, None))

                                return nodes  # Tile done, move to next tile. x0, y0 = x1, y1


# Find Start and End
start = find_start()
start_X, start_Y = start
target = find_target()
target_X, target_Y = target

Open = []  # Set of nodes to be evaluated
Closed = []  # Set of nodes already evaluated

# Add start node to Open
Open.append(NodeData(start, None, 0, None, find_h_cost(start_X, start_Y, target_X, target_Y),))

while True:
    # Set current to node with lowest F cost, remove from open
    current = min(Open, key=lambda x: x.f)
    Open.pop(Open.index(current))
    Closed.append(current)






image = Image.open(maze).convert("RGB")
array = np.array(image)
Image.Image.close(image)
array.setflags(write=True)

for i in Closed:
    if i.p == target:
        current = i
        break
currentNode = current

pathLen = 0
while True:
    pathLen += 1
    if currentNode.p == start:
        break
    else:
        currentNode = currentNode.parent

for i in range(pathLen):
    array[current.p] = [255 - (((i + 1) / (pathLen + 1)) * 255), 0, ((i + 1) / (pathLen + 1)) * 255]
    if current.p == start:
        break
    else:
        print(current.parent.p)
        current = current.parent

img = Image.fromarray(array)
img.save(fileSave)
