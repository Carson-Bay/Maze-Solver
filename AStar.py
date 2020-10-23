from PIL import Image
import numpy as np
from heapq import heapify, heappush, heappop
import time
import threading

global yCord, xCord, current_X, current_Y
cardinalCost = 10
diagonalCost = 22

maze = str(input("Image Name: "))
fileSave = str(input("Set New File Name: "))


# load the image
image = Image.open(maze).convert('1')
# convert image to numpy array
array = np.asarray(image)
print(array)

Image.Image.close(image)


class NodeData:
    def __init__(self, position, parent, direction, f_cost, g_cost):
        self.p = position
        # nodeData object of parent
        self.parent = parent
        # Direction to parent is a Bool, true if horizontal or vertical, false if diagonal
        self.direction = direction
        self.f = f_cost
        self.g = g_cost

    def __eq__(self, other):
        return (self.f, self.g, self.p, self.parent, self.direction) == (other.f, other.g, other.p, other.parent, other.direction)

    def __ne__(self, other):
        return (self.f, self.g, self.p, self.parent, self.direction) != (other.f, other.g, other.p, other.parent, other.direction)

    def __lt__(self, other):
        return (self.f, self.g, self.p, self.parent, self.direction) < (other.f, other.g, other.p, other.parent, other.direction)

    def __le__(self, other):
        return (self.f, self.g, self.p, self.parent, self.direction) <= (other.f, other.g, other.p, other.parent, other.direction)

    def __gt__(self, other):
        return (self.f, self.g, self.p, self.parent, self.direction) > (other.f, other.g, other.p, other.parent, other.direction)

    def __ge__(self, other):
        return (self.f, self.g, self.p, self.parent, self.direction) >= (other.f, other.g, other.p, other.parent, other.direction)


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


# Find H cost aka: distance to target disregarding "walls"
def find_h_cost(x_initial, y_initial, x_final, y_final):
    x_change = x_final - x_initial
    y_change = y_final - y_initial

    return cardinalCost * (x_change + y_change)  # + (diagonalCost - 2 * cardinalCost) * min(x_change, y_change)


def find_g_cost(node_object):
    g_cost = 0
    while True:
        if node_object.direction is True:
            g_cost += cardinalCost
        elif node_object.direction is False:
            g_cost += diagonalCost
        if node_object.parent is None:
            break

        else:
            node_object = node_object.parent
    return g_cost


def find_lowest_f(node_list):
    lowest_f_cost = node_list[0].f
    test_node = node_list[0]
    for node in node_list:
        if node.f < lowest_f_cost:
            lowest_f_cost = node.f
            test_node = node
    Open.pop(Open.index(test_node))
    Closed.append(test_node)
    return test_node


def direction_of_neighbor(neighbor_node, current_node):
    current_X, current_Y = current_node.p
    if ((current_X - 1, current_Y) == neighbor_node or
            (current_X - 1, current_Y) == neighbor_node or
            (current_X, current_Y - 1) == neighbor_node or
            (current_X, current_Y + 1) == neighbor_node):

        return True
    else:

        return False


# Function to find index's of neighbors in the array
X, Y = array.shape
neighbors = lambda x, y: [(x2, y2) for x2 in range(x - 1, x + 2)
                          for y2 in range(y - 1, y + 2)
                          if (-1 < x <= X and
                              -1 < y <= Y and
                              (x != x2 or y != y2) and
                              (0 <= x2 <= X) and
                              (0 <= y2 <= Y))]


def calculate_node(current_node):
    nodes = []
    current_X, current_Y = current_node.p

    for neighbor in neighbors(current_X, current_Y):
        neighbor_x, neighbor_y = neighbor

        w = False
        for node in Closed:
            node_X, node_Y = node.p
            if node_X == neighbor_x and node_Y == neighbor_y:
                w = True
                break
            else:
                w = False

        if array[neighbor_x][neighbor_y] == False or w:
            continue

        if direction_of_neighbor(neighbor, current_node):
            neighbor_cost = cardinalCost
        else:
            neighbor_cost = diagonalCost

        neighbor_object = NodeData(neighbor, None, None, 0, 0)

        for i in Open:
            if i.p == neighbor:
                neighbor_object = i
                break

        for node in Open:
            node_X, node_Y = node.p
            if node_X == neighbor_x and node_Y == neighbor_y:
                w = True
                break
            else:
                w = False

        if not w or current_node.g + neighbor_cost < neighbor_object.g:
            neighbor_object.parent = current_node
            neighbor_object.f = find_h_cost(neighbor_x, neighbor_y, target_X, target_Y) + find_g_cost(neighbor_object)
            neighbor_object.g = find_g_cost(neighbor_object)
            if neighbor_object not in Open:
                nodes.append(neighbor_object)
    return nodes


# Find Start and End
start = find_start()
start_X, start_Y = start
target = find_target()
target_X, target_Y = target

Open = []  # Set of nodes to be evaluated
Closed = []  # Set of nodes already evaluated

# Add start node to Open
Open.append(NodeData(start, None, None, find_h_cost(start_X, start_Y, target_X, target_Y), 0))
startTime = time.perf_counter()

# Run first few iterations
mazeSolved = False

for i in range(0, 7):
    current = find_lowest_f(Open)
    if current.p == target:
        mazeSolved = True
        break
    for node in calculate_node(current):
        Open.append(node)

# Check if Binary min heap or unordered list is faster depending on branching factor
if mazeSolved is False:

    averageDiversions = 3

    if len(Open) + len(Closed) > averageDiversions * 8:
        print("Used Binary Heap Tree")
        heapify(Open)
        while True:
            current = heappop(Open)
            Closed.append(current)
            if current.p == target:
                break
            else:
                for node in calculate_node(current):
                    heappush(Open, node)

    else:
        print("Used Unordered List")
        while True:
            current = find_lowest_f(Open)
            if current.p == target:
                break
            else:
                for node in calculate_node(current):
                    Open.append(node)


runtime = time.perf_counter() - startTime

image = Image.open(maze).convert("RGB")
array = np.array(image)
Image.Image.close(image)
array.setflags(write=True)

current = Open[0]
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
        current = current.parent


print(runtime)

img = Image.fromarray(array)
img.save(fileSave)
