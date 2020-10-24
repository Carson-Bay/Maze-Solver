from PIL import Image
import numpy as np
from heapq import heapify, heappush, heappop

global yCord, xCord, current_X, current_Y, Open, Closed, array, target_X, target_Y, size
cardinalCost = 10


class NodeData:
    def __init__(self, position, parent, f_cost, g_cost):
        self.p = position
        # nodeData object of parent
        self.parent = parent
        self.f = f_cost
        self.g = g_cost

    def __eq__(self, other):
        return (self.f, self.g, self.p, self.parent) == (other.f, other.g, other.p, other.parent)

    def __ne__(self, other):
        return (self.f, self.g, self.p, self.parent) != (other.f, other.g, other.p, other.parent)

    def __lt__(self, other):
        return (self.f, self.g, self.p, self.parent) < (other.f, other.g, other.p, other.parent)

    def __le__(self, other):
        return (self.f, self.g, self.p, self.parent) <= (other.f, other.g, other.p, other.parent)

    def __gt__(self, other):
        return (self.f, self.g, self.p, self.parent) > (other.f, other.g, other.p, other.parent)

    def __ge__(self, other):
        return (self.f, self.g, self.p, self.parent) >= (other.f, other.g, other.p, other.parent)


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

    return cardinalCost * (x_change + y_change)


def find_g_cost(node_object):
    g_cost = 0
    while True:
        g_cost += cardinalCost

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


def neighbors(coord):
    n = []
    x, y = coord

    for i in range(-1, 2, 2):
        if 0 < (x + i) < size:
            n.append((x + i, y))

    for j in range(-1, 2, 2):

        if 0 < (y + j) < size:
            n.append((x, y + j))

    return n


def calculate_node(current_node):
    nodes = []

    for neighbor in neighbors(current_node.p):
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

        neighbor_object = NodeData(neighbor, None, 0, 0)

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

        if not w or current_node.g + cardinalCost < neighbor_object.g:
            neighbor_object.parent = current_node
            neighbor_object.f = find_h_cost(neighbor_x, neighbor_y, target_X, target_Y) + find_g_cost(neighbor_object)
            neighbor_object.g = find_g_cost(neighbor_object)
            if neighbor_object not in Open:
                nodes.append(neighbor_object)
    return nodes


def solve(mazeNumber, s):
    global Open, Closed, size, start_X, start_Y, target_X, target_Y
    size = s
    # load the image
    image = Image.open("generatedMazes/maze{}.png".format(mazeNumber)).convert('1')
    # convert image to numpy array
    global array
    array = np.asarray(image)

    Image.Image.close(image)


# Find Start and End
    start = find_start()
    start_X, start_Y = start
    target = find_target()
    target_X, target_Y = target

    Open = []  # Set of nodes to be evaluated
    Closed = []  # Set of nodes already evaluated

    # Add start node to Open
    Open.append(NodeData(start, None, find_h_cost(start_X, start_Y, target_X, target_Y), 0))

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

        averageDiversions = 2.5

        if len(Open) + len(Closed) > averageDiversions * 8:
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
            while True:
                current = find_lowest_f(Open)
                if current.p == target:
                    break
                else:
                    for node in calculate_node(current):
                        Open.append(node)

    image = Image.new("RGB", (size, size), color="White")
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
        array[current.p] = [0, 0, 0]
        if current.p == start:
            break
        else:
            current = current.parent

    img = Image.fromarray(array, "RGB")
    img.save("generatedMazes/solve{}.png".format(mazeNumber))
