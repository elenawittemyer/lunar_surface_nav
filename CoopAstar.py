#This implementation comes from from https://www.redblobgames.com/pathfinding/a-star/implementation.html

import numpy as np
import collections
import heapq

class SimpleGraph:
    def __init__(self):
        self.edges = {}
    def neighbors(self, id):
        return self.edges[id]
    
GridLocation = tuple[int, int]

class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []
    
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id):
        return id not in self.walls
    
    def neighbors(self, id):
        (x, y) = id
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)] # E W N S
        # see "Ugly paths" section for an explanation:
        if (x + y) % 2 == 0: neighbors.reverse() # S N W E
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        return results

class WeightedGraph():
    def cost(self, from_id, to_id):
        pass


class Queue:
    def __init(self):
        self.elements = collections.deque()
    def empty(self):
        return not self.elements
    def put(self, x):
        self.elements.append(x)
    def get(self):
        return self.elements.popleft()

class PriorityQueue:
    def __init__(self):
        self.elements={}
    def empty(self):
        return not self.elements
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    def get(self):
        return heapq.heappop(self.elements)[1]

def heuristic(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return abs(x1-x2) + abs(y1-y2)

def astar(graph):
    return

