#!/usr/bin/python3
# -*- coding: utf-8 -*-
# *****************************************************************************/
# * Copyright 2021 UCR CS205, Artificial Intelligence. All Rights Reserved.
# * Closed source repository. Do not share any content without permission written from Rogelio Macedo.
# * Authors: Rogelio Macedo
# * Template Credit: Joseph Tarango
# *****************************************************************************/


"""

Usage: $: python3 trenchPuzzleSearch.py -i 12345670 -m 3

each element in the heap be a tuple, where the first element is of a type which accepts normal python comparisons
for example, say I have a class named node, and some cost associated with it.
Then, I could have a tuple for it in the heap as follows: (node.getCost(), node)
Due to issues with nodes of equal cost, then the tuples in the heap have three elements: (cost, insertion index, node)
"""

import os
import argparse
import pprint
import traceback
import datetime
import numpy as np
import heapq


# import time
# import pandas as pd

repeated = set()
# recessPositions = {3, 5, 7}
# experimental: 3 is the goal, changed goal test and manhattan distance functions
recessPositions = {2, 3}



# unit test node cost function for min heap wrapper class
def nodeCost(node):
    return node[0]


def manhattanDistanceHeuristic(node=None):
    return node.weight + node.heuristic


def uniformCost(node=None):
    return node.weight


def trenchEnqueue(priorityQ=None, trenchnode=None):
    # We can move a blank tile/recess in four directions: up, down, left, right
    indexMatrix = trenchnode.findBlanks()
    for i in range(len(indexMatrix[0])):
        flagUp    = True
        flagDown  = True
        flagLeft  = True
        flagRight = True
        row       = indexMatrix[0][i]
        col       = indexMatrix[1][i]
        if row == 0:
            flagUp = False
            flagLeft = False
            flagRight = False
        else:
            flagDown = False
            if col == 0:
                flagLeft = False
            elif col == trenchnode.state.shape[1] - 1:
                flagRight = False
            if col not in recessPositions:
                flagUp = False

        if flagUp:
            newnode = trenchNode(parent=trenchnode)
            newnode.up(i=row, j=col)
            newnode.weight += 1
            newnode.heuristic = newnode.manhattanDistance()
            if np.array2string(newnode.state) not in repeated:
                priorityQ.push(newnode)

        if flagDown:
            newnode = trenchNode(parent=trenchnode)
            newnode.down(i=row, j=col)
            newnode.weight += 1
            newnode.heuristic = newnode.manhattanDistance()
            if np.array2string(newnode.state) not in repeated:
                priorityQ.push(newnode)

        if flagLeft:
            newnode = trenchNode(parent=trenchnode)
            newnode.left(i=row, j=col)
            newnode.weight += 1
            newnode.heuristic = newnode.manhattanDistance()
            if np.array2string(newnode.state) not in repeated:
                priorityQ.push(newnode)

        if flagRight:
            newnode = trenchNode(parent=trenchnode)
            newnode.right(i=row, j=col)
            newnode.weight += 1
            newnode.heuristic = newnode.manhattanDistance()
            if np.array2string(newnode.state) not in repeated:
                priorityQ.push(newnode)
    return


def getPath(trenchnode):
    stack = list()
    while trenchnode is not None:
        stack.append(trenchnode)
        trenchnode = trenchnode.parent

    while len(stack) > 0:
        top = stack.pop()
        print(f'The best state to expand when g(n) = {top.weight} and h(n) = {top.heuristic} is')
        print(top)
    return


class myHeap(object):
    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        self.index = 0
        if initial is not None:
            self._data = [(key(item), i, item) for i, item in enumerate(initial)]
            self.index = len(self._data)
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), self.index, item))
        self.index += 1

    def pop(self):
        if not self.empty():
            return heapq.heappop(self._data)[2]
        else:
            print("Calling pop on an empty queue. Returning None")
            return None

    def getHeap(self):
        return self._data

    def empty(self):
        return len(self._data) <= 0

    def size(self):
        return len(self._data)


class generalSearch(object):
    def __init__(self, problem=None, queueingHeuristic=None, queueingFunc=None):
        self.problem = problem
        self.queueingHeuristic = queueingHeuristic
        self.queueingFunction = queueingFunc
        return

    def search(self):
        qSize               = 0
        totalNodeExpansions = 0
        finalDepth          = 0
        goalNode            = None
        # nodes =           MAKE-QUEUE(MAKE-NODE(self.problem.getInitialState()))
        nodes               = myHeap(initial=[self.problem], key=self.queueingHeuristic)
        minCostSoFar        = 100
        while not nodes.empty():
            qSize = max(qSize, nodes.size())
            node = nodes.pop()
            if minCostSoFar > node.heuristic:
                minCostSoFar = min(minCostSoFar, node.heuristic)
                print(f'Minimum Heuristic has been updated to {minCostSoFar}!')
            repeated.add(np.array2string(node.state))
            if node.goalTest():
                goalNode = node
                finalDepth = node.weight
                getPath(trenchnode=node)
                return True, qSize, totalNodeExpansions, finalDepth
            else:
                totalNodeExpansions += 1
                self.queueingFunction(nodes, node)
        return False, qSize, totalNodeExpansions, finalDepth


class trenchNode(object):
    def __init__(self, startState: str = None,
                 heuristic: str = 0,
                 weight: int = 0,
                 parent=None
    ):
        if parent is not None:
            self.parent = parent
            self.weight = parent.weight
            self.heuristic = parent.heuristic
            self.state = np.array(parent.state)
        else:
            if startState is None:
                raise ValueError("No state passed for node initialization")
            tempState = list(startState)
            tempStateTop = [-1 for _ in range(len(tempState))]
            tempStateBottom = [int(tempState[i]) for i in range(len(tempState))]
            self.state = np.zeros(shape=(2, len(tempState)))
            self.state[0, :] = tempStateTop
            self.state[0, list(recessPositions)] = 0
            self.state[1, :] = tempStateBottom
            self.state = self.state.astype('int8')
            self.weight = weight
            self.parent = None
            if heuristic == 'Manhattan':
                self.heuristic = self.manhattanDistance()
        return

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)

    def __repr__(self):
        string = f'Trench Node distance: {self.weight + self.heuristic} \nTrench:\n{np.array2string(self.state)}'
        return string  # TODO, verify this later

    def goalTest(self):
        # The sergent has the highest rank, so it's represented as the value 9
        # does the value in row 1 column 0 have a value of 9?
        if self.state[1, 0] == 1:
            return True
        else:
            return False

    def manhattanDistance(self):
        dist = 0
        x = np.where(self.state == 1)
        dist += abs(1 - x[0][0]) + abs(0 - x[1][0])
        return dist

    def findBlanks(self):
        return np.where(self.state == 0)

    def up(self, i=None, j=None):
        # move blank up
        temp                 = self.state[i - 1, j]
        self.state[i - 1, j] = 0
        self.state[i, j]     = temp
        return

    def down(self, i=None, j=None):
        # move blank down
        temp                 = self.state[i + 1, j]
        self.state[i + 1, j] = 0
        self.state[i, j]     = temp
        return

    def left(self, i=None, j=None):
        # move blank left
        temp                 = self.state[i, j - 1]
        self.state[i, j - 1] = 0
        self.state[i, j]     = temp
        return

    def right(self, i=None, j=None):
        # move blank right
        temp                 = self.state[i, j + 1]
        self.state[i, j + 1] = 0
        self.state[i, j]     = temp
        return


def main():
    # ##########################################################################################
    # Main function, Options
    # ##########################################################################################
    parser = argparse.ArgumentParser(description="Nine Men in a Trench Search")
    parser.add_argument('-i', '--input', help='Name of input trace file which should be in cwd')
    parser.add_argument("-m", "--mode",
                        default='3',
                        help='mode 1: Uniform Cost Search; mode 2: A* with Misplaced Tile heuristic; mode 3: \
                              A* with Manhattan Distance Heuristic')
    parser.add_argument('-d', '--debug', help='Debug mode')
    args = parser.parse_args()
    print("Output args:")
    pprint.pprint(args)

    # ##########################################################################################
    # Main
    # ##########################################################################################
    unitTest = False
    if args.input is not None:
        inputTrace = args.input
        print(inputTrace)
    else:
        print('No input passed in, see -h argument for help. Exiting...')
        return

    if args.mode is not None:
        mode = args.mode

    if mode == '1':
        print(mode)
    elif mode == '2':
        print(mode)
    elif mode == '3':
        # mode == 3
        print(mode)
    else:
        print("Error: Invalid mode chosen.")
    # Unit test min heap wrapper
    # ##########################################################################################
    # test = [(4, 'young'), (5, 'old'), (1, 'baby')]
    # myheap = myHeap(initial=test, key=nodeCost)
    # print(myheap.getHeap())
    # myheap.pop()
    # print(myheap.empty())
    # print(myheap.getHeap())
    # myheap.pop()
    # print(myheap.getHeap())
    # myheap.pop()
    # print(myheap.getHeap())
    # print(myheap.empty())
    # ##########################################################################################
    trench    = trenchNode(   startState=inputTrace,
                              weight=0,
                              heuristic='Manhattan'
    )

    genSearch = generalSearch(problem=trench,
                              queueingHeuristic=manhattanDistanceHeuristic,
                              queueingFunc=trenchEnqueue
    )

    answer, maxQueueSize, totalNodesExpanded, solutionDepth = genSearch.search()

    print(f'Solution Exists?    {answer}')
    print(f'Maximum queue size: {maxQueueSize}')
    print(f'Nodes expanded:     {totalNodesExpanded}')
    print(f'Solution depth:     {solutionDepth}')
    return


if __name__ == '__main__':
    """Performs execution delta of the process."""
    pStart = datetime.datetime.now()
    try:
        main()
    except Exception as errorMain:
        print("Fail End Process: {0}".format(errorMain))
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop - pStart))
