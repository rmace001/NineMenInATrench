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
maxQueueSize = 0       # reset this when the set is reset
totalNodesExpanded = 0 # reset this when the set is reset
solutionDepth = 0


def compareTrenchNodes(node=None):
    return node


# unit test node cost function for min heap wrapper class
def nodeCost(node):
    return node[0]


def manhattanDistanceHeuristic(node=None):
    return node.weight + node.manhattanDistance()


def uniformCost(node=None):
    return node.weight


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
        return heapq.heappop(self._data)[2]

    def getHeap(self):
        return self._data


class generalSearch(object):
    def __init__(self, problem=None, queueingHeuristic=None):
        self.problem = problem
        self.queueingHeuristic = queueingHeuristic
        return

    def search(self):
        # nodes = makeQueue(makeNode(self.problem.getInitialState()))
        problemList = list()
        problemList.append(self.problem)
        nodes = myHeap(initial=problemList, key=self.queueingHeuristic)
        print(nodes.getHeap())
        return True


class trenchNode(object):
    def __init__(self, startState: str = None,
                 recessPositions: list = [3, 5, 7],
                 heuristic: int = 0,
                 weight: int = 0,
                 parent=None
    ):
        if parent is not None:
            self.weight = self.parent.weight
            self.heuristic = self.parent.heuristic
            self.state = self.parent.state
            self.recessPositions = parent.recessPositions
        else:
            if startState is None:
                raise ValueError("No state passed for node initialization")
            tempState = list(startState)
            tempStateTop = [-1 for i in range(len(tempState))]
            tempStateBottom = [int(tempState[i]) for i in range(len(tempState))]
            self.state = np.zeros(shape=(2, 9))
            self.state[0, :] = tempStateTop
            self.state[0, recessPositions] = 0
            self.state[1, :] = tempStateBottom
            self.state = self.state.astype('int8')
            self.weight = weight
            self.parent = None
            self.recessPositions = recessPositions
            if heuristic == 'Manhattan':
                self.heuristic = self.manhattanDistance()
        return

    def printTrench(self):
        print(self.state)
        return

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)

    def __repr__(self):
        string = f'Trench Node distance: {self.weight + self.heuristic} \nTrench:\n{np.array2string(self.state)}'
        return string  # TODO, verify this later

    def goalTest(self):
        # The sergent has the highest rank, so it's represented as the value 9
        # does the value in row 1 column 0 have a value of 9?
        print(self.state[1, 0])
        if self.state[1, 0] == 9:
            return True
        else:
            return False

    def manhattanDistance(self):
        dist = 0
        for i in range(len(self.state[1, :])):
            if self.state[1, i] != 9:
                dist += 1
        return dist


def main():
    ##############################################
    # Main function, Options
    ##############################################
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
    ##############################################
    # Main
    ##############################################
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
    # test = [(4, 'young'), (5, 'old'), (1, 'baby')]
    # myheap = myHeap(initial=test, key=nodeCost)
    # print(myheap.getHeap())
    # myheap.pop()
    # print(myheap.getHeap())
    trench = trenchNode(startState=inputTrace, weight=0, heuristic='Manhattan')
    genSearch = generalSearch(problem=trench, queueingHeuristic=manhattanDistanceHeuristic)
    genSearch.search()
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
