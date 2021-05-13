#!/usr/bin/python3
# -*- coding: utf-8 -*-
# *****************************************************************************/
# * Copyright 2021 UCR CS205, Artificial Intelligence. All Rights Reserved.
# * Closed source repository. Do not share any content without permission written from Rogelio Macedo.
# * Authors: Rogelio Macedo
# * Template Credit: Joseph Tarango
# *****************************************************************************/


"""

Usage: $: python3 trenchPuzzleSearch.py -i 0234567891 -r 357
Args:
    -i: input trench
    -r: input recess positions starting at position 0

"""

import argparse
import pprint
import traceback
import datetime
import numpy as np
import heapq

repeated = set()


def aStar(node=None):
    return node.weight + node.heuristic


# Trench queueing Function
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
            if col not in trenchnode.recessPos:
                flagUp = False

        if flagUp:
            newnode               = trenchNode(parent=trenchnode)
            newnode.up(i=row, j=col)
            if np.array2string(newnode.state) not in repeated:
                newnode.parent    = trenchnode
                newnode.weight    = trenchnode.weight + 1
                newnode.heuristic = newnode.manhattanDistance()
                priorityQ.push(newnode)

        if flagDown:
            newnode               = trenchNode(parent=trenchnode)
            newnode.down(i=row, j=col)
            if np.array2string(newnode.state) not in repeated:
                newnode.parent    = trenchnode
                newnode.weight    = trenchnode.weight + 1
                newnode.heuristic = newnode.manhattanDistance()
                priorityQ.push(newnode)

        if flagLeft:
            newnode               = trenchNode(parent=trenchnode)
            newnode.left(i=row, j=col)
            if np.array2string(newnode.state) not in repeated:
                newnode.parent    = trenchnode
                newnode.weight    = trenchnode.weight + 1
                newnode.heuristic = newnode.manhattanDistance()
                priorityQ.push(newnode)

        if flagRight:
            newnode               = trenchNode(parent=trenchnode)
            newnode.right(i=row, j=col)
            if np.array2string(newnode.state) not in repeated:
                newnode.parent    = trenchnode
                newnode.weight    = trenchnode.weight + 1
                newnode.heuristic = newnode.manhattanDistance()
                priorityQ.push(newnode)
    return


# Get solution path
def getPath(trenchnode=None):
    stack = list()
    while trenchnode is not None:
        stack.append(trenchnode)
        trenchnode = trenchnode.parent

    top = stack.pop()
    print('Start state:')
    print(top)
    while len(stack) > 0:
        top = stack.pop()
        print(f'The best state to expand when g(n) = {top.weight} and h(n) = {top.heuristic} is')
        print(top)
    return


# Queue class with user-defined node
class myHeap(object):
    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        self.index = 0
        if initial is not None:
            self.data = [(key(item), i, item) for i, item in enumerate(initial)]
            self.index = len(self.data)
            heapq.heapify(self.data)
        else:
            self.data = []

    def push(self, item):
        heapq.heappush(self.data, (self.key(item), self.index, item))
        self.index += 1

    def pop(self):
        if not len(self.data) <= 0:
            return heapq.heappop(self.data)[2]
        else:
            print("Calling pop on an empty queue. Returning None")
            return None

    def getHeap(self):
        return self.data

    def empty(self):
        return len(self.data) <= 0

    def size(self):
        return len(self.data)


class generalSearch(object):
    def __init__(self, problem=None, queueingHeuristic=None, queueingFunc=None):
        self.problem           = problem
        self.queueingHeuristic = queueingHeuristic
        self.queueingFunction  = queueingFunc
        return

    def search(self):
        qSize                  = 0
        totalNodeExpansions    = 0
        nodes                  = myHeap(initial=[self.problem], key=self.queueingHeuristic)
        # while not nodes.empty():
        while not len(nodes.data) <= 0:  # can use statement above but using this for performance
            nodeSize = nodes.size()
            qSize    = nodeSize if nodeSize > qSize else qSize
            node     = nodes.pop()
            repeated.add(np.array2string(node.state))
            if node.goalTest():
                getPath(trenchnode=node)
                return True, qSize, totalNodeExpansions, node.weight
            else:
                totalNodeExpansions += 1
                self.queueingFunction(nodes, node)
        return False, qSize, totalNodeExpansions, node.weight


# Class for a trench node, where the 2-D array is a NumPy array
# Row 0 is the recess row
# Row 1 is the trench with men
class trenchNode(object):
    recessPos = None

    def __init__(self, startState: str = None, heuristic: str = None, weight: int = 0, parent=None):
        if parent is not None:
            self.state       = np.array(parent.state)
        else:
            if startState is None:
                raise ValueError("No state passed for node initialization")

            tempState        = list(startState)
            tempStateTop     = [-1 for _ in range(len(tempState))]
            tempStateBottom  = [int(tempState[i]) for i in range(len(tempState))]
            self.state       = np.zeros(shape=(2, len(tempState)))
            self.state[0, :] = tempStateTop

            self.state[0, list(trenchNode.recessPos)] = 0

            self.state[1, :] = tempStateBottom
            self.state       = self.state.astype('int8')
            self.weight      = weight
            self.parent      = None
            if heuristic == 'Manhattan':
                self.heuristic = self.manhattanDistance()
        return

    def __repr__(self):
        return f'f(n) = g(n) + h(n) = {self.weight + self.heuristic}. Trench:\n{np.array2string(self.state)}\n'

    def goalTest(self):
        # Does the value in row 1 column 0 have a value of 1?
        return self.state[1, 0] == 1

    def manhattanDistance(self):
        x = np.where(self.state == 1)
        return abs(1 - x[0][0]) + abs(0 - x[1][0])

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
    parser.add_argument('-i', '--input', help='Input Trench')
    parser.add_argument('-r', '--recesses', help='Input Recess Positions')
    parser.add_argument("-m", "--mode", default='1', help='mode 1: A* with Manhattan Distance Heuristic')
    parser.add_argument('-d', '--debug', help='Debug mode')
    args = parser.parse_args()
    print("Args:")
    pprint.pprint(args)

    # ##########################################################################################
    # Main
    # ##########################################################################################
    if args.input is not None:
        inputTrace = args.input
    else:
        print('No input passed in, see -h argument for help. Exiting...')
        return

    if args.mode is not None:
        mode = args.mode

    if mode == '1':
        pass
    else:
        print("Error: Invalid mode chosen.")

    trenchNode.recessPos = set([int(i) for i in args.recesses])
    trench               =    trenchNode(startState=inputTrace,
                                         weight=0,
                                         heuristic='Manhattan'
    )

    genSearch            = generalSearch(problem=trench,
                                         queueingHeuristic=aStar,
                                         queueingFunc=trenchEnqueue
    )

    answer, maxQueueSize, totalNodesExpanded, solutionDepth = genSearch.search()

    print(f'Solution Exists?             {answer}')
    print(f'Maximum queue size:          {maxQueueSize}')
    print(f'Nodes expanded:              {totalNodesExpanded}')
    print(f'Solution depth:              {solutionDepth}')
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
    print("Execution time:              " + str(qStop - pStart))
