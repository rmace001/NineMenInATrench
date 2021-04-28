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

"""


import os
import argparse
import pprint
import traceback
import datetime
import time
import numpy as np
import pandas as pd

class trenchSearch(object):
    def __init__(self, startState: str = None, searchOption: str = None, recessPositions: list = [3, 5, 7]):
        tempState = list(startState)
        tempStateBottom = [int(tempState[i]) for i in range(len(tempState))]
        self.state = np.zeros(shape=(2, 9))
        self.state[1, :] = tempStateBottom
        self.state = self.state.astype('int8')
        return

    def goalTest(self):
        pass




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

    args = parser.parse_args()
    print("Output args:")
    pprint.pprint(args)
    ##############################################
    # Main
    ##############################################
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
    search = trenchSearch(startState=inputTrace, searchOption=mode)
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
