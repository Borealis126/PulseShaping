import subprocess
import math
import numpy as np

def initStudy(studyPath):
    subprocess.call("if not exist " + str(studyPath) + " mkdir " + str(studyPath), shell=True)

    resultsFolder = studyPath / 'results' / 'gate' / 'run0'
    initialStatesResultsFolder = studyPath / 'results' / 'initialStates' / 'run0'
    gatePulseFolder = studyPath / 'pulses' / 'gate'
    initialStatesPulseFolder = studyPath / 'pulses' / 'initialStates'

    for folder in [resultsFolder, initialStatesResultsFolder, gatePulseFolder, initialStatesPulseFolder]:
        subprocess.call("if not exist " + str(folder) + " mkdir " + str(folder), shell=True)


def expIndex(numInitialStates, numPostRotations, gateIndex, initialStateIndex, postRotationIndex):
    return gateIndex*(numInitialStates*numPostRotations)+initialStateIndex*numPostRotations+postRotationIndex

