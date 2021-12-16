from WaveformConstructorPrimitives_JH import *
import numpy as np
import json
from copy import deepcopy
import math
from scipy.integrate import quad
from scipy.optimize import fsolve


def jsonRead(file):
    with open(str(file), "r") as read_file:
        readDict = json.load(read_file)
    return readDict


def identityPulse(duration=0):
    return Pulse(0, duration, 0)


def identityOp(duration=0):
    return Op(pulseList=[identityPulse(duration)], name="Id")


def identityExpSlice(qSys, duration):
    """Returns an experiment slice, not a full experiment, since is often used as a component of experiments."""
    expSlice = ExpSlice(name="Id")  # Each element is the op on the respective qubit. Not a sequence!
    for qubit in range(qSys.N):
        expSlice.opList.append(identityOp(duration))
    return expSlice


def singleQubitSquareExpSlice(qSys, targetQubitIndex, amp, duration, phase):
    """Returns an experiment slice, not a full experiment, since is often used as a component of experiments."""
    expSlice = ExpSlice(name=qSys.qubits[targetQubitIndex].name+"_square")
    for qubitIndex in range(qSys.N):
        if qubitIndex == targetQubitIndex:
            expSlice.opList.append(Op(pulseList=[Pulse(amp, duration, phase)], name="square"))
        else:
            expSlice.opList.append(identityOp(duration))
    return expSlice


def singleQubitSquareRotationExpSlice(qSys, targetQubitIndex, theta, phi):
    '''Correcting for leakage here assuming this is called in isolation, i.e. not in the middle of a  pulse train'''
    phi_leak = qSys.qubits[targetQubitIndex].leakagePhase
    # phi_leak = 0
    omega_leak = 2 * np.pi * qSys.qubits[targetQubitIndex].maxAmpStrength
    # print(phi_leak, phi_leak/omega_leak)
    duration = theta / (2 * np.pi * qSys.qubits[targetQubitIndex].maxAmpStrength) - phi_leak/(omega_leak/2) # cos^2

    targetQubitRotPulse = Pulse(qSys.maxAPSAmp, duration, phi)

    name = qSys.qubits[targetQubitIndex].name+"_rot_"+str(theta*360/(2*np.pi))

    expSlice = ExpSlice([identityOp(duration)] * qSys.N,
                        name=name)
    expSlice.opList[targetQubitIndex] = Op(pulseList=[targetQubitRotPulse], name=name)

    return expSlice


def singleQubitRamseyExp(qSys, qubitIndex, waitDuration):
    prePio2_slice = singleQubitSquareRotationExpSlice(qSys, qubitIndex, np.pi / 2, 0)
    wait_slice = identityExpSlice(qSys, waitDuration)
    # wait_slice = singleQubitSquareExpSlice(qSys, qubitIndex, 0.3, waitDuration, 0) #For test
    postPio2_slice = singleQubitSquareRotationExpSlice(qSys, qubitIndex, np.pi / 2, 0)
    return Exp([prePio2_slice, wait_slice, postPio2_slice])


def singleQubitRamseyBatchExp(qSys, qubitIndex, waitDurations):
    return BatchExp([singleQubitRamseyExp(qSys, qubitIndex, waitDuration) for waitDuration in waitDurations])


def rabiAmpBatchExp(qSys, q_index, amps, duration, phase):
    return BatchExp([Exp([singleQubitSquareExpSlice(qSys, q_index, amp, duration, phase)]) for amp in amps])


def rabiTimeBatchExp(qSys, q_Index, amp, durations, phase):
    return BatchExp([Exp([singleQubitSquareExpSlice(qSys, q_Index, amp, duration, phase)]) for duration in durations])


def loadOptimSlices(qSys, pulseDataFilePath): #No scaling
    pulseData = jsonRead(pulseDataFilePath)
    expSlices = []
    for gateIndex in range(len(pulseData["gateData"])):
        gateData = pulseData["gateData"][str(gateIndex)]
        tVals = gateData["t"]
        expSlice = ExpSlice(name="G" + str(gateIndex))
        for qubit in qSys.qubits:
            qubitOp = Op()
            for i, t in enumerate(tVals):
                if gateData["shape"][i] == "Square":
                    xVal, yVal = gateData[str(qubit.name)][i]
                    R, phi = pulseCartToPolar(xVal, yVal)
                    qubitOp.pulseList.append(Pulse(R, t, phi))
                elif gateData["shape"][i] == "Gauss":
                    A, phi = gateData[str(qubit.name)][i]
                    qubitOp.pulseList.append(gaussPulse(A, t, phi))
            expSlice.opList.append(qubitOp)
        expSlices.append(expSlice)
    return expSlices


def pulseCartToPolar(X, Y):
    return np.sqrt(X ** 2 + Y ** 2), np.arctan2(Y, X)


def scaleOptimSlice(qSys, expSlice):
    expSlice_temp = deepcopy(expSlice)
    for qubitIndex, qubitOp in enumerate(expSlice_temp.opList):
        for pulse in qubitOp.pulseList:
            pulse.amp *= qSys.maxAPSAmp/qSys.qubits[qubitIndex].RofPhi(pulse.phase)
    return expSlice_temp


def getBestGateExpSlice(pulseDataFilePath, qSys):
    pulseDict = jsonRead(pulseDataFilePath)
    lowestIF = 1
    lowestIFIndex = 0
    for gateIndex, gate in pulseDict["gateData"].items():
        IF = float(gate["IF"])
        if IF < lowestIF:
            lowestIF = IF
            lowestIFIndex = int(gateIndex)
    bestGateExpSlice = optimExpSliceList(qSys, pulseDataFilePath)[lowestIFIndex]
    return bestGateExpSlice


def addTomo(qSys, rotType, batch_exp):
    '''tomography will be added to each experiment in batch_exp'''
    tomo_batch_exp = BatchExp([])

    if rotType == "OPT":
        rotationExpSlices_Q0 = [identityExpSlice(qSys, 0),
                                addPhase_expSlice(qSys.qubits[0].pio2_expSlice, [0], np.pi),
                                addPhase_expSlice(qSys.qubits[0].pio2_expSlice, [0], np.pi / 2)]
        rotationExpSlices_Q1 = [identityExpSlice(qSys, 0),
                                addPhase_expSlice(qSys.qubits[1].pio2_expSlice, [1], np.pi),
                                addPhase_expSlice(qSys.qubits[1].pio2_expSlice, [1], np.pi / 2)]
        tomoSliceLists = [[Q0ExpSlice, Q1ExpSlice] for Q0ExpSlice in rotationExpSlices_Q0
                          for Q1ExpSlice in rotationExpSlices_Q1]  # Has length 9.
    elif rotType == "TAG":
        Id = identityExpSlice(qSys, 0)
        Id.name = "tomo: Id"

        Q0_X = addPhase_expSlice(qSys.qubits[0].TAG_ExpSlice("Pio2", 1), [0], np.pi)
        Q0_X.name = "tomo: Q0 -sx90"
        Q0_Y = addPhase_expSlice(qSys.qubits[0].TAG_ExpSlice("Pio2", 1), [0], np.pi / 2)
        Q0_Y.name = "tomo: Q0 sy90"

        Q1_X = addPhase_expSlice(qSys.qubits[1].TAG_ExpSlice("Pio2", 0), [1], np.pi)
        Q1_X.name = "tomo: Q1 -sx90"
        Q1_Y = addPhase_expSlice(qSys.qubits[1].TAG_ExpSlice("Pio2", 0), [1], np.pi / 2)
        Q1_Y.name = "tomo: Q1 sy90"

        rotationExpSlices_Q0 = [Id, Q0_X, Q0_Y]
        rotationExpSlices_Q1 = [Id, Q1_X, Q1_Y]

        tomoSliceLists = [[rotQ0, rotQ1] for rotQ0 in rotationExpSlices_Q0 for rotQ1 in rotationExpSlices_Q1]
    for exp in batch_exp.expList:
        for tomoExpSliceList in tomoSliceLists:
            expCopy = deepcopy(exp)
            for tomoExpSlice in tomoExpSliceList:
                expCopy.sliceList.append(tomoExpSlice)
            tomo_batch_exp.expList.append(expCopy)

    return tomo_batch_exp


def processTomography(qSys, gateExpList): #batch_exp is only gates
    batch_exp = BatchExp()
    preparationSlices = []

    IdSlice = identityExpSlice(qSys, 0)
    for q1 in [0, 1]:
        negZSlice = qSys.qubits[q1].TAG_ExpSlice("Pi", int(not q1))
        negZSlice.name = "Q"+str(q1)+".-Z"
        YSlice = addPhase_expSlice(qSys.qubits[q1].TAG_ExpSlice("Pio2", int(not q1)), [q1], np.pi / 2)
        YSlice.name = "Q"+str(q1)+".Ypio2"
        negYSlice = addPhase_expSlice(qSys.qubits[q1].TAG_ExpSlice("Pio2", int(not q1)), [q1], -np.pi / 2)
        negYSlice.name = "Q"+str(q1)+".-Ypio2"
        negXSlice = addPhase_expSlice(qSys.qubits[q1].TAG_ExpSlice("Pio2", int(not q1)), [q1], -np.pi)
        negXSlice.name = "Q"+str(q1)+".-Xpio2"
        XSlice = addPhase_expSlice(qSys.qubits[q1].TAG_ExpSlice("Pio2", int(not q1)), [q1], 0)
        XSlice.name = "Q"+str(q1)+".Xpio2"

        preparationSlices.append([IdSlice, negZSlice, YSlice, negYSlice, negXSlice, XSlice])

    for gateExp in gateExpList:
        for prepSlice1 in preparationSlices[0]:
            for prepSlice2 in preparationSlices[1]:
                gateExpPrep = deepcopy(gateExp)
                gateExpPrep.sliceList = [prepSlice1, prepSlice2] + gateExpPrep.sliceList
                batch_exp.expList.append(gateExpPrep)
    full_batch_exp = addTomo(qSys, "TAG", batch_exp)

    return full_batch_exp


def numDivs_n(n, approxExpsPerDiv):
    divisors = [x for x in range(1, int(math.sqrt(n))+1) if n % x == 0]
    opps = [int(n/x) for x in divisors] # get divisors > sqrt(n) by division instead
    factors = divisors+opps
    closestTo_val = factors[np.argmin(np.array([abs(x-approxExpsPerDiv) for x in factors]))]
    return int(n/closestTo_val)


def addCalib(qSys, batch_exp, approxExpsPerDiv):
    numDivs = numDivs_n(len(batch_exp.expList), approxExpsPerDiv)
    batch_exp_temp = BatchExp()
    expsPerDiv = int(len(batch_exp.expList) / numDivs)

    for i in range(numDivs):
        batch_exp_temp.expList += qSys.basisStatesExps(TAG_or_not=True) + batch_exp.expList[i * expsPerDiv:(i + 1) * expsPerDiv]

    return batch_exp_temp, numDivs


def gaussPulse(maxA, t, phi):
    return Pulse(maxA, t, phi, lambda x: pulseGauss(x, t), "Gaussian")


def gaussPulse_preserveInteg(maxA, amp, duration, phase):
    integVal = amp*duration

    APrime = maxA*amp/abs(amp)
    TPrime = fsolve(lambda endT: quad(lambda y: APrime * pulseGauss(y, endT), 0, endT)[0] - integVal,
                    4 * duration)[0]
    print("TPrime = ", TPrime)
    print("APrime = ", APrime)
    print("Sigma = ", TPrime / 7.0)
    print(quad(lambda y: APrime * pulseGauss(y, TPrime), 0, TPrime)[0], integVal)
    shapeFunc = lambda x: pulseGauss(x, TPrime)
    return Pulse(maxA, TPrime, phase, shapeFunc, "Gaussian")


def pulseGauss(x, TPrime):
    sigma = TPrime / 7.0
    return np.exp(-(x - TPrime / 2.0) ** 2 / (2 * sigma ** 2))
