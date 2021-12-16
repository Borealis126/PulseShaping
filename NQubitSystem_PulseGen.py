# Must be python 2.7 compatible!
import numpy as np
import math
import sys
import json
from pathlib import Path
from copy import deepcopy

sys.path.append('O:\\68707\\JoelHoward\\DataAnalysis')
pyQLabDir = r"C:\Users\68707\Desktop\LV_JH\PyQLab-Working"
sys.path.append(pyQLabDir)
from NQubitSystem import *
import AdvancedWaveforms_JH as wfm_adv
from WaveformConstructorPrimitives_JH import *


class Transition_PG(Transition):
    def __init__(self, states, freq, dipoleStrength):
        super(Transition_PG, self).__init__(states)
        self.freq = freq
        self.dipoleStrength = dipoleStrength  # Frequency of the dipole rabi.


class Qubit_PG(Qubit):
    def __init__(self, index, otherQubitIndices):
        super(Qubit_PG, self).__init__(index, otherQubitIndices)
        self.ch_No = int(self.index - 2 * math.floor(self.index / 2)) + 1  # Even=1, odd=2
        self.QGLChannelStr = 'q' + str(self.index + 1)  # QGL indexes at 1.
        self.modFreq = 0  # Hz
        self.angleError = 0  # Radians
        self.detuning = 0  # Hz
        self.maxAPSAmp = 0
        self.maxAmpStrength = 0  # Rabi strength for a pyQLab pulse set to maxAPSAmp. Hz.
        self.leakagePhase = 0 #Phase offset for sin**2(Omega*t/2)
        self.pio2_opt_expSlice = []
        self.pi_opt_expSlice = []
        self.phiRPairs = np.array([])

        tagParams = {i: {"CF": {"Amp": 0, "Time(ns)": 0}, "G": {"Amp": 0, "Time(ns)": 0}, "phaseComp": 0} for i in
                     ["Pio2", "Pi"]}
        tagParams["ZZ"] = 0
        self.TAG = {i: deepcopy(tagParams) for i in self.otherQubitIndices}

    def TAG_ExpSlice(self, rot, otherQubitIndex):
        TAGdict = self.TAG[otherQubitIndex][rot]
        ampCF = TAGdict["CF"]["Amp"]
        lengthCF = TAGdict["CF"]["Time(ns)"]*1e-9
        ampG = TAGdict["G"]["Amp"]
        lengthG = TAGdict["G"]["Time(ns)"]*1e-9

        name = self.name+"TAG-"+rot
        expSlice = ExpSliceTAG([0]*self.numQubits, name, otherQubitIndex, self.TAG[otherQubitIndex][rot]["phaseComp"])

        CFPulse = Pulse(ampCF, lengthCF, 0)
        GPulse = Pulse(ampG, lengthG, 0)
        # CFPulse = wfm_adv.gaussPulse_preserveInteg(0.8, ampCF, lengthCF, 0)
        # GPulse = wfm_adv.gaussPulse_preserveInteg(0.8, ampG, lengthG, 0)

        TAGOp = Op(pulseList=[deepcopy(CFPulse), deepcopy(GPulse), deepcopy(CFPulse)], name=name)
        expSlice.opList[self.index] = TAGOp
        for qubitIndex in self.otherQubitIndices:
            expSlice.opList[qubitIndex] = wfm_adv.identityOp(2*CFPulse.duration+GPulse.duration)

        return expSlice

    def TAG_duration(self, rot, otherQubitIndex):
        TAGdict = self.TAG[otherQubitIndex][rot]
        lengthCF = TAGdict["CF"]["Time(ns)"] * 1e-9
        lengthG = TAGdict["G"]["Time(ns)"] * 1e-9
        return 2*lengthCF+lengthG

    def RofPhi(self, phi):
        '''Returns a number (nominally close to unity) which is is the fraction Omega/maxAmpStrength for that phase.'''
        return np.interp(phi, self.phiRPairs[:, 0], self.phiRPairs[:, 1])


class NQubitSystem_PG(NQubitSystem):
    def __init__(self, paramsFilePath):
        super(NQubitSystem_PG, self).__init__(paramsFilePath=paramsFilePath)
        self.maxAPSAmp = 0
        self.calibrationDirectory = str()

        self.loadQSysParams_PG()

    def twoAxisGateTomo(self, q0Index, q1Index):
        """Returns an array of the tomographic pulses for each qubit in the format required for Tongyu's function."""
        q0 = self.qubits[q0Index]
        q1 = self.qubits[q1Index]
        Q0TomoRotationOps = [Op([identityPulse(0.3e-6)]),
                             q0.twoAxisPio2Op(controlQubitIndex=q1Index, phase=np.pi),
                             q0.twoAxisPio2Op(controlQubitIndex=q1Index,
                             phase=np.pi / 2)]  # Looks like a sequences, but here is just a list of ops.
        phaseComp = q0.twoAxisPio2[q1Index]["phaseComp"]
        Q1TomoRotationOps = [Op([q1.identityPulse(0.3e-6)]),
                             q1.twoAxisPio2Op(controlQubitIndex=0, phase=np.pi + phaseComp),
                             q1.twoAxisPio2Op(controlQubitIndex=0, phase=np.pi / 2 + phaseComp)]
        Q0TomoOps = []
        Q1TomoOps = []
        for i in Q0TomoRotationOps:
            for j in Q1TomoRotationOps:
                Q0TomoOps.append(i)
                Q1TomoOps.append(j)
        return Q0TomoOps, Q1TomoOps  # Format is [Q0Tomo0Op,Q0Tomo1Op...Q1Tomo8Op],[Q1Tomo0Op,Q1Tomo1Op,...,Q1Tomo8Op]

    def TAG_matlab(self, q0Index, q1Index):
        # dr =|01> - |11>/|00>-|10>

        SWIPHT_delta = self.twoQubitValues[min(q0Index, q1Index)][max(q0Index, q1Index)]["ZZ"]

        state_00 = self.stateList([[q0Index, 0], [q1Index, 0]])
        state_10 = self.stateList([[q0Index, 1], [q1Index, 0]])
        state_01 = self.stateList([[q0Index, 0], [q1Index, 1]])
        state_11 = self.stateList([[q0Index, 1], [q1Index, 1]])

        dr = self.transitions[transitionString(state_str(state_01), state_str(state_11))].dipoleStrength / \
             self.transitions[transitionString(state_str(state_00), state_str(state_10))].dipoleStrength

        def ftheta_G_2(rotation_angle, dr):
            return SWIPHT_delta * (1 / np.tan(2 * np.arccos(
                (np.sqrt((2 * np.pi / rotation_angle) ** 2 + 8 * (1 + dr)) - (2 * np.pi / rotation_angle)) / (4)))) / dr

        def ftheta_G_tot_2(rotation_angle, dr):
            return SWIPHT_delta / np.sin(2 * np.arccos(
                (np.sqrt((2 * np.pi / rotation_angle) ** 2 + 8 * (1 + dr)) - (2 * np.pi / rotation_angle)) / (4)))

        def fthetaO2_CF_2(rotation_angle, dr):
            return SWIPHT_delta * (1 / np.tan(np.arccos(
                (np.sqrt((2 * np.pi / rotation_angle) ** 2 + 8 * (1 + dr)) - (2 * np.pi / rotation_angle)) / (4)))) / dr

        def fthetaO2_CF_tot_2(rotation_angle, dr):
            return SWIPHT_delta / np.sin(np.arccos(
                (np.sqrt((2 * np.pi / rotation_angle) ** 2 + 8 * (1 + dr)) - (2 * np.pi / rotation_angle)) / (4)))

        def roundForHardware(RabiCF, t_CF_ns, RabiG, t_G_ns):
            T_total = 2*t_CF_ns+t_G_ns
            addedT = np.round(T_total/(10.0/3))*(10.0/3)-T_total

            t_CF_new = t_CF_ns+addedT/3
            t_G_new = t_G_ns+addedT/3

            RabiCF_new = RabiCF*t_CF_ns/t_CF_new
            RabiG_new = RabiG * t_G_ns / t_G_new


            return RabiCF_new, t_CF_new, RabiG_new, t_G_new

        Rotation_angle_Pi = np.pi  # this is the angle rotated for off resonance transition
        Rotation_angle_PiO2 = np.pi / 2  # this is the angle rotated for off resonance transition

        RabiCF_Pi = fthetaO2_CF_2(Rotation_angle_Pi, dr)
        RabiG_Pi = ftheta_G_2(Rotation_angle_Pi, dr)

        RabiCF_Pio2 = fthetaO2_CF_2(Rotation_angle_PiO2, dr)
        RabiG_Pio2 = ftheta_G_2(Rotation_angle_PiO2, dr)

        T_CF_Pi = 0.5 / fthetaO2_CF_tot_2(Rotation_angle_Pi, dr) * 1e9
        T_G_Pi = Rotation_angle_Pi / (2 * np.pi) / ftheta_G_tot_2(Rotation_angle_Pi, dr) * 1e9

        T_CF_Pio2 = 0.5 / fthetaO2_CF_tot_2(Rotation_angle_PiO2, dr) * 1e9
        T_G_Pio2 = Rotation_angle_PiO2 / (2 * np.pi) / ftheta_G_tot_2(Rotation_angle_PiO2, dr) * 1e9

        #Round to the nearest 10/3 ns
        RabiCF_Pi, T_CF_Pi, RabiG_Pi, T_G_Pi = roundForHardware(RabiCF_Pi, T_CF_Pi, RabiG_Pi, T_G_Pi)
        RabiCF_Pio2, T_CF_Pio2, RabiG_Pio2, T_G_Pio2 = roundForHardware(RabiCF_Pio2, T_CF_Pio2, RabiG_Pio2, T_G_Pio2)

        temp = [RabiCF_Pi, RabiG_Pi, RabiCF_Pio2, RabiG_Pio2]

        tempAmp = [i / self.qubits[q0Index].maxAmpStrength * self.maxAPSAmp for i in temp]

        AmpCF_Pi = tempAmp[0]
        AmpG_Pi = tempAmp[1]
        AmpCF_Pio2 = tempAmp[2]
        AmpG_Pio2 = tempAmp[3]

        pio2_phaseComp = SWIPHT_delta / 2 * (2 * T_CF_Pio2 + T_G_Pio2)*1e-9
        pi_phaseComp = SWIPHT_delta / 2 * (2 * T_CF_Pi + T_G_Pi)*1e-9

        pio2_CF_return = [AmpCF_Pio2, T_CF_Pio2, pio2_phaseComp]
        pio2_G_return = [AmpG_Pio2, T_G_Pio2, pio2_phaseComp]

        pi_CF_return = [AmpCF_Pi, T_CF_Pi, pi_phaseComp]
        pi_G_return = [AmpG_Pi, T_G_Pi, pi_phaseComp]

        return [[pio2_CF_return, pio2_G_return], [pi_CF_return, pi_G_return]]

    def ESL_NO_Tau_Experiments(self, q0Index, q1Index, ampsQ0, ampsQ1, tauValue):
        """Returns a list of experiments (structurally identical to a batch_exp, but used differently)."""
        numDriveSegments = len(ampsQ0)
        segLength = tauValue / numDriveSegments
        Q0DriveOp = [[i, segLength, 0, self.qubits[0].detuning] for i in ampsQ0]  # Phase is zero for Zhexuan's pulses.
        Q1DriveOp = [[i, segLength, 0, self.qubits[1].detuning] for i in ampsQ1]
        '''At this point we have the drive. Now add tomography. First, get the tomographic sequences.'''
        Q0TomoOps, Q1TomoOps = self.twoAxisGateTomo(q0Index, q1Index)
        experiments = []

        for i in range(len(Q0TomoOps)):
            Q0Seq = [Q0DriveOp, Q0TomoOps[i]]
            Q1Delay = sum([pulse[1] for pulse in Q0TomoOps[i]])
            Q1Seq = [Q1DriveOp, [identityPulse(Q1Delay)], Q1TomoOps[i]]

            experiments.append(self.buildExp([[q0Index, Q0Seq], [q1Index, Q1Seq]]))
        return experiments  # A list of 9 experiments corresponding to tomography over the one tauValue.

    def loadQSysParams_PG(self):

        self.calibrationDirectory = Path(self.data["Calibration Directory"])
        for index, qubit in enumerate(self.qubits):
            self.qubits[index] = Qubit_PG(qubit.index, qubit.otherQubitIndices)  # Upgrade qubit class
        # Drive parameters

        self.maxAPSAmp = self.data["maxAPSAmp"]
        for qubitIndex, q in enumerate(self.qubits):
            q.modFreq = self.data["Q" + str(qubitIndex) + " modFreq"]
            q.angleError = self.data["Q" + str(qubitIndex) + " angleError"]*np.pi/180
            q.maxAmpStrength = self.data["Q" + str(qubitIndex) + " maxAmpStrength (MHz)"] * 1e6
            q.leakagePhase = self.data["Q" + str(qubitIndex) + " Leakage(rad)"]
            # Load R of Phi
            phiRPairs_unsorted = [[float(key[10:-1]), val] for key, val in self.data.items()
                                  if "Q"+str(qubitIndex)+" R(phi) " in key]
            q.phiRPairs = np.array(sorted(phiRPairs_unsorted, key=lambda x: x[0]))

        # Load Transitions
        for transitionStr, transition in self.transitions.items():
            freq = self.data[transitionStr + " Freq (GHz)"] * 1e9
            dipoleStrength = self.data[transitionStr + " Dipole Strength (MHz)"] * 1e6
            self.transitions[transitionStr] = Transition_PG(transition.states, freq,
                                                            dipoleStrength)  # Upgrade transition class

        # Load two-qubit values (values that are order-independent are always lowerval, higher val)
        for pair in twoQubitPairs_ordered(self.N):
            self.twoQubitValues[pair[0]] = dict()
            self.twoQubitValues[pair[0]][pair[1]] = dict()

            if pair == [0, 1]:
                self.twoQubitValues[pair[0]][pair[1]]["ZZ"] = (self.data['0-1|1-1 Freq (GHz)']
                                                               - self.data['0-0|1-0 Freq (GHz)']) * 1e9
            if pair == [0, 1] or pair == [1, 0]: #Target, control
                targetQubitIndex, controlQubitIndex = pair

                pairIndex = 0
                if pair == [0, 1]:
                    pairIndex = 1
                elif pair == [1, 0]:
                    pairIndex = 2
                self.qubits[targetQubitIndex].TAG[controlQubitIndex]["ZZ"] = self.twoQubitValues[0][1]["ZZ"]
                separator = "_"
                for rot in ["Pio2", "Pi"]:
                    phaseCompKey = separator.join(["TAG", str(pairIndex), rot, "phaseComp"])
                    self.qubits[targetQubitIndex].TAG[controlQubitIndex][rot]["phaseComp"] = self.data[phaseCompKey]
                    for pulse in ["CF", "G"]:
                        for param in ["Amp", "Time(ns)"]:
                            valKey = separator.join(["TAG", str(pairIndex), rot, pulse, param])
                            if valKey in self.data:
                                self.qubits[targetQubitIndex].TAG[controlQubitIndex][rot][pulse][param] = self.data[valKey]

    def basisStatesExps(self, TAG_or_not):
        exps = list()

        # |00>
        exps.append(Exp([wfm_adv.identityExpSlice(self, 0)]))

        # |01>, |10>
        if TAG_or_not:
            exps.append(Exp([self.qubits[1].TAG_ExpSlice("Pi", 0)]))
            exps.append(Exp([self.qubits[0].TAG_ExpSlice("Pi", 1)]))
        else:
            exps.append(Exp([wfm_adv.singleQubitSquareRotationExpSlice(self, 1, np.pi, 0)]))
            exps.append(Exp([wfm_adv.singleQubitSquareRotationExpSlice(self, 0, np.pi, 0)]))
        # |11>, TAG
        exps.append(Exp([self.qubits[0].TAG_ExpSlice("Pi", 1), self.qubits[1].TAG_ExpSlice("Pi", 0)]))
        return exps


def durationString(duration):
    return str(int(duration * 1e9)) + "ns"


def jsonRead(file):
    with open(file, "r") as read_file:
        readDict = json.load(read_file)
    return readDict




