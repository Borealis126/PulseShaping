from copy import deepcopy

# [a,b,c,d] this is a "pulse".
# [[a,b,c,d],[e,f,g,h]] this is an "operation".
# [[[a,b,c,d],[e,f,g,h]],[[i,j,k,l],[m,n,o,p]]] this is the "list of operations defining a sequence for a single qubit", just referred to as a "sequence". Every sequence is followed by a readout pulse.

# An "experiment slice" consists of a single operation for each of N qubits running in parallel. It is structurally identical to a sequence.
# An "experiment" consists of a single sequence for each of N qubits running in parallel. [Utheta*Utheta] is a QGL_exp, the QGL representation of an experiment.
# A "batch experiment" consists of multiple sequential experiments. This is what compile_to_hardware needs takes as input.

# The ESL test is a batch experiment. With 10 taus it has has 4+10*9=94 experiments, where each sequence in the experiment (after the first 4 calibration ones) consists
# of a drive operation followed by a tomography operation. Before compiling to hardware, each experiment is flattened to an experiment slice (each sequence becomes a single operation)
# The drive operation contains 5 pulses, and the tomography operation contains three (two-axis gate).


class Pulse:
    def __init__(self, amp=1, duration=1, phase=1, shapeFunc=lambda t: 1, shapeFuncName="Square"):
        self.amp = amp
        self.duration = duration
        self.phase = phase
        self.detuning = 0
        self.shapeFunc = shapeFunc
        self.shapeFuncName = shapeFuncName
    @property
    def tzForm(self):
        return [self.amp, self.duration, self.phase, self.detuning, self.shapeFunc]

    @property
    def view(self):
        return ["A: "+str(round(self.tzForm[0], 2)),
                "t: "+str(round(self.tzForm[1]/1e-9, 2))+"ns",
                "phi: "+str(round(self.tzForm[2], 2)),
                "shapeFuncName: ", self.shapeFuncName]


class Op:  # A list of pulses
    def __init__(self, pulseList=None, name="Op"):
        self.name = name
        if pulseList is None:
            pulseList = []
        self.pulseList = pulseList

    @property
    def view(self):
        return [["A: "+str(round(pulse.tzForm[0], 2)),
                 "t: "+str(round(pulse.tzForm[1]/1e-9, 2))+"ns",
                 "phi: "+str(round(pulse.tzForm[2], 2))]
                for pulse in self.pulseList]


class ExpSlice(object): #A list of ops, one for each qubit. Each op must be the same duration!
    def __init__(self, opList=None, name="Slice"):
        self.name = name
        if opList is None:
            opList = []
        self.opList = opList

    @property
    def view(self):
        return ["Q" + str(i) + ": " + op.name for i, op in enumerate(self.opList)]

    @property
    def duration(self):
        return sum([pulse.duration for pulse in self.opList[0].pulseList])


class ExpSliceTAG(ExpSlice):
    def __init__(self, opList=None, name="TAG", otherQubitIndex=0, phaseComp=0):
        super(ExpSliceTAG, self).__init__(opList=opList, name=name)
        self.otherQubitIndex = otherQubitIndex
        self.phaseComp = phaseComp


class Exp:  # A list of ExpSlices
    def __init__(self, sliceList=None):
        if sliceList is None:
            sliceList = list()
        self.sliceList = sliceList

    @property
    def view(self):
        return [expSlice.name for expSlice in self.sliceList]

    @property
    def viewAll(self):
        print(self.view)
        for sliceIndex, expSlice in enumerate(self.sliceList):
            print("    " + expSlice.name + ":" + str(expSlice.view))
            for opIndex, op in enumerate(expSlice.opList):
                print("        " + op.name + ":")
                for pulseIndex, pulse in enumerate(op.pulseList):
                    print("            " + str(pulse.view))


class BatchExp: # A list of Exps
    def __init__(self, expList=None):
        if expList is None:
            expList = []
        self.expList = expList


def addPhase_op(op, phase):
    op_new = deepcopy(op)
    for pulse in op_new.pulseList:
        pulse.phase += phase
    return op_new


def addPhase_expSlice(expSlice, qubitIndices, phase):
    expSlice_new = deepcopy(expSlice)
    for qubitIndex in qubitIndices:
        expSlice_new.opList[qubitIndex] = addPhase_op(expSlice_new.opList[qubitIndex], phase)
    return expSlice_new


def addExpSliceToAll(batch_exp, expSlice):#Adds an expslice to every experiment in the batch. Used for gates.
    for exp in batch_exp.expList:
        exp.sliceList.append(expSlice)


def addTAGPhases_exp(exp):
    exp_new = deepcopy(exp)
    for sliceIndex, slice in enumerate(exp_new.sliceList):
        if isinstance(slice, ExpSliceTAG):
            for i in range(sliceIndex+1, len(exp_new.sliceList)):
                exp_new.sliceList[i] = addPhase_expSlice(exp_new.sliceList[i], [slice.otherQubitIndex], slice.phaseComp)
            # exp_new.viewAll
    return exp_new
