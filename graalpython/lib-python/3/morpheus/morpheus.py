"""
Python package to support 'morpheus' rewrites of linear algebra operations
"""

import polyglot
import numpy as np
import numpy.core.numeric as N
from numpy.matrixlib.defmatrix import matrix

class MorpheusInvalidArgsError(TypeError):
    pass

class TensorFromNumpy(object):
    
    def __init__(self,  numpyArr):
        # TODO: validate!
        self.numpyArr = numpyArr

    def rightMatrixMultiply(self, otherArr):
        return numpyModule.matmul(self.numpyArr, otherArr)

    def leftMatrixMultiply(self, otherArr):
        return numpyModule.matmul(otherArr, self.numpyArr)

    def scalarAddition(self, scalar):
        return self.numpyArr + scalar

    def scalarMultiplication(self, scalar):
        return self.numpyArr * scalar

    def crossProduct(self):
        pass
        #return numpyModule.matmul(otherArr, self.numpyArr)

    # TODO: Do we want this?
    def transpose(self):
        pass

    # TODO: scalarOps, aggregation, others   

# TODO: this should inherit from np.array, thus behave like one
class NormalizedTable(np.ndarray):

    def __getNormalizedTable(self, tensorS, tensorK, tensorR):
        #TODO: add validation: inputs should be of tensor-wrapper type
        normalizedTable = polyglot.eval(language="thanos",string="")
        normalizedTable.build(tensorS, tensorK, tensorR)
        return normalizedTable

    # TODO: 
    # 1. Do need __array_priority__ (?)
    # 2. In MorpheusPy, why does `__getitem__` cause significant performance penalties?
    # 3. Why inherit from matrix and yet call numeric as the super?
    # 
    def __new__(cls, S, K, R, is_transposed=False):
        obj = N.ndarray.__new__(cls, None)
        tensorS = TensorFromNumpy(S)
        tensorK = TensorFromNumpy(K) 
        tensorR = TensorFromNumpy(R)
        obj.is_transposed = is_transposed
        obj.normalizedTable = obj.__getNormalizedTable(tensorS, tensorK, tensorR)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        tensorS = TensorFromNumpy(getattr(obj, 'S', None))
        tensorK = TensorFromNumpy(getattr(obj, 'K', None))
        tensorR = TensorFromNumpy(getattr(obj, 'R', None))
        obj.is_tranposed = getattr(obj, 'is_transposed', None)
        obj.normalizedTable = obj.__getNormalizedTable(tensorS, tensorK, tensorR)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        #TODO: provide more information
        return "NormalizedMatrix"

    def __getitem__(self, index):
        # TODO: provide better error
        return NotImplementedError



    def transpose(self):
        return NormalizedMatrix(self.ent_table, self.att_table, self.kfkds,dtype=self.dtype, trans=(not self.trans), stamp=self.stamp)

    @property
    def T(self):
        return self.transpose()
