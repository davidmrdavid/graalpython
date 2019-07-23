#TODO: PEP8 me!

import numpy as np
from numpy.matrixlib.defmatrix import matrix
import numpy.core.numeric as N

def getNormalizedTable(S, K, R):
    return None

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

    def crossProduct(self):
        pass
        #return numpyModule.matmul(otherArr, self.numpyArr)

    # TODO: Do we want this?
    def transpose(self):
        pass

    # TODO: scalarOps, aggregation, others   

# TODO: this should inherit from np.array, thus behave like one
class NormalizedTable(np.ndarray):

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
        obj.normalizedTable = getNormalizedTable(tensorS, tensorK, tensorR)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        tensorS = TensorFromNumpy(getattr(obj, 'S', None))
        tensorK = TensorFromNumpy(getattr(obj, 'K', None))
        tensorR = TensorFromNumpy(getattr(obj, 'R', None))
        obj.is_tranposed = getattr(obj, 'is_transposed', None)
        obj.normalizedTable = getNormalizedTable(tensorS, tensorK, tensorR)

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

    
    
    
arr = np.matrix([[1,2]])
#print(dir(tensor))
print(dir(NormalizedTable(arr,arr,arr)))
a = NormalizedTable(arr,arr,arr)
print(a.T)
#print(NormalizedTable(arr, arr, arr))
#print(type(NormalizedTable(arr, arr, arr))i)
