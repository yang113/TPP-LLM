import numpy as np

def MSLE(pred, true):
    return np.mean(np.square(np.log2(pred+1) - np.log2(true+1)))


def metric(pred, true):
    msle = MSLE(pred, true)

    return msle
