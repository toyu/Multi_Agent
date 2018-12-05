from joblib import Parallel, delayed
from multiprocessing import Value, Array
from time import time

# -*- coding: utf-8 -*-
from joblib import Parallel, delayed
from multiprocessing import Value, Array

shared_int = Value('i', 1)

def process(n):
    shared_int.value = 3.14
    return sum([i*n for i in range(100000)])

# 繰り返し計算 (並列化)
Parallel(n_jobs=10)( [delayed(process)(i) for i in range(10000)] )

print(shared_int.value)