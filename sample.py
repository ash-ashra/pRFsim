#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 17:18:12 2018

@author: arash
"""
import prfsim.sim as psim
import numpy as np
# from matplotlib import pyplot as plt
# figSize = 5


# experiment parameters
radius = 10.001
precision = 0.1

angles = [-90, 45, -180, 315, 90, 225, 0, 135
          # ,-91, 44, -181, 314, 91, 224, 1, 134
          # ,-88, 46, -179, 316, 89, 226, -1, 136
          # ,-91, 44, -181, 314, 91, 224, 1, 134
          ]

TR = 3.0
TRs = 5  # number of TRs for each frame

noise = 1.0
sqrtVoxels = 30

# parameters for double gamma distribution function hrf:
n1 = 4
lmbd1 = 2.0
t01 = 0
n2 = 7
lmbd2 = 3
t02 = 0
a = 0.3

t = np.arange(0, len(angles)*3*TRs*TR, TR)
hrf_gen = psim.hrf_double_gamma(t, n1, n2, lmbd1, lmbd2, t01, t02, a)
hrf_est = hrf_gen

stim = psim.init(radius, precision, TR, TRs, sqrtVoxels,
                 angles, t, isCheckerboard=False)
print('stimulus generated')

# for n in np.arange(0.5, 1, 0.1):
n = 0.8
neuronal_responses = psim.getNeuronalResponse(n)
print('Neuronal responses generated')

bolds = psim.generateData(neuronal_responses, hrf_gen, noise)
print('BOLD responses generated')

print('pRF estimations started...')
results = psim.estimateAll(bolds, hrf_est, n, margin=0)
print('pRF estimation errors generated')
