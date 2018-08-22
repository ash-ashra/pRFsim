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
radius = 10.3
precision = 0.1

angles = [
          -90, 45, -180, 315, 90, 225, 0, 135
          # ,-91, 44, -181, 314, 91, 224, 1, 134
          # -88, 46, -179, 316, 89, 226, -1, 136
          # ,-91, 44, -181, 314, 91, 224, 1, 134
          ]

TR = 3.0
TRs = 5  # number of TRs for each frame

noise = 1.0
sqrtVoxels = 6

# parameters for non-linear frinson hrf:
n_hrf_pars = [0.5, -1.4, 11.3, 0.1, 0.9, 0.2,
              -0.9, 0.9, -5.4, 1.4, -0.4, 1.9]

t = np.arange(0, len(angles)*3*TRs*TR, TR)

for shift in np.arange(0, radius/4, radius/16):
    title = '%.2f' % shift
    stim = psim.init(radius, radius/4+shift, precision, TR, TRs, sqrtVoxels,
                     angles, t, title, makeDiscontinous=False)
    print('stimulus generated')

    # for n in np.arange(0.5, 1, 0.1):
    exponent = 0.8
    neuronal_responses = psim.getNeuronalResponse(exponent)
    print('Neuronal responses generated')

    # n_hrf_pars = psim.findNonLinearHRF(neuronal_responses, hrf, noise)
    # print(n_hrf_pars)
    x = psim.findLinearHRF(neuronal_responses, n_hrf_pars, noise)
    hrf = psim.hrf_double_gamma(t, x[0], x[1], x[2], x[3], x[4], x[5], x[6])
    print('Equivalent linear HRF is found')

    bolds = psim.generateData(neuronal_responses, noise,
                              hrf, n_hrf_pars,
                              makeNonLinear=True)
    print('BOLD responses generated')

    print('pRF estimations started...')
    results = psim.estimateAll(bolds, exponent,
                               hrf, n_hrf_pars, title,
                               assumeLinear=True)
    print('pRF estimation errors generated')
