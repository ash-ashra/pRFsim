#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 17:18:12 2018

@author: arash
"""
import prfsim.sim as psim
import numpy as np

# experiment parameters
radius = 10
precision = 0.1
barWidth = radius / 4
angles = [-90, 45, -180, 315, 90, 225, 0, 135]
nFrames = len(angles)*3
TR = 3.0
TRs = 5 # number of TRs for each frame
duration = nFrames*TRs

x, y = np.mgrid[-radius:radius:precision,
                -radius:radius:precision]
pos = np.dstack((x, y))
length = len(x[0])
nVoxels = 6

# parameters for double gamma distribution function hrf:
n1 = 4
lmbd1 = 2.0
t01 = 0
n2 = 7
lmbd2 = 3
t02 = 0
a = 0.3

t = np.arange(0,nFrames*TRs*TR,TR)
hrf_gen = psim.hrf_double_gamma(t, n1, n2, lmbd1, lmbd2, t01, t02, a)
hrf_est = hrf_gen

stim = psim.generateStim(radius=radius, precision=precision,
                    barWidth=barWidth, angles=angles,
                    nFrames=nFrames, length=length,
		            TR=TR, TRs=TRs)

print('stimulus generated')

neuronal_responses = psim.getNeuronalResponse(stim=stim, nVoxels=nVoxels,
                                        radius=radius, precision=precision,
                                        duration=duration)

print('Neuronal responses generated')

bolds = psim.generateData(neuronal_responses=neuronal_responses,
                     hrf=hrf_gen,
                     duration=duration, nVoxels=nVoxels)

print('BOLD responses generated')


print('pRF estimations started...')
results = psim.estimateAll(bolds=bolds, stim=stim,
                      hrf=hrf_est, radius=radius,
                      precision=precision,
                      nVoxels=nVoxels, margin = 1)
print('pRF estimation errors generated')
