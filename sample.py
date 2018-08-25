#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 17:18:12 2018

@author: arash
"""
import prfsim.sim as psim
import numpy as np
from matplotlib import pyplot as plt


# experiment parameters
radius = 10  # in cm
precision = 0.2  # in cm

angles = [
          -90, 45, -180, 315, 90, 225, 0, 135
          # ,-91, 44, -181, 314, 91, 224, 1, 134
          # -88, 46, -179, 316, 89, 226, -1, 136
          # ,-91, 44, -181, 314, 91, 224, 1, 134
          ]

TR = 3.0  # in seconds
TRs = 3   # number of TRs for each frame

noise = 1.0  # in cm
exponent = 0.8  # power of the CSS model

# parameters for double gamma distribution function hrf:
n1 = 4
lmbd1 = 2.0
t01 = 0
n2 = 7
lmbd2 = 3
t02 = 0
a = 0.3

# try first time for finding HRFs
sqrtVoxels = 20  # number of voxels in each dimension
title = 'cont_0.00'
t = np.arange(0, len(angles)*3*TRs*TR, TR)
stim = psim.init(radius, radius/4, precision, TR, TRs, sqrtVoxels,
                 angles, t, title, makeDiscontinous=False)
print('stimulus generated')


neuronal_responses = psim.getNeuronalResponse(exponent)
print('Neuronal responses generated')


hrf = psim.hrf_double_gamma(t, n1, n2, lmbd1, lmbd2, t01, t02, a)
n_hrf_pars = psim.findNonLinearHRF(neuronal_responses, hrf)
print(n_hrf_pars)
x = psim.findLinearHRF(neuronal_responses, n_hrf_pars)
print(x)
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


# optimizing the parameters
# bar widths
barWidths = []
errors = []
for barWidth in np.arange(radius/64, radius/4+radius/64, radius/64):
    title = '%.2f' % barWidth
    stim = psim.init(radius, barWidth, precision, TR, TRs, sqrtVoxels,
                     angles, t, title, makeDiscontinous=False)
    print('stimulus generated')

    neuronal_responses = psim.getNeuronalResponse(exponent)
    print('Neuronal responses generated')

    bolds = psim.generateData(neuronal_responses, noise,
                              hrf, n_hrf_pars,
                              makeNonLinear=True)
    print('BOLD responses generated')

    print('pRF estimations started...')
    results = psim.estimateAll(bolds, exponent,
                               hrf, n_hrf_pars, title,
                               assumeLinear=True)
    barWidths.append(barWidth)
    s_mean = np.mean(results['s_err%'])
    x_mean = np.mean(results['x_err%'])
    y_mean = np.mean(results['y_err%'])
    errors.append((s_mean+x_mean+y_mean)/3)
    print('pRF estimation errors generated')

plt.axes(facecolor='w')
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)
ax.grid(None)
ax.patch.set_facecolor('white')
ax.plot(barWidths, errors, color='b', ls='solid')
ax.set_xlabel('Bar width (cm)')

ax.set_ylabel('Average estimiation errors ($\%$)')
plt.savefig('/home/arash/results/barWidths.eps')
barWidthMax = barWidths[np.argmin(errors)]

# rotation of bars
rotations = []
errors = []
for rot in np.arange(0, 45, 5):
    title = 'rot_%d' % rot
    stim = psim.init(radius, barWidthMax, precision, TR, TRs, sqrtVoxels,
                     angles+rot, t, title, makeDiscontinous=False)
    print('stimulus generated')

    neuronal_responses = psim.getNeuronalResponse(exponent)
    print('Neuronal responses generated')

    bolds = psim.generateData(neuronal_responses, noise,
                              hrf, n_hrf_pars,
                              makeNonLinear=True)
    print('BOLD responses generated')

    print('pRF estimations started...')
    results = psim.estimateAll(bolds, exponent,
                               hrf, n_hrf_pars, title,
                               assumeLinear=True)
    rotations.append(rot)
    s_mean = np.mean(results['s_err%'])
    x_mean = np.mean(results['x_err%'])
    y_mean = np.mean(results['y_err%'])
    errors.append((s_mean+x_mean+y_mean)/3)
    print('pRF estimation errors generated')


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)
ax.grid(None)
ax.patch.set_facecolor('white')

ax.plot(rotations, errors, color='b', ls='solid')
ax.set_xlabel('Rotation (deg)')
ax.set_ylabel('Average estimiation errors ($\%$)')
plt.savefig('/home/arash/results/Rotation.eps')
rotMax = rotations[np.argmin(errors)]

# Now plot the optimal
# try first time for finding HRFs
title = 'optimal_%.2f_%d' % (barWidthMax, rotMax)
t = np.arange(0, len(angles)*3*TRs*TR, TR)
stim = psim.init(radius, barWidthMax, precision, TR, TRs, sqrtVoxels,
                 angles+rotMax, t, title, makeDiscontinous=False)
print('stimulus generated')


neuronal_responses = psim.getNeuronalResponse(exponent)
print('Neuronal responses generated')

bolds = psim.generateData(neuronal_responses, noise,
                          hrf, n_hrf_pars,
                          makeNonLinear=True)
print('BOLD responses generated')


print('pRF estimations started...')
results = psim.estimateAll(bolds, exponent,
                           hrf, n_hrf_pars, title,
                           assumeLinear=True)
print('pRF estimation errors generated')
