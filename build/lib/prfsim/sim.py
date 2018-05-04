#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 21:07:09 2018

@author: arash
"""

from matplotlib import pyplot as plt; figSize = 5
import numpy as np
import pandas as pd

from scipy.stats import gamma
from scipy.misc import factorial
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import least_squares

import seaborn as sns; sns.set()
import time

    
    
    
def generateStim(radius, precision, barWidth, angles, nFrames, length, TR, TRs):
    ## Creating bar stimulus 3D array
    stim = np.zeros((nFrames*TRs, length, length))
    X = Y = np.arange(-radius, radius , precision)
    f = 0
    for angle in angles:
        for k in range(3):
            for i, x in enumerate(X): # do with multithreading
                for j, y in enumerate(Y):
                    if np.sin(np.deg2rad(angle))*x - np.cos(np.deg2rad(angle))*y <= (0.5-0.5*k)*radius + barWidth:
                        if np.sin(np.deg2rad(angle))*x - np.cos(np.deg2rad(angle))*y >= (0.5-0.5*k)*radius - barWidth:
                            for TR in range(TRs):
                                stim[f*TRs + TR, j, i] = 1
            # updates the frame
            f = f + 1
            
    return stim
  
  
def pRF_size(x, y):
  return 0.5*np.log(np.e+np.sqrt(x**2+y**2))

def pRF_model2(x, y, sigma, radius, precision):
  mean = np.array([x, y])
  cov = np.diag([sigma, sigma])
  rf = multivariate_normal(mean=mean, cov=cov)
  x, y = np.mgrid[-radius:radius:precision, -radius:radius:precision]
  pos = np.dstack((x, y))
  model = rf.pdf(pos)
  return model

def apply_pRF2(x, y, stim, radius, precision): 
  model = pRF_model2(x, y, pRF_size(x, y), radius, precision)
  return np.sum(np.sum(stim*model, axis=1),axis=1)


def getNeuronalResponse(stim, nVoxels, radius, precision, duration): 
  pRF_responses = np.zeros((duration, nVoxels, nVoxels))
  X_pRF = Y_pRF = np.linspace(-radius, radius , nVoxels)

  for i, x in enumerate(X_pRF): # do with multithreading
    for j, y in enumerate(Y_pRF):
      pRF_responses[:, i, j] = apply_pRF2(x, y, stim, radius, precision)
      
  return pRF_responses


def hrf_single_gamma(t,n,lmbd,t0):
    return gamma.pdf(t,n,loc=t0,scale=lmbd)


def hrf_double_gamma(t,n1,n2,lmbd1,lmbd2,t01,t02,a):
    c = (gamma.cdf(t[t.size-1],n1,loc=t01,scale=lmbd1) 
        - a * gamma.cdf(t[t.size-1],n2,loc=t02,scale=lmbd2))
            
    return ( 1/c * (gamma.pdf(t,n1,scale=lmbd1,loc=t01) 
                   - a * gamma.pdf(t,n2,scale=lmbd2,loc=t02)) )
  
def hrf_friston(t, beta1, beta2, beta3):
    return (beta1 * 1.0/factorial(3) * np.power(t,3) * np.exp(-t) 
    + beta2 * 1.0/factorial(7) * np.power(t,7) * np.exp(-t) 
    + beta3 * 1.0/factorial(15) * np.power(t,15) * np.exp(-t) )
  
  
def generateData(neuronal_responses, hrf, duration, nVoxels):
  ## Hemodynamic Responses
  bolds = np.zeros((duration, nVoxels, nVoxels))
  for i in range(nVoxels): # do with multithreading
      for j in range(nVoxels):
          n = neuronal_responses[:, i, j]
          bolds[:, i, j] = np.convolve(hrf, n)[0:duration] + norm.rvs(scale=0.1, size=duration)
  return bolds


def estimatePRF(bold, stim, hrf, radius, precision):
  def MSE2(x, bold, hrf, radius, precision):
    model = pRF_model2(x[0], x[1], x[2], radius, precision)

    response = stim*model
    rf_response = np.sum(np.sum(response, axis=1),axis=1)
    prediction = np.convolve(rf_response, hrf)[:bold.shape[0]]
    prediction = prediction / np.max(prediction)
    bold = bold / np.max(bold)
    return np.sum((bold - prediction)**2)

  bnds = ((-radius, radius), (-radius, radius), (0.1, 5))
  try_count = 0
  while True:
    try_count += 1
    x0 = (np.random.rand(3)-0.5)*20
    res = minimize(MSE2, x0, (bold, hrf, radius, precision), bounds=bnds)
    if res.success and res.fun < 0.1 or try_count > 10:
      break

  x_min = res.x[0]
  y_min = res.x[1]
  s_min = res.x[2]

  return x_min, y_min, s_min
  
  
  
def estimateAll(bolds, stim, hrf, radius, precision, nVoxels, margin = 1):
  Xs = [];Ys = []; Ss = []; Xs_est = []; Ys_est = []; Ss_est = []
  Xs_err = []; Ys_err = []; Ss_err = []
  Xs_err_im = np.zeros((nVoxels, nVoxels))
  Ys_err_im = np.zeros((nVoxels, nVoxels))
  Ss_err_im = np.zeros((nVoxels, nVoxels))

  X_pRF = Y_pRF = np.linspace(-radius, radius , nVoxels)
  
  for i, x in enumerate(X_pRF): # do with multithreading
    for j, y in enumerate(Y_pRF):
      bold = bolds[:, i, j]
      x_est, y_est, s_est = estimatePRF(bold, stim, hrf, radius, precision)
      s = pRF_size(x, y)
      Xs.append(x)
      Ys.append(y)
      Ss.append(s)
      Xs_est.append(x_est)
      Ys_est.append(y_est)
      Ss_est.append(s_est)
      x_err = abs((x_est - x)/x)*100
      y_err = abs((y_est - y)/y)*100
      s_err = abs((s_est - s)/s)*100
      Xs_err.append(x_err)
      Ys_err.append(y_err)
      Ss_err.append(s_err)
      Xs_err_im[i, j] = x_err
      Ys_err_im[i, j] = y_err
      Ss_err_im[i, j] = s_err

      
      
  results = pd.DataFrame()
  results['x_tru'] = pd.Series(Xs)
  results['x_est'] = pd.Series(Xs_est)


  results['y_tru'] = pd.Series(Ys)
  results['y_est'] = pd.Series(Ys_est)


  results['s_tru'] = pd.Series(Ss)
  results['s_est'] = pd.Series(Ss_est)

  results['x_err%'] = pd.Series(Xs_err)
  results['y_err%'] = pd.Series(Ys_err)
  results['s_err%'] = pd.Series(Ss_err)
  
  

  if margin > 0:
    img = Xs_err_im[margin:-margin,margin:-margin]
  else:
    img = Xs_err_im
  fig = plt.figure(figsize=(figSize, figSize))
  plt.grid(None)
  plt.title('$x$ estimation error \% of each voxel')
  ax = sns.heatmap(img, square=True, cmap="YlGnBu")
  plt.show()
  
  if margin > 0:
    img = Ys_err_im[margin:-margin,margin:-margin]
  else:
    img = Ys_err_im
  fig = plt.figure(figsize=(figSize, figSize))
  plt.grid(None) 
  plt.title('$y$ estimation error \% of each voxel')
  ax = sns.heatmap(img, square=True, cmap="YlGnBu")
  plt.show()
  
  if margin > 0:
    img = Ss_err_im[margin:-margin,margin:-margin]
  else:
    img = Ss_err_im
  fig = plt.figure(figsize=(figSize, figSize))
  plt.grid(None) 
  plt.title('$\sigma$ estimation error \% of each voxel')
  ax = sns.heatmap(img, square=True, cmap="YlGnBu")
  plt.show()
  return results


