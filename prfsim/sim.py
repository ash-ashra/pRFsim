#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 21:07:09 2018

@author: arash
"""
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize
from scipy.stats import gamma
from scipy.misc import factorial
from scipy.stats import norm
# from scipy.stats import multivariate_normal

from scipy.optimize import least_squares

import seaborn as sns
# from pylab import savefig
sns.set()
figSize = 5


def init(radius_tmp, barWidth_tmp, precision_tmp, TR_tmp,
         TRs_tmp, sqrtVoxels_tmp, angles, t_tmp, title,
         makeDiscontinous=False):
    """Short summary.

    Parameters
    ----------
    radius_tmp : type
        Description of parameter `radius_tmp`.
    barWidth_tmp : type
        Description of parameter `barWidth_tmp`.
    precision_tmp : type
        Description of parameter `precision_tmp`.
    TR_tmp : type
        Description of parameter `TR_tmp`.
    TRs_tmp : type
        Description of parameter `TRs_tmp`.
    sqrtVoxels_tmp : type
        Description of parameter `sqrtVoxels_tmp`.
    angles : type
        Description of parameter `angles`.
    t_tmp : type
        Description of parameter `t_tmp`.
    title : type
        Description of parameter `title`.
    makeDiscontinous : type
        Description of parameter `makeDiscontinous`.

    Returns
    -------
    type
        Description of returned object.

    """
    global radius
    radius = radius_tmp
    global precision
    precision = precision_tmp
    global TR
    TR = TR_tmp
    global TRs
    TRs = TRs_tmp
    global sqrtVoxels
    sqrtVoxels = sqrtVoxels_tmp
    global t
    t = t_tmp

    global nFrames
    nFrames = len(angles) * 3

    global duration
    duration = nFrames * TRs

    global barWidth
    barWidth = barWidth_tmp

    X = Y = np.arange(-radius, radius, precision)

    global x_grid
    global y_grid
    x_grid, y_grid = np.mgrid[-radius:radius:precision,
                              -radius:radius:precision]

    global length
    length = len(X)

    # if isCheckerboard:
    #     # create the checkerboard pattern
    #     arr = np.zeros((16, 16))
    #     arr[::2, ::2] = 1
    #     arr[1::2, 1::2] = 1
    #     checkerboard = imresize(arr, np.array((length, length)),
    #                             interp='nearest')

    # create three vertical bars
    img_temps = np.zeros((3, length, length))
    for k in range(3):
        for i, x in enumerate(X):  # do with multithreading
            for j, y in enumerate(Y):
                if x <= (-0.5 * k + 0.5) * radius + barWidth:
                    if x >= (-0.5 * k + 0.5) * radius - barWidth:
                        # if isCheckerboard:
                        #     img_temps[k, i, j] = checkerboard[i, j]
                        # else:
                        # if makeDiscontinous:
                        #     index = (2*k) % 3
                        #     img_temps[index, i, j] = 1
                        # else:
                        img_temps[k, i, j] = 1

    # rotate them
    stim_compact = np.zeros((nFrames, length, length))
    f = 0
    for angle in angles:
        for k in range(3):
            rot = rotate(img_temps[k], angle=angle,
                         mode='nearest', reshape=False)

            if makeDiscontinous:
                index = (5*f) % nFrames
            else:
                index = f
            stim_compact[index, :, :] = (rot > np.max(rot) / 2) * 1.0
            f += 1

    for f in range(nFrames):
        plt.figure(figsize=(figSize, figSize))
        plt.grid(None)
        plt.axis('off')
        plt.imshow(stim_compact[f, :, :],
                   cmap="Blues")
        plt.savefig('/home/arash/results/stim_%s_%d.png' % (title, f))
        plt.close()

    # delay them
    duration = nFrames * TRs
    global stim
    stim = np.zeros((duration, length, length))
    for f in range(nFrames):
        for TR in range(TRs):
            stim[f * TRs + TR, :, :] = stim_compact[f, :, :]

    shape2D = (len(X), len(Y))
    proj2D = np.zeros(shape2D)
    for i in range(nFrames):
        proj2D = proj2D + (stim_compact[i, :, :]).reshape(shape2D)
    plt.figure(figsize=(figSize, figSize))
    plt.grid(None)
    plt.title('stimulus projection across time')
    plt.imshow(proj2D)
    plt.savefig('/home/arash/results/stimProj_%s.eps' % title)
    plt.close()
    return stim


def pRF_size(x, y):
    """Short summary.

    Parameters
    ----------
    x : type
        Description of parameter `x`.
    y : type
        Description of parameter `y`.

    Returns
    -------
    type
        Description of returned object.

    """
    return 0.5 * np.log(np.e + np.sqrt(x**2 + y**2))


def pRF_model(x0, y0, sigma):
    """Short summary.

    Parameters
    ----------
    x0 : type
        Description of parameter `x0`.
    y0 : type
        Description of parameter `y0`.
    sigma : type
        Description of parameter `sigma`.

    Returns
    -------
    type
        Description of returned object.

    """
    global x_grid
    global y_grid
    # mean = np.array([x0, y0])
    # cov = np.diag([sigma, sigma])
    # rf = multivariate_normal(mean=mean, cov=cov)
    # pos = np.dstack((x, y))
    # model = rf.pdf(pos)
    model = np.exp((-(x_grid - x0)**2 - (y_grid - y0)**2) / (2 * sigma**2))
    return model


def pRF_response(x, y, sigma, exponent=1.0):
    """Short summary.

    Parameters
    ----------
    x : type
        Description of parameter `x`.
    y : type
        Description of parameter `y`.

    Returns
    -------
    type
        Description of returned object.

    """
    global stim
    model = pRF_model(x, y, sigma)
    return np.power(np.sum(np.sum(stim * model, axis=1), axis=1), exponent)


def getNeuronalResponse(exponent):
    """Short summary.

    Returns
    -------
    type
        Description of returned object.

    """
    global sqrtVoxels
    global radius
    global precision
    global duration
    global stim

    pRF_responses = np.zeros((duration, sqrtVoxels, sqrtVoxels))
    X_pRF = Y_pRF = np.linspace(-radius, radius, sqrtVoxels)

    for i, x in enumerate(X_pRF):  # do with multithreading
        for j, y in enumerate(Y_pRF):
            sigma = pRF_size(x, y)
            pRF_responses[:, i, j] = pRF_response(x, y, sigma, exponent)

    return pRF_responses


def hrf_single_gamma(t, n, lmbd, t0):
    """Short summary.

    Parameters
    ----------
    t : type
        Description of parameter `t`.
    n : type
        Description of parameter `n`.
    lmbd : type
        Description of parameter `lmbd`.
    t0 : type
        Description of parameter `t0`.

    Returns
    -------
    type
        Description of returned object.

    """
    return gamma.pdf(t, n, loc=t0, scale=lmbd)


def hrf_double_gamma(t, n1, n2, lmbd1, lmbd2, t01, t02, a):
    """Short summary.

    Parameters
    ----------
    t : type
        Description of parameter `t`.
    n1 : type
        Description of parameter `n1`.
    n2 : type
        Description of parameter `n2`.
    lmbd1 : type
        Description of parameter `lmbd1`.
    lmbd2 : type
        Description of parameter `lmbd2`.
    t01 : type
        Description of parameter `t01`.
    t02 : type
        Description of parameter `t02`.
    a : type
        Description of parameter `a`.

    Returns
    -------
    type
        Description of returned object.

    """
    c = (gamma.cdf(t[t.size - 1], n1, loc=t01, scale=lmbd1)
         - a * gamma.cdf(t[t.size - 1], n2, loc=t02, scale=lmbd2))

    return (1 / c * (gamma.pdf(t, n1, scale=lmbd1, loc=t01)
                     - a * gamma.pdf(t, n2, scale=lmbd2, loc=t02)))


def hrf_friston(t, beta1, beta2, beta3):
    """Short summary.

    Parameters
    ----------
    t : type
        Description of parameter `t`.
    beta1 : type
        Description of parameter `beta1`.
    beta2 : type
        Description of parameter `beta2`.
    beta3 : type
        Description of parameter `beta3`.

    Returns
    -------
    type
        Description of returned object.

    """
    return (beta1 * 1.0 / factorial(3) * np.power(t, 3) * np.exp(-t)
            + beta2 * 1.0 / factorial(7) * np.power(t, 7) * np.exp(-t)
            + beta3 * 1.0 / factorial(15) * np.power(t, 15) * np.exp(-t))


def bold_response_nonlinear(n, t,
                            beta1, beta2, beta3,
                            beta11, beta12, beta13,
                            beta21, beta22, beta23,
                            beta31, beta32, beta33):
    """Short summary.

    Parameters
    ----------
    n : type
        Description of parameter `n`.
    t : type
        Description of parameter `t`.
    beta1 : type
        Description of parameter `beta1`.
    beta2 : type
        Description of parameter `beta2`.
    beta3 : type
        Description of parameter `beta3`.
    beta11 : type
        Description of parameter `beta11`.
    beta12 : type
        Description of parameter `beta12`.
    beta13 : type
        Description of parameter `beta13`.
    beta21 : type
        Description of parameter `beta21`.
    beta22 : type
        Description of parameter `beta22`.
    beta23 : type
        Description of parameter `beta23`.
    beta31 : type
        Description of parameter `beta31`.
    beta32 : type
        Description of parameter `beta32`.
    beta33 : type
        Description of parameter `beta33`.

    Returns
    -------
    type
        Description of returned object.

    """
    b1 = 1.0 / factorial(3) * np.power(t, 3) * np.exp(-t)
    b2 = 1.0 / factorial(7) * np.power(t, 7) * np.exp(-t)
    b3 = 1.0 / factorial(15) * np.power(t, 15) * np.exp(-t)

    x1 = np.convolve(n, b1)[0:len(t)]
    x2 = np.convolve(n, b2)[0:len(t)]
    x3 = np.convolve(n, b3)[0:len(t)]

    y11 = np.multiply(x1, x1)
    y12 = np.multiply(x1, x2)
    y13 = np.multiply(x1, x3)
    y21 = np.multiply(x2, x1)
    y22 = np.multiply(x2, x2)
    y23 = np.multiply(x3, x3)
    y31 = np.multiply(x3, x1)
    y32 = np.multiply(x3, x2)
    y33 = np.multiply(x3, x3)

    return (beta1 * x1 + beta2 * x2 + beta3 * x3
            + beta11 * y11 + beta12 * y12 + beta13 * y13
            + beta21 * y21 + beta22 * y22 + beta23 * y23
            + beta31 * y31 + beta32 * y32 + beta33 * y33)


def RSS_NHRF(x, bold, n):
    """Short summary.

    Parameters
    ----------
    x : type
        Description of parameter `x`.
    bold : type
        Description of parameter `bold`.
    n : type
        Description of parameter `n`.

    Returns
    -------
    type
        Description of returned object.

    """
    prediction = bold_response_nonlinear(n=n, t=t,
                                         beta1=x[0], beta2=x[1], beta3=x[2],
                                         beta11=x[3], beta12=x[4], beta13=x[5],
                                         beta21=x[6], beta22=x[7], beta23=x[8],
                                         beta31=x[9], beta32=x[10],
                                         beta33=x[11])
    return np.sum((bold - prediction)**2)


def findNonLinearHRF(neuronal_responses, hrf, noise):
    """Short summary.

    Parameters
    ----------
    neuronal_responses : type
        Description of parameter `neuronal_responses`.
    hrf : type
        Description of parameter `hrf`.
    noise : type
        Description of parameter `noise`.

    Returns
    -------
    type
        Description of returned object.

    """
    # count = 0
    # prev_cost = 1e30
    bolds = np.zeros((duration, sqrtVoxels, sqrtVoxels))
    for i in range(sqrtVoxels):
        for j in range(sqrtVoxels):
            # count += 1
            n = neuronal_responses[:, i, j]
            bolds[:, i, j] = np.convolve(
                hrf, n)[0:duration] + norm.rvs(scale=noise, size=duration)
            # x0 = 10 * (np.random.rand(12) + 1e-10)
            # res = least_squares(RSS_NHRF, x0, args=(bolds[:, i, j], n))
            # if res.cost < prev_cost:
            #     pars = res.x
            #     prev_cost = res.cost
            #     print(count)
    x0 = 10 * (np.random.rand(12) + 1e-10)
    res = least_squares(RSS_NHRF, x0, args=(np.mean(np.mean(bolds)),
                        np.mean(np.mean(neuronal_responses))))
    pars = res.x
    return pars


def RSS_HRF(x, bold, n):
    """Short summary.

    Parameters
    ----------
    x : type
        Description of parameter `x`.
    bold : type
        Description of parameter `bold`.
    n : type
        Description of parameter `n`.

    Returns
    -------
    type
        Description of returned object.

    """
    hrf = hrf_double_gamma(t, x[0], x[1], x[2], x[3], x[4], x[5], x[6])
    prediction = np.convolve(hrf, n)[0:duration]
    return np.sum((bold - prediction)**2)


def findLinearHRF(neuronal_responses, n_hrf_pars, noise):
    """Short summary.

    Parameters
    ----------
    neuronal_responses : type
        Description of parameter `neuronal_responses`.
    n_hrf_pars : type
        Description of parameter `n_hrf_pars`.
    noise : type
        Description of parameter `noise`.

    Returns
    -------
    type
        Description of returned object.

    """
    # count = 0
    # prev_cost = 1e30
    x = n_hrf_pars
    bolds = np.zeros((duration, sqrtVoxels, sqrtVoxels))
    for i in range(sqrtVoxels):
        for j in range(sqrtVoxels):
            # count += 1
            n = neuronal_responses[:, i, j]
            bolds[:, i, j] = bold_response_nonlinear(n=n, t=t,
                                                     beta1=x[0],
                                                     beta2=x[1], beta3=x[2],
                                                     beta11=x[3],
                                                     beta12=x[4], beta13=x[5],
                                                     beta21=x[6],
                                                     beta22=x[7], beta23=x[8],
                                                     beta31=x[9],
                                                     beta32=x[10],
                                                     beta33=x[11]) + norm.rvs(
                                                     scale=noise,
                                                     size=duration)
            # x0 = 10 * (np.random.rand(12) + 1e-10)
            # res = least_squares(RSS_NHRF, x0, args=(bolds[:, i, j], n))
            # if res.cost < prev_cost:
            #     pars = res.x
            #     prev_cost = res.cost
            #     print(count)
    x0 = 10 * (np.random.rand(7) + 1e-10)
    res = least_squares(RSS_HRF, x0, args=(np.mean(np.mean(bolds)),
                        np.mean(np.mean(neuronal_responses))))
    pars = res.x
    return pars


def generateData(neuronal_responses, noise, hrf, n_hrf_pars, makeNonLinear):
    """Short summary.

    Parameters
    ----------
    neuronal_responses : type
        Description of parameter `neuronal_responses`.
    noise : type
        Description of parameter `noise`.
    hrf : type
        Description of parameter `hrf`.
    n_hrf_pars : type
        Description of parameter `n_hrf_pars`.
    makeNonLinear : type
        Description of parameter `makeNonLinear`.

    Returns
    -------
    type
        Description of returned object.

    """
    global sqrtVoxels
    global duration
    # Hemodynamic Responses
    bolds = np.zeros((duration, sqrtVoxels, sqrtVoxels))

    if not makeNonLinear:
        for i in range(sqrtVoxels):
            for j in range(sqrtVoxels):
                n = neuronal_responses[:, i, j]
                bolds[:, i, j] = np.convolve(
                    hrf, n)[0:duration] + norm.rvs(scale=noise, size=duration)
        return bolds
    else:
        x = n_hrf_pars
        non_bolds = np.zeros((duration, sqrtVoxels, sqrtVoxels))
        for i in range(sqrtVoxels):
            for j in range(sqrtVoxels):
                n = neuronal_responses[:, i, j]
                non_bolds[:, i, j] = bold_response_nonlinear(n=n, t=t,
                                                             beta1=x[0],
                                                             beta2=x[1],
                                                             beta3=x[2],
                                                             beta11=x[3],
                                                             beta12=x[4],
                                                             beta13=x[5],
                                                             beta21=x[6],
                                                             beta22=x[7],
                                                             beta23=x[8],
                                                             beta31=x[9],
                                                             beta32=x[10],
                                                             beta33=x[11])
                + norm.rvs(scale=noise, size=duration)
        return non_bolds


def pred_pRF_response(x, hrf, exponent):
    """Short summary.

    Parameters
    ----------
    x : type
        Description of parameter `x`.
    hrf : type
        Description of parameter `hrf`.

    Returns
    -------
    type
        Description of returned object.

    """
    global stim
    global duration
    rf_response = pRF_response(x[0], x[1], x[2], exponent)
    pred = x[3] * np.convolve(rf_response, hrf)[:duration]
    return pred


def MSE(x, bold, hrf, exponent):
    """Short summary.

    Parameters
    ----------
    x : type
        Description of parameter `x`.
    bold : type
        Description of parameter `bold`.
    hrf : type
        Description of parameter `hrf`.

    Returns
    -------
    type
        Description of returned object.

    """
    prediction = pred_pRF_response(x, hrf, exponent)
    return (bold - prediction)


def estimatePRF(bold, hrf, exponent):
    """Short summary.

    Parameters
    ----------
    bold : type
        Description of parameter `bold`.
    hrf : type
        Description of parameter `hrf`.

    Returns
    -------
    type
        Description of returned object.

    """
    global t
    global radius
    global precision
    global stim

    x0 = [0, 0, pRF_size(radius, radius), 1]
    res = least_squares(MSE, args=(bold, hrf, exponent),
                        x0=x0, bounds=(-radius, radius))

    # plt.plot(t, bold, t, pred_pRF_response(res.x, hrf), t, hrf)
    # plt.legend(['bold', 'model', 'hrf'])
    # plt.show()
    x_min = res.x[0]
    y_min = res.x[1]
    s_min = res.x[2]

    return x_min, y_min, s_min


def estimateAll(bolds, exponent, hrf, n_hrf_pars,
                title, assumeLinear=False, margin=0):
    """Short summary.

    Parameters
    ----------
    bolds : type
        Description of parameter `bolds`.
    exponent : type
        Description of parameter `exponent`.
    hrf : type
        Description of parameter `hrf`.
    n_hrf_pars : type
        Description of parameter `n_hrf_pars`.
    assumeNonLinear : type
        Description of parameter `assumeNonLinear`.
    title : type
        Description of parameter `title`.
    margin : type
        Description of parameter `margin`.

    Returns
    -------
    type
        Description of returned object.

    """
    global radius
    global precision
    global sqrtVoxels
    global stim
    Xs = []
    Ys = []
    Ss = []
    Xs_est = []
    Ys_est = []
    Ss_est = []
    Xs_err = []
    Ys_err = []
    Ss_err = []
    Xs_err_im = np.zeros((sqrtVoxels, sqrtVoxels))
    Ys_err_im = np.zeros((sqrtVoxels, sqrtVoxels))
    Ss_err_im = np.zeros((sqrtVoxels, sqrtVoxels))

    X_pRF = Y_pRF = np.linspace(-radius, radius, sqrtVoxels)

    count = 0
    for i, x in enumerate(X_pRF):  # do with multithreading
        for j, y in enumerate(Y_pRF):
            count += 1
            if count % 10 == 0:
                print(count)
            bold = bolds[:, i, j]
            x_est, y_est, s_est = estimatePRF(bold, hrf, exponent)
            s = pRF_size(x, y)
            Xs.append(x)
            Ys.append(y)
            Ss.append(s)
            Xs_est.append(x_est)
            Ys_est.append(y_est)
            Ss_est.append(s_est)

            if y == 0:
                y = precision
            if x == 0:
                x = precision
            x_err = abs((x_est - x) / x) * 100
            y_err = abs((y_est - y) / y) * 100
            s_err = abs((s_est - s) / s) * 100
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

    Xs_err_mean = Xs_err_im.mean()
    Ys_err_mean = Ys_err_im.mean()
    Ss_err_mean = Ss_err_im.mean()

    strTime = time.strftime("%Y-%m-%d-%H:%M")

    if margin > 0:
        img = Xs_err_im[margin:-margin, margin:-margin]
    else:
        img = Xs_err_im
    plt.figure(1, figsize=(figSize, figSize))
    plt.grid(None)
    plt.title('average $x$ error percentage = %.2f' %
              Xs_err_mean)
    sns.heatmap(img, square=True, cmap="Blues", vmin=0, vmax=100)
    exponentInt = int(10 * exponent)
    plt.savefig('/home/arash/results/%s_%s_x_%d.eps' % (strTime,
                title, exponentInt))
    plt.clf()
    plt.cla()
    plt.close()

    if margin > 0:
        img = Ys_err_im[margin:-margin, margin:-margin]
    else:
        img = Ys_err_im

    plt.figure(2, figsize=(figSize, figSize))
    plt.grid(None)
    plt.title('average $y$ error percentage = %.2f' %
              Ys_err_mean)
    sns.heatmap(img, square=True, cmap="Blues", vmin=0, vmax=100)
    plt.savefig('/home/arash/results/%s_%s_y_%d.eps' % (strTime,
                title, exponentInt))
    plt.clf()
    plt.cla()
    plt.close()

    if margin > 0:
        img = Ss_err_im[margin:-margin, margin:-margin]
    else:
        img = Ss_err_im

    plt.figure(3, figsize=(figSize, figSize))
    plt.title(
        'average $\sigma$ error percentage = %.2f' %
        Ss_err_mean)
    sns.heatmap(img, square=True, cmap="Blues", vmin=0, vmax=100)
    plt.savefig('/home/arash/results/%s_%s_sigma_%d.eps' % (strTime,
                title, exponentInt))
    plt.clf()
    plt.cla()
    plt.close()
    return results
