import requests
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation
from sklearn import linear_model
from astropy.io.votable import parse_single_table
import tempfile

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
sharepath = '/home/apps/astro/home/jmartinez'

deg2rad = 0.0174532925
rad2deg = 57.2957795


def panstarrs_query(ra_deg, dec_deg, rad_deg, mindet=1, maxsources=10000,
                    server=('https://archive.stsci.edu/panstarrs/search.php')):
    """
    Query Pan-STARRS DR1 @ MAST
    parameters: ra_deg, dec_deg, rad_deg: RA, Dec, field
                                          radius in degrees
                mindet: minimum number of detection (optional)
                maxsources: maximum number of sources
                server: servername
    returns: astropy.table object
    """
    print '\tRA center : %f' % (ra_deg)
    print '\tDEC center: %f' % (dec_deg)
    print '\tradius    : %f' % (rad_deg)
    r = requests.get(server, params={'RA': ra_deg, 'DEC': dec_deg,
                                     'SR': rad_deg, 'max_records': maxsources,
                                     'outputformat': 'VOTable',
                                     'ndetections': ('>%d' % mindet)})

    # write query data into local file
    file_out = '%s/PS1/PS1_temp_%.6f_%.6f.xml' % (jorgepath, ra_deg, dec_deg)
    outf = open(file_out, 'w')
    outf.write(r.text)
    outf.close()

    # parse local file into astropy.table object
    data = parse_single_table(file_out)
    # remove temp file
    os.remove(file_out)
    return data.to_table(use_names_over_ids=True)


def reg_RANSAC(X, y, thresh=0., main_path='',
               pl_name='', plots=False, log=False):
    # Robustly fit linear model with RANSAC algorithm
    X = X.reshape((len(X), 1))
    model_ransac = linear_model.RANSACRegressor(
        linear_model.LinearRegression())
    if thresh != 0:
        threshhold = median_absolute_deviation(y) * thresh
        model_ransac.set_params(residual_threshold=threshhold)
    model_ransac.fit(X, y)
    inlier_mask = model_ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.linspace(np.min(X), np.max(X), 100)
    line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])
    # Ploting in and outliers + linear regression
    if plots:
        fig, ax = plt.subplots(1)
        fig.suptitle(pl_name, fontsize=15)
        ax.scatter(X[inlier_mask], y[inlier_mask], marker='.', color='b')
        ax.scatter(X[outlier_mask], y[outlier_mask], marker='.', color='r')
        ax.plot(line_X, line_y_ransac, color='c')
        if log:
            ax.set_xlabel('Ref (log)')
            ax.set_ylabel('Single epoch (log)')
        else:
            ax.set_xlabel('Ref')
            ax.set_ylabel('Single epoch')
        ax.grid(which='major', axis='x', linewidth=0.5, linestyle='-',
                color='0.8')
        ax.grid(which='major', axis='y', linewidth=0.5, linestyle='-',
                color='0.8')
        plt.savefig('%s/%s_ransac.png' % (main_path, pl_name), dpi=300)
        plt.clf()
        plt.close()
    # Return the coef of the linear regression
    return model_ransac.estimator_.coef_[0][0], \
        model_ransac.estimator_.intercept_[0], line_X, \
        line_y_ransac, outlier_mask


def linear_reg_RANSAC(X, y, X_err=None, y_err=None, thresh=0.,
                      main_path='', pl_name='', plots=False, log=False):
    # Robustly fit linear model with RANSAC algorithm
    X_old = X
    X = X.reshape((len(X), 1))
    model_ransac = linear_model.RANSACRegressor(
        linear_model.LinearRegression())
    if thresh != 0:
        threshhold = median_absolute_deviation(y) * thresh
        model_ransac.set_params(residual_threshold=threshhold)
    model_ransac.fit(X, y)
    inlier_mask = model_ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    # Predict data of estimated models
    line_X = np.linspace(np.min(X), np.max(X), 100)
    line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])
    y_predicted = model_ransac.predict(X)
    slope = model_ransac.estimator_.coef_[0][0]
    inter = model_ransac.estimator_.intercept_[0]
    score = np.sqrt(((y[inlier_mask] - y_predicted[inlier_mask])**2).sum() /
                    (np.std(X[inlier_mask]) * len(X[inlier_mask]) *
                     (len(X[inlier_mask]) - 2)))
    if X_err is not None and y_err is not None:
        slope_err = (score *np.dot(X_err[inlier_mask], y_err[inlier_mask]) /
                     np.dot(X_old[inlier_mask], X_old[inlier_mask]))**2
    else:
        slope_err = score
    # Ploting in and outliers + linear regression
    if plots:
        fig, ax = plt.subplots(1)
        fig.suptitle(pl_name, fontsize=15)
        ax.scatter(X[inlier_mask], y[inlier_mask], marker='.', c='b',
                   edgecolors='None')
        ax.scatter(X[outlier_mask], y[outlier_mask], marker='.', c='r',
                   edgecolors='None')
        ax.loglog(line_X, line_y_ransac, color='c', alpha=.7,
                  label='slope = %.5f\nerror = %.5f' % (slope,  slope_err))
        ax.set_xlabel('Ref')
        ax.set_ylabel('Single epoch')
        ax.legend(loc='best', fontsize='x-small')
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.grid(which='major', axis='x', linewidth=0.5, linestyle='-',
                color='0.8')
        ax.grid(which='major', axis='y', linewidth=0.5, linestyle='-',
                color='0.8')
        plt.savefig('%s/%s_ransac.png' % (main_path, pl_name), dpi=300)
        plt.clf()
        plt.close()
    # Return the coef of the linear regression
    return slope, inter, line_X, line_y_ransac, outlier_mask, slope_err


# Invers
def applyinversetransformation(order, x1t, y1t, sol):

    x0 = np.array([x1t, y1t])
    err = 1.
    while err > 0.1:
        x1ti, y1ti = applytransformation(order, x0[0], x0[1], sol)
        dx1, dx2 = np.linalg.solve(transformation_J(order, x0[0], x0[1], sol),
                                   np.array([x1t - x1ti,
                                             y1t - y1ti]).transpose())
        x0 = x0 + np.array([dx1, dx2])
        err = np.sqrt(dx1**2 + dx2**2)

    return x0[0], x0[1]

# T
def applytransformation(order, x1, y1, sol):

    x1t = sol[0] + sol[2] * x1 + sol[3] * y1
    y1t = sol[1] + sol[4] * x1 + sol[5] * y1
    if order > 1:
        x1t = x1t + sol[6] * x1 * x1 + sol[7] * x1 * y1 + sol[8] * y1 * y1
        y1t = y1t + sol[9] * x1 * x1 + sol[10] * x1 * y1 + sol[11] * y1 * y1
    if order > 2:
        x1t = x1t + sol[12] * x1 * x1 * x1 + sol[13] * x1 * x1 * \
            y1 + sol[14] * x1 * y1 * y1 + sol[15] * y1 * y1 * y1
        y1t = y1t + sol[16] * x1 * x1 * x1 + sol[17] * x1 * x1 * \
            y1 + sol[18] * x1 * y1 * y1 + sol[19] * y1 * y1 * y1

    return x1t, y1t

##########################################################################


def transformation_J(order, x1, y1, sol):

    x1tx = sol[2]
    y1tx = sol[4]
    if order > 1:
        x1tx = x1tx + 2. * sol[6] * x1 + sol[7] * y1
        y1tx = y1tx + 2. * sol[9] * x1 + sol[10] * y1
    if order > 2:
        x1tx = x1tx + 3. * sol[12] * x1 * x1 + 2. * \
            sol[13] * x1 * y1 + sol[14] * y1 * y1
        y1tx = y1tx + 3. * sol[16] * x1 * x1 + 2. * \
            sol[17] * x1 * y1 + sol[18] * y1 * y1

    x1ty = sol[3]
    y1ty = sol[5]
    if order > 1:
        x1ty = x1ty + sol[7] * x1 + 2. * sol[8] * y1
        y1ty = y1ty + sol[10] * x1 + 2. * sol[11] * y1
    if order > 2:
        x1ty = x1ty + sol[13] * x1 * x1 + 2. * \
            sol[14] * x1 * y1 + 3. * sol[15] * y1 * y1
        y1ty = y1ty + sol[17] * x1 * x1 + 2. * \
            sol[18] * x1 * y1 + 3. * sol[19] * y1 * y1

    return np.array([[x1tx, x1ty], [y1tx, y1ty]])


##########################################################################

def RADEC(i, j, CD11, CD12, CD21, CD22, CRPIX1, CRPIX2, CRVAL1, CRVAL2, PV):

    x = CD11 * (i - CRPIX1) + CD12 * (j - CRPIX2)  # deg
    y = CD21 * (i - CRPIX1) + CD22 * (j - CRPIX2)  # deg

    # if no PV, use linear transformation
    if PV is None:
        print "\n\nWARNING: No PV terms found\n\n"
        return ((CRVAL1 + x) / 15., CRVAL2 + y)

    # x, y to xi, eta
    (xi, eta) = xieta(x, y, PV)

    # xi, eta to RA, DEC
    num1 = (xi * deg2rad) / np.cos(CRVAL2 * deg2rad)  # rad
    den1 = 1. - (eta * deg2rad) * np.tan(CRVAL2 * deg2rad)  # rad
    alphap = np.arctan2(num1, den1)  # rad
    RA = CRVAL1 + alphap * rad2deg  # deg
    num2 = (eta * deg2rad + np.tan(CRVAL2 * deg2rad)) * np.cos(alphap)  # rad
    DEC = np.arctan2(num2, den1) * rad2deg  # deg

    return (RA / 15., DEC)  # apply previous transformation


# tangent projection coordinates
def xieta(x, y, PV):  # all in degrees
    r = np.sqrt(x**2 + y**2)
    xicomp = PV[0, 0] + PV[0, 1] * x + PV[0, 2] * y + PV[0, 3] * r + \
        PV[0, 4] * x**2 + PV[0, 5] * x * y + PV[0, 6] * y**2 + \
        PV[0, 7] * x**3 + PV[0, 8] * x**2 * y + PV[0, 9] * x * y**2 + \
        PV[0, 10] * y**3
    etacomp = PV[1, 0] + PV[1, 1] * y + PV[1, 2] * x + PV[1, 3] * r + \
        PV[1, 4] * y**2 + PV[1, 5] * y * x + PV[1, 6] * x**2 + \
        PV[1, 7] * y**3 + PV[1, 8] * y**2 * x + PV[1, 9] * y * x**2 + \
        PV[1, 10] * x**3
    return (xicomp, etacomp)


def MAD(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed(
    )  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))
