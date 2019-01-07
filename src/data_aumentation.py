# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import pandas as pd
import tarfile
import P4J
from astropy.table import Table
from astropy.stats import sigma_clip
from metric_tools import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
import FATS
from gatspy.periodic import LombScargle
from multiprocessing import Pool
import getopt
import sys

sns.set(style="white", color_codes=True, context="notebook", font_scale=1.4)

import warnings
warnings.filterwarnings('ignore')

mainpath = '/Users/jorgetil/Astro/HITS'

global var_class
global empirical
path = '%s/tables/Blind15A_kronPhot_only.csv' % (mainpath)
empirical = pd.read_csv(path)


def GP_fit(time, mag, err, x_pred=None, plot=True):

    kernel = C(constant_value=1., constant_value_bounds=(.1,.5)) *              RBF(length_scale=.3, length_scale_bounds=(0.2, .5))
    #kernel = C(constant_value=1., constant_value_bounds=(0.1,10.)) * \
    #         Matern(length_scale=4., length_scale_bounds=(.1,10.))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=(err)**2, optimizer='fmin_l_bfgs_b',
                                  n_restarts_optimizer=10, normalize_y=True)
    gp.fit(time[:,None], mag)
    x_test=np.linspace(np.min(time),np.max(time),1000)
    y_pred, sigma = gp.predict(x_test[:,None], return_std=True)

    mag_pred = gp.predict(x_pred[:,None], return_std=False)

    if plot:
        plt.figure(figsize=(9,4))
        plt.errorbar(time, mag, yerr=err, fmt='k.', ms=7, lw=1,alpha=1)
        plt.plot(x_test, y_pred, 'r-', lw= 1)
        plt.fill(np.concatenate([x_test, x_test[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma,
                                (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None')
        plt.plot(x_pred, mag_pred, 'b.')
        plt.gca().invert_yaxis()
        plt.show()

    return mag_pred


def GP_fit_periodic(time, mag, err, T, T_1='same', x_pred=None, plot=True):
    if T_1 == 'same':
        T_1 = T
    phase = np.mod(time, T) / T
    sort_idx = np.argsort(phase)
    PHASE = phase[sort_idx]
    MAG = mag[sort_idx]
    ERR = err[sort_idx]
    k=1
    while k < 2:
        MAG = np.concatenate([MAG,MAG])
        PHASE = np.concatenate([PHASE,PHASE+k])
        ERR = np.concatenate([ERR,ERR])
        k += 1
    #PHASE *= T
    kernel = C(constant_value=1., constant_value_bounds=(.1,.5)) *              RBF(length_scale=.3, length_scale_bounds=(0.2, .5))
    #kernel = C(constant_value=1., constant_value_bounds=(0.1,10.)) * \
    #         Matern(length_scale=4., length_scale_bounds=(.1,10.))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=(ERR)**2, optimizer='fmin_l_bfgs_b',
                                  n_restarts_optimizer=10, normalize_y=True)
    gp.fit(PHASE[:,None], MAG)
    x_test=np.linspace(0,2,1000)
    y_pred, sigma = gp.predict(x_test[:,None], return_std=True)

    phase_pred = np.mod(x_pred, T_1) / T_1
    mag_pred = gp.predict(phase_pred[:,None], return_std=False)

    if plot:
        plt.figure(figsize=(9,4))
        plt.errorbar(PHASE, MAG, yerr=ERR, fmt='k.', ms=7, lw=1,alpha=1)
        plt.plot(x_test, y_pred, 'r-', lw= 1)
        plt.fill(np.concatenate([x_test, x_test[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma,
                                (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None')
        plt.plot(phase_pred, mag_pred, 'b.')
        plt.gca().invert_yaxis()
        plt.show()

    return mag_pred


def HiTS_depth_noise(mag):
    scale = np.random.choice(empirical.gMedianKronMag.values, size=1)
    btw = empirical.query('gMedianKronMag > %f-.05 and gMedianKronMag < %f+.05 and gMedianKronMagErr < .5' % (scale,scale))
    errors = np.random.choice(btw.gMedianKronMagErr,
                              size=len(mag))
    new_mag = scale*mag/mag.mean()
    #noise_mag = new_mag * np.random.normal(loc=1, scale=errors**2)
    noise_mag = np.zeros_like(new_mag)
    for k,m in enumerate(new_mag):
        rand = np.random.normal(loc=1, scale=(errors[k]/2)**2)
        noise_mag[k] = m * rand
        #print k,m, rand, noise_mag[k],errors[k]
    return new_mag, noise_mag, errors



def calculate_FATS_features(id_lc, lc_g, verbose=True):

    time, mag, error = np.array(lc_g['MJD'].values), np.array(lc_g['MAG_KRON'].values), np.array(lc_g['MAGERR_KRON'].values)
    lc = np.array([mag, time, error])

    if verbose: print 'N Epochs? %i' % len(lc[0])
    # Available imput data
    fats = FATS.FeatureSpace(Data = ['magnitude','time', 'error'],
                                excludeList = ['StetsonJ', 'StetsonL', 'Eta_color', 'Q31_color', 'Color'])
    feat = fats.calculateFeature(lc)
    result = feat.result(method = 'dict')
    result_df = pd.DataFrame.from_dict(result, orient = 'index')
    result_df.columns = [id_lc]
    return result_df.T

def P4J_period(lc_g):

    try:
        time, mag, error = np.array(lc_g['MJD'].values), np.array(lc_g['MAG_KRON'].values), np.array(lc_g['MAGERR_KRON'].values)

        WMCC_model = P4J.periodogram(M=1, method='WMCC')
        WMCC_model.fit(time, mag-np.mean(mag), error)

        per_f = np.max(time) - np.min(time)
        freq, obj = WMCC_model.grid_search(fmin=1/30., fmax=1/.01, fres_coarse=2, fres_fine=0.05, n_local_max=10)
        fbest = WMCC_model.get_best_frequency()
        WMCC_model.fit_extreme_cdf(n_bootstrap=40, n_frequencies=40)
        confidence_best_freq = WMCC_model.get_confidence(fbest[1])
        return 1./fbest[0], confidence_best_freq
    except:
        print 'P4J fail calculating period...'
        return None, None

def gatspy_period(lc_g):

    time, mag, error = np.array(lc_g['MJD'].values), np.array(lc_g['MAG_KRON'].values), np.array(lc_g['MAGERR_KRON'].values)

    per_f = (np.max(time) - np.min(time))*10
    periods = np.linspace(.01, per_f, 10000)

    model = LombScargle(fit_offset=True).fit(time, mag, error)
    power = model.score(periods)

    best_per = periods[np.argmax(power)]

    return best_per

def preprocess_lc(lc, min_mag=15.):
    #time, mag, error = np.array(lc_g['MJD'].quantity), np.array(lc_g['MAG_KRON'].quantity),\
    #                   np.array(lc_g['MAGERR_KRON'].quantity)
    # remove saturated points
    #print 'Original size: %i |' % len(lc),
    lc = lc[lc['MAG_KRON'].values >= min_mag]

    # sigma clipping
    filtered_data = sigma_clip(np.array(lc['MAG_KRON'].values), sigma=4, iters=1, cenfunc=np.mean, copy=False)
    lc = lc[~filtered_data.mask]
    #print ' clipped: %i |' % np.sum(filtered_data.mask),
    #print 'Final size: %i' % len(lc)

    return lc

def get_features(data):
    lc = data[0]
    k = data[1]
    name = data[2]
    id_lc = '%s_%i' % (name, k+1)
    print '\r LC %i: %s...' % (k, id_lc)
    try:
        lc = preprocess_lc(lc)
        featur = calculate_FATS_features(id_lc, lc, verbose=False)
        if True:
            period, conf = P4J_period(lc)
            featur['PeriodWMCC'] = period
            featur['PeriodWMCC_conf'] = conf

        if True:
            period = gatspy_period(lc)
            featur['PeriodGLS'] = period
        return featur
    except:
        return None


def run_code(var_class = 'RRLYR', sample = 50, n_gen = 20, n_process = 3, plot=False):
    table_file = '%s/tables/Blind15A_training_set_goodP.csv' % (mainpath)
    table_15 = pd.read_csv(table_file)
    table_15.set_index('internalID', inplace=True)
    print 'Training set opened...'

    class_example = table_15.query('Var_Type == "%s"' % (var_class))
    if sample != 'False':
        class_example = class_example.sample(sample)

    time_all = []
    for k in range(50):
        print '\r', k,
        if k == 2:continue
        path = '%s/INFO/times/Blind15A_%02i_epochs_g.txt' % (mainpath, int(k+1))
        aux = np.loadtxt(path)
        time_all.append(aux)
    print 'Empirical times loaded'

    all_lcs, all_per = [], []
    count = 0
    for item in class_example.iterrows():
        if item[0] == 'Blind15A_22_S27_0428_0159' or \
           item[0] == 'Blind15A_33_S19_1145_0097' or \
           item[0] == 'Blind15A_05_S23_0455_3568' or \
           item[0] == 'Blind15A_19_N4_0938_3573': continue

        T = float(item[1]['PeriodLS'])
        T_W = float(item[1]['PeriodWMCC'])
        T_G = float(item[1]['PeriodGLS'])
        print item[0]
        field, CCD, X, Y = re.findall(
                    r'(\w+\d+\w?\_\d\d?)\_(\w\d+?)\_(\d+)\_(\d+)', item[0])[0]
        time, mag, err, time2, mag2, err2 = give_me_lc(field, CCD, X, Y,extract=False)
        filtered_data = sigma_clip(mag, sigma=3, iters=1,
                                   cenfunc=np.mean, copy=False)
        time = time[~filtered_data.mask]
        mag = mag[~filtered_data.mask]
        err = err[~filtered_data.mask]
        print 'Period: %f days' % (T)
        n = np.random.choice(range(49))
        hits_time = time_all[n][:,1]
        # predicted magnitude for GP
        if var_class == 'NV':
            mag_obs = GP_fit(time, mag, err, plot=plot, x_pred=hits_time)
        else:
            mag_obs = GP_fit_periodic(time, mag, err, T, T_1=T, plot=plot, x_pred=hits_time)

        # scale to HiTS depth and add empirical uncertainties
        hits_mag, hits_noise_mag, hits_err = HiTS_depth_noise(mag_obs)
        if hits_mag.mean() < 20:
            size = int(len(hits_time)*np.random.uniform(.95,1., size=1))
        elif hits_mag.mean() < 22 and hits_mag.mean() > 20:
            size = int(len(hits_time)*np.random.uniform(.85,1., size=1))
        else:
            size = int(len(hits_time)*np.random.uniform(.75,1., size=1))
        print hits_mag.mean(), size,
        idx = np.sort(np.random.choice(np.arange(len(hits_time)),
                           size=size, replace=False))
        hits_time = hits_time[idx]
        hits_mag = hits_mag[idx]
        hits_noise_mag = hits_noise_mag[idx]
        hits_err = hits_err[idx]

        if plot:
            if var_class == 'NV':
                plt.figure(figsize=(9,4))
                plt.errorbar(hits_time, hits_mag, yerr=hits_err, fmt='k.', ms=7, lw=1,alpha=1)
                plt.errorbar(hits_time, hits_noise_mag, yerr=hits_err, fmt='r.', ms=7, lw=1,alpha=1)
                plt.gca().invert_yaxis()
                plt.show()

            else:
                phase = np.mod(hits_time, T) / T
                sort_idx = np.argsort(phase)

                hits_PHASE = phase[sort_idx]
                hits_MAG = hits_mag[sort_idx]
                hits_noise_MAG = hits_noise_mag[sort_idx]
                hits_ERR = hits_err[sort_idx]

                hits_MAG = np.concatenate([hits_MAG,hits_MAG])
                hits_noise_MAG = np.concatenate([hits_noise_MAG,hits_noise_MAG])
                hits_PHASE = np.concatenate([hits_PHASE,hits_PHASE+1])
                hits_ERR = np.concatenate([hits_ERR,hits_ERR])

                plt.errorbar(hits_PHASE, hits_MAG, yerr=hits_ERR, fmt='k.', ms=7, lw=1,alpha=1)
                plt.errorbar(hits_PHASE, hits_noise_MAG, yerr=hits_ERR, fmt='r.', ms=7, lw=1,alpha=1)
                plt.gca().invert_yaxis()
                plt.show()


        for k in range(n_gen):
            print '\r', k,
            if var_class == 'ROTVAR':
                up_lim = 10*T
                if up_lim > 15: up_lim = 15.
                T_1 = np.random.uniform(0.01, up_lim)
            elif var_class == 'RRLYR':
                T_1 = np.random.uniform(.05, 1,1)
            elif var_class == 'RRLYR':
                T_1 = np.random.uniform(0.02, 0.33)
            elif var_class == 'NV':
                T_1 = 'same'

            # HiTS sampling function
            #hits_time = time_obs[HiTS_sample_func(time_obs)]
            n = np.random.choice(range(49))
            hits_time = time_all[n][:,1]
            # predicted magnitude for
            if var_class == 'NV':
                mag_obs = GP_fit(time, mag, err, plot=False, x_pred=hits_time)
            else:
                mag_obs = GP_fit_periodic(time, mag, err, T, T_1=T_1, plot=False, x_pred=hits_time)
            # scale to HiTS depth and add empirical uncertainties
            hits_mag, hits_noise_mag, hits_err = HiTS_depth_noise(mag_obs)
            if hits_mag.mean() < 20:
                size = int(len(hits_time)*np.random.uniform(.95,1., size=1))
            elif hits_mag.mean() < 22 and hits_mag.mean() > 20:
                size = int(len(hits_time)*np.random.uniform(.85,1., size=1))
            else:
                size = int(len(hits_time)*np.random.uniform(.75,1., size=1))
            print '\r',k, T_1, hits_mag.mean(), size,len(hits_time),
            idx = np.sort(np.random.choice(np.arange(len(hits_time)),
                           size=size, replace=False))
            hits_time = hits_time[idx]
            hits_mag = hits_mag[idx]
            hits_noise_mag = hits_noise_mag[idx]
            hits_err = hits_err[idx]

            df = pd.DataFrame(np.array([hits_time, hits_noise_mag, hits_err]).T,
                              columns=['MJD','MAG_KRON','MAGERR_KRON'])
            if False:
                path = '%s/synt_lcs/%s_%i_%.6f.csv' % (mainpath, var_class,k,T_1)
                df.to_csv(path)
            all_lcs.append(df)
            all_per.append(T_1)

    print 'LC generated...'

    print 'Calculating features values...'
    #p = Pool(processes=n_process)
    #frames = p.map(get_features, zip(all_lcs, range(len(all_lcs)), [var_class]*len(all_lcs)))
    frames = []
    for data in zip(all_lcs, range(len(all_lcs)), [var_class]*len(all_lcs)):
        frames.append(get_features(data))

    fats_feat = pd.concat(frames, axis = 0)
    fats_feat = fats_feat[np.sort(fats_feat.columns.values)]
    fats_feat['Var_Type'] = var_class
    print fats_feat.shape

    fats_feat.to_csv('%s/tables/Syntectic_%s_features_noise.csv' % (mainpath, var_class))
    print 'Done!'


if __name__ == '__main__':

    clase = 'RRLYR'
    use = 50
    total = 20
    cores = 3

    if len(sys.argv) == 1:
        print help
        sys.exit()

    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'C:s:g:p:')
    except getopt.GetoptError, err:
        print help
        sys.exit()

    for o, a in optlist:
        if o in ('-C'):
            clase = str(a)
        elif o in ('-s'):
			use = int(a)
        elif o in ('-g'):
            total = int(a)
        elif o in ('-p'):
            cores = int(a)
        else:
            continue

    print clase, use, total, cores
    run_code(var_class=clase, sample=use, n_gen=total, n_process=cores, plot=True)
