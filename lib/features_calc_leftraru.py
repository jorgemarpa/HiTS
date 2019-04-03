import os
import glob
import sys
import re
import itertools
import numpy as np
import FATS
import getopt
import tarfile
from datetime import datetime
import warnings
import pandas as pd
import P4J
from astropy.table import Table
from astropy.io import ascii
from gatspy.periodic import LombScargle
from astropy.stats import sigma_clip
from scipy.spatial import cKDTree
from scipy.stats import chi2

import emcee

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'

help = '''
This script read light-curves of a given field/CCD and
then calculate features using FATS

Usage: python lc_features.py [OPTION...]

    -F : string with field
    -C : string with CCD or 'all'
'''


def rms(x):
    return np.sqrt(x.dot(x) / x.size)


def read_lc_field(field, verbose=True, band='g'):

    g_list = np.sort(glob.glob('%s/lightcurves/galaxy/%s/%s_*_%s_psf_ff.csv' %
                               (jorgepath, field, field, band)),
                     kind='mergesort')

    if len(g_list) == 0:
        print '\t No LC in this field...'
        return [None], [None]

    print '    Number of LC for %s: %i' % (band, len(g_list))

    lcs_id = []
    lcs_g = []
    for k, g_lc_file in enumerate(g_list):
        try:
            g_lc = pd.read_csv(g_lc_file, skiprows=5)
            aux_lc = Table.from_pandas(g_lc)
        except BaseException:
            continue
        aux_id = os.path.split(g_lc_file)[1].replace('_%s_02' % (band), '')
        aux_id = aux_id.replace('_g_psf_ff.csv', '')
        lcs_id.append(aux_id)
        if verbose:
            print '\t', lcs_id[-1]

        # aux_lc = Table(g_lc)
        lcs_g.append(aux_lc)

    return lcs_id, lcs_g


def read_lc(field, CCD, occ=15, verbose=True, band='g'):

    g_list = np.sort(glob.glob('%s/lightcurves/%s/%s/%s_%s_*_%s.dat' %
                               (jorgepath, field, CCD, field, CCD, band)),
                     kind='mergesort')
    if len(g_list) == 0:
        print '\tUncompressing tar.gz files...'
        print field[:8]
        tar_file = '%s/lightcurves/%s/%s/%s_%s_LC_occ%i.tar.gz' % (
            jorgepath, field, CCD, field, CCD, occ)
        print tar_file
        if not os.path.exists(tar_file):
            print '\t\t No tar file in this folder...'
        else:
            tar = tarfile.open(tar_file, 'r')
            tar.extractall('%s/lightcurves/%s/%s/' % (jorgepath, field, CCD))

    g_list = np.sort(glob.glob('%s/lightcurves/%s/%s/%s_%s_*_%s.dat' %
                               (jorgepath, field, CCD, field, CCD, band)),
                     kind='mergesort')

    if len(g_list) == 0:
        print '\t No LC in this CCD...'
        return [None], [None]

    print '    Number of LC for %s: %i' % (band, len(g_list))

    lcs_id = []
    lcs_g = []
    for k, g_lc_file in enumerate(g_list):
        try:
            g_lc = Table.read(g_lc_file, format='ascii')
        except BaseException:
            continue
        lcs_id.append(
            os.path.split(g_lc_file)[1].replace('_%s.dat' % (band), ''))
        if verbose:
            print '\t', lcs_id[-1]
        # Loading g LC
        # if len(g_lc) < 20:
        #     continue
        lcs_g.append(g_lc[['MJD', 'MAG_KRON', 'MAGERR_KRON']])

    print 'Removing LC files...'
    filelist = glob.glob('%s/lightcurves/%s/%s/%s_%s_*.dat' %
                         (jorgepath, field, CCD, field, CCD))
    for f in filelist:
        os.remove(f)

    return lcs_id, lcs_g


def calculate_FATS_features(id_lc, lc_g, lc_r=None,
                            preprocess=True, sync=False, verbose=True,
                            mjd_key='MJD', mag_key='MAG_KRON',
                            emag_key='MAGERR_KRON'):

    time, mag, error = np.array(lc_g[mjd_key].quantity), \
        np.array(lc_g[mag_key].quantity),\
        np.array(lc_g[emag_key].quantity)
    lc = np.array([mag, time, error])
    if lc_r is not None:
        time2, mag2, error2 = lc_r['MJD'], lc_r['MAG_KRON'], \
            lc_r['MAGERR_KRON']
        lc = np.array([mag, time, error, mag2])

    if verbose:
        print 'N Epochs? %i' % len(lc[0])

    # We preprocess the data
    if preprocess:
        preproccesed_data = FATS.Preprocess_LC(mag, time, error)
        [mag, time, error] = preproccesed_data.Preprocess()
        lc = np.array([mag, time, error])
        if lc_r is not None:
            preproccesed_data = FATS.Preprocess_LC(mag2, time2, error2)
            [mag2, time2, error2] = preproccesed_data.Preprocess()
            lc = np.array([mag, time, error, mag2])

    if verbose:
        print 'N Epochs? %i' % len(lc[0])
    # We synchronize the data
    if sync:
        if len(mag) != len(mag2):
            [aligned_mag, aligned_mag2, aligned_time,
             aligned_error, aligned_error2] = FATS.Align_LC(time, time2, mag,
                                                            mag2, error,
                                                            error2)
        lc = np.array([mag, time, error, mag2, aligned_mag,
                       aligned_mag2, aligned_time, aligned_error,
                       aligned_error2])

    # Available imput data
    if lc_r is None:
        gal_feat_list = ['Amplitude', 'AndersonDarling', 'Autocor_length',
                         'Beyond1Std', 'CAR_sigma', 'CAR_mean', 'CAR_tau',
                         'Con', 'Eta_e', 'FluxPercentileRatioMid20',
                         'FluxPercentileRatioMid35',
                         'FluxPercentileRatioMid50',
                         'FluxPercentileRatioMid65',
                         'FluxPercentileRatioMid80', 'Gskew', 'LinearTrend',
                         'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
                         'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude',
                         'PercentDifferenceFluxPercentile', 'PeriodLS',
                         'Period_fit', 'Psi_CS', 'Psi_eta', 'Q31', 'Rcs',
                         'Skew', 'SlottedA_length', 'SmallKurtosis', 'Std',
                         'StetsonK', 'StetsonK_AC']
        fats = FATS.FeatureSpace(Data=['magnitude', 'time', 'error'],
                                 featureList=gal_feat_list,
                                 excludeList=['StetsonJ', 'StetsonL',
                                              'Eta_color', 'Q31_color',
                                              'Color'])
    else:
        if sync:
            fats = FATS.FeatureSpace(Data='all')
        else:
            fats = FATS.FeatureSpace(Data=['magnitude', 'time', 'error',
                                           'magnitude2'],
                                     excludeList=['StetsonJ', 'StetsonL',
                                                  'Eta_color', 'Q31_color'])

    feat = fats.calculateFeature(lc)
    result = feat.result(method='dict')
    result_df = pd.DataFrame.from_dict(result, orient='index')
    result_df.columns = [id_lc]
    return result_df.T


def P4J_period(lc_g):

    # if True:
    try:
        time, mag, error = np.array(lc_g['MJD'].quantity), \
            np.array(lc_g['MAG_KRON'].quantity),\
            np.array(lc_g['MAGERR_KRON'].quantity)

        P4J_model = P4J.periodogram(method='QMIEU')

        P4J_model.set_data(time, mag, error, whitten=True)
        P4J_model.frequency_grid_evaluation(fmin=1 / 30., fmax=1 / .01,
                                            fresolution=0.1)
        P4J_model.finetune_best_frequencies(fresolution=0.01,
                                            n_local_optima=10)
        fbest = P4J_model.get_best_frequency()

        return 1. / fbest
    except BaseException:
        print 'P4J fail calculating period...'
        return None


def gatspy_period(lc_g):

    time, mag, error = np.array(lc_g['MJD'].quantity), \
        np.array(lc_g['MAG_KRON'].quantity),\
        np.array(lc_g['MAGERR_KRON'].quantity)

    per_f = np.max(time) - np.min(time)
    periods = np.linspace(.01, per_f, 4000)

    model = LombScargle(fit_offset=True).fit(time, mag, error)
    power = model.score(periods)

    best_per = periods[np.argmax(power)]

    return best_per


def get_colors(field, ccd, id):

    bands = ['u', 'g', 'r', 'i']

    photometry_path = '%s/catalogues/%s/%s/%s_%s_HiTS_n3_table.csv' % \
        (jorgepath, field, CCD, field, CCD)
    phot_table = pd.read_csv(photometry_path)

    for b in bands:
        if '%sMedianKronMag' % (b) in phot_table.columns:
            continue
            # print 'filter %s exist' % (b)
        else:
            # print 'No filter'
            bands.remove(b)
    print 'Effective bands: ', bands

    _, _, X, Y = re.findall(
        r'(\w+\d+\w?\_\d\d?)\_(\w\d+?)\_(\S+)\_(\S+)', id)[0]
    X, Y = int(X), int(Y)
    tree = cKDTree(phot_table[['X', 'Y']].values)
    dist, idx = tree.query([X, Y], k=1, distance_upper_bound=.75)
    print dist, idx
    if dist < .75:
        row = phot_table.iloc[idx]
        # print row
    else:
        row = phot_table[phot_table.internalID == id]
        if row.empty and np.isinf(dist):
            print 'No match in catalog table...'
            return []
        else:
            row = row.iloc[0]
        # print row

    colorss = []
    for b1, b2 in itertools.combinations(bands, 2):
        b1_ = row['%sMedianKronMag' % (b1)]
        b2_ = row['%sMedianKronMag' % (b2)]
        if b1_ > 0 and b2_ > 0:
            colorss.append(['%s-%s' % (b1, b2), b1_ - b2_])

    return colorss


def median_error(lc_g, mjd_key='MJD', mag_key='MAG_KRON',
                 emag_key='MAGERR_KRON'):
    _, _, error = np.array(lc_g[mjd_key].quantity), \
        np.array(lc_g[mag_key].quantity),\
        np.array(lc_g[emag_key].quantity)

    return np.median(error)


def flux_variation(lc_g, mjd_key='MJD', mag_key='MAG_KRON',
                   emag_key='MAGERR_KRON'):
    _, mag, err = np.array(lc_g[mjd_key].quantity), \
        np.array(lc_g[mag_key].quantity),\
        np.array(lc_g[emag_key].quantity)
    # excess2 = (np.std(mag)**2 - rms(error)**2) / np.mean(mag)**2
    # excess2_err = np.sum(((mag - mag.mean())**2 -
    #                       error**2 -
    #                       (error * mag.mean())**2)**2) / len(mag)
    # if excess2 < 0:
    #     excess2 = 0
    #     excess2_err = 0
    # excess2 = np.sqrt(excess2)
    #
    # return excess2, excess2_err / (mag.mean()**2 * len(mag)**.5), \
    #     np.std(mag) / rms(error)
    mean = np.mean(mag)
    nepochs = float(len(mag))
    a = (mag - mean)**2
    ex_var = (np.sum(a - err**2) / ((nepochs * (mean**2))))
    sd = np.sqrt((1. / (nepochs - 1)) * np.sum(((a - err**2) -
                                                ex_var * (mean**2))**2))
    ex_verr = sd / ((mean**2) * np.sqrt(nepochs))

    return ex_var, ex_verr, np.std(mag) / rms(err)


def ratio_max_min_flux(lc_g, mjd_key='MJD', mag_key='MAG_KRON',
                       emag_key='MAGERR_KRON'):
    _, mag, _ = np.array(lc_g[mjd_key].quantity), \
        np.array(lc_g[mag_key].quantity),\
        np.array(lc_g[emag_key].quantity)
    return np.max(mag) / np.min(mag)


def chi_sq_prob(lc_g, mjd_key='MJD', mag_key='MAG_KRON',
                emag_key='MAGERR_KRON'):
    _, mag, err = np.array(lc_g[mjd_key].quantity), \
        np.array(lc_g[mag_key].quantity),\
        np.array(lc_g[emag_key].quantity)
    # chi_sq = np.sqrt(np.sum((mag - mag.mean()) ** 2 / error ** 2))
    #
    # return 1 - chi2.sf(chi_sq, len(mag) - 1)
    mean = np.mean(mag)
    nepochs = float(len(mag))
    chi = np.sum((mag - mean)**2. / err**2.)
    q_chi = chi2.cdf(chi, (nepochs - 1))

    return q_chi


def n_points_out(lc_g, mjd_key='MJD', mag_key='MAG_KRON',
                 emag_key='MAGERR_KRON'):
    _, mag, _ = np.array(lc_g[mjd_key].quantity), \
        np.array(lc_g[mag_key].quantity),\
        np.array(lc_g[emag_key].quantity)
    n = 2.
    mask = (mag > mag.mean() - n * mag.std()) & \
           (mag < mag.mean() + n * mag.std())
    return len(mag[~mask]) / float(len(mag))


def preprocess_lc(lc, min_mag=15, mag_key='MAG_KRON'):
    # time, mag, error = np.array(lc_g['MJD'].quantity), \
    #                   np.array(lc_g['MAG_KRON'].quantity),\
    #                   np.array(lc_g['MAGERR_KRON'].quantity)
    # remove saturated points
    print 'Original size: %i |' % len(lc),
    lc = lc[lc[mag_key].quantity >= min_mag]

    # sigma clipping
    filtered_data = sigma_clip(np.array(lc[mag_key].quantity), sigma=3,
                               iters=1, cenfunc=np.mean, copy=False)
    lc = lc[~filtered_data.mask]
    print ' clipped: %i |' % np.sum(filtered_data.mask),
    print 'Final size: %i' % len(lc)

    return lc


def var_parameters(jd, mag, err):
    # function to calculate the probability of a light curve to be variable,
    # and the excess variance

    # nepochs, maxmag, minmag, mean, variance, skew, kurt = st.describe(mag)

    mean = np.mean(mag)
    nepochs = float(len(jd))

    chi = np.sum((mag - mean)**2. / err**2.)
    q_chi = chi2.cdf(chi, (nepochs - 1))

    a = (mag - mean)**2
    ex_var = (np.sum(a - err**2) / ((nepochs * (mean**2))))
    sd = np.sqrt((1. / (nepochs - 1)) *
                 np.sum(((a - err**2) - ex_var * (mean**2))**2))
    ex_verr = sd / ((mean**2) * np.sqrt(nepochs))

    return [q_chi, ex_var, ex_verr]


#######################################
# determine single SF using emcee

# calculate an array with (m(t)-m(t+tau)), whit (err(t)^2+err(t+tau)^2)
# and another with tau=dt
def SFarray(jd, mag, err):
    sfarray = []
    tauarray = []
    errarray = []
    for i, item in enumerate(mag):
        for j in range(i + 1, len(mag)):
            dm = mag[i] - mag[j]
            sigma = err[i]**2 + err[j]**2
            dt = (jd[j] - jd[i])
            sfarray.append(np.abs(dm))
            tauarray.append(dt)
            errarray.append(sigma)
    sfarray = np.array(sfarray)
    tauarray = np.array(tauarray)
    errarray = np.array(errarray)
    return (tauarray, sfarray, errarray)


def Vmod(dt, A, gamma):  # model
    return (A * ((dt / 1.)**gamma))


def Veff2(dt, sigma, A, gamma):  # model plus the error
    return ((Vmod(dt, A, gamma))**2 + sigma)


def like_one(theta, dt, dmag, sigma):  # likelihood for one value of dmag

    gamma, A = theta
    aux = (1 / np.sqrt(2 * np.pi * Veff2(dt, sigma, A, gamma))) * \
        np.exp(-1.0 * (dmag**2) / (2.0 * Veff2(dt, sigma, A, gamma)))

    return aux


# we define the likelihood following the same function used by Schmidt et
# al. 2010
def lnlike(theta, dtarray, dmagarray, sigmaarray):
    gamma, A = theta

    '''
    aux=0.0

    for i in xrange(len(dtarray)):
    aux+=np.log(like_one(theta,dtarray[i],dmagarray[i],sigmaarray[i]))
    '''

    aux = np.sum(np.log(like_one(theta, dtarray, dmagarray, sigmaarray)))

    return aux


def lnprior(theta):
    # we define the prior following the same functions
    # implemented by Schmidt et al. 2010

    gamma, A = theta

    if 0.0 < gamma and 0.0 < A < 2.0:
        return (np.log(1.0 / A) + np.log(1.0 / (1.0 + (gamma**2.0))))

    return -np.inf
    # return -(10**32)

# the product of the prior and the likelihood in a logaritmic format
def lnprob(theta, dtarray, dmagarray, sigmaarray):

    lp = lnprior(theta)

    if not np.isfinite(lp):
        # if (lp==-(10**32)):
        return -np.inf
        # return -(10**32)
    return lp + lnlike(theta, dtarray, dmagarray, sigmaarray)

# function that fits the values of A and gamma using mcmc with the package
# emcee.
def fitSF_mcmc(lc_g, ndim=2, nwalkers=100, nit=250, nthr=1,
               mjd_key='MJD', mag_key='MAG_KRON', emag_key='MAGERR_KRON'):
    # It recives the array with dt in days, dmag and the errors, besides the
    # number of dimensions of the parameters, the number of walkers and the
    # number of iterations
    time, mag, error = np.array(lc_g[mjd_key].quantity), \
        np.array(lc_g[mag_key].quantity),\
        np.array(lc_g[emag_key].quantity)

    # we calculate the arrays of dm, dt and sigma
    dtarray, dmagarray, sigmaarray = SFarray(time, mag, error)

    ndt = np.where((dtarray <= 10) & (dtarray >= 0))
    dtarray = dtarray[ndt]
    dmagarray = dmagarray[ndt]
    sigmaarray = sigmaarray[ndt]

    # definition of the optimal initial position of the walkers

    # p0 = np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim)) #gess to
    # start the burn in fase
    p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim)) * 0.1 + 0.5

    # run mcmc
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    threads=nthr,
                                    args=(dtarray, dmagarray, sigmaarray))

    # from pos we have a best gess of the initial walkers
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    print("Running MCMC...")
    sampler.run_mcmc(pos, nit, rstate0=state)
    print("Done.")

    # Compute the quantiles.
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    A_fin = samples[:, 1]
    gamma_fin = samples[:, 0]

    A_mcmc = (np.percentile(A_fin, 50),
              np.percentile(A_fin, 50) - np.percentile(A_fin, 15.865),
              np.percentile(A_fin, 84.135) - np.percentile(A_fin, 50))
    g_mcmc = (np.percentile(gamma_fin, 50),
              np.percentile(gamma_fin, 50) - np.percentile(gamma_fin, 15.865),
              np.percentile(gamma_fin, 84.135) - np.percentile(gamma_fin, 50))

    sampler.reset()
    return A_mcmc, g_mcmc


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    startTime = datetime.now()
    field = 'Blind15A_01'
    CCD = 'N1'
    occ = 15
    p4j_bool = True
    gatspy_bool = True
    color_bool = True
    band = 'g'
    file_path = ''
    n_app = 1

    if len(sys.argv) == 1:
        print help
        sys.exit()

    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'F:C:b:o:a:')
    except getopt.GetoptError as err:
        print help
        sys.exit()

    for o, a in optlist:
        if o in ('-F'):
            field = str(a)
        elif o in ('-C'):
            CCD = str(a)
        elif o in ('-o'):
            occ = int(a)
        elif o in ('-b'):
            band = str(a)
        elif o in ('-f'):
            file_path = str(a)
        elif o in ('-a'):
            n_app = int(a)
        else:
            continue

    print 'Field: %s' % (field)
    print 'CCD: %s' % (CCD)

    if field[:8] == 'Blind15A':
        color_bool = True
    else:
        color_bool = False

    # Load lightcurves from archive for each CCDs
    print 'Loading LC for this field/CCD...'

    if file_path:
        lcs_id = []
        lcs_g = []
        data = ascii.read(file_path, format='cds')
        print data
    else:
        if CCD == 'all':
            p4j_bool = False
            gatspy_bool = False
            color_bool = False
            lcs_id, lcs_g = read_lc_field(field, band=band)
            mjd_key = 'mjd'
            mag_key = 'aperture_mag_%i' % (n_app)
            emag_key = 'aperture_mag_err_%i' % (n_app)

        else:
            lcs_id, lcs_g = read_lc(field, CCD, occ=occ, band=band)
            mjd_key = 'MJD'
            mag_key = 'MAG_KRON'
            emag_key = 'MAGERR_KRON'
            if len(lcs_g) == 0:
                print 'No LC in this field/CCD!!!'
                sys.exit()

    print '---------------------------'

    print 'It took', (datetime.now() - startTime), 'seconds'
    print 'Number of LC for %s: %i' % (band, len(lcs_g))

    # Calculate features for all LC using FATS
    print 'Calculating features from LCs using app: %s...' % (mag_key)
    print ''
    startTime = datetime.now()
    count = 1
    frames = []
    for id_lc, lc_g in zip(lcs_id, lcs_g):
        print 'LC %i: %s...' % (count, id_lc), len(lc_g)
        lc_g = preprocess_lc(lc_g, mag_key=mag_key)
        if len(lc_g) < 10:
            continue
        # print len(lc_g)
        # print lc_g
        # sys.exit()

        featur = calculate_FATS_features(id_lc, lc_g, lc_r=None,
                                         verbose=False, preprocess=False,
                                         mjd_key=mjd_key,
                                         mag_key=mag_key,
                                         emag_key=emag_key)
        if p4j_bool:
            period = P4J_period(lc_g)
            featur['PeriodQMIEU'] = period

        if gatspy_bool:
            per = gatspy_period(lc_g)
            featur['PeriodGLS'] = per

        if color_bool:
            color = get_colors(field, CCD, id_lc)
            if len(color) != 0:
                for col in color:
                    featur[col[0]] = col[1]

        if True:
            med_err = median_error(lc_g, mjd_key=mjd_key,
                                   mag_key=mag_key, emag_key=emag_key)
            featur['MedianErr'] = med_err

            f_var, f_var_err, StdErr = flux_variation(lc_g, mjd_key=mjd_key,
                                                      mag_key=mag_key,
                                                      emag_key=emag_key)
            featur['ExcessVariance'] = f_var
            featur['ExcessVariance_e'] = f_var_err
            featur['ExcessVariance_cor'] = f_var - f_var_err
            featur['StdToErr'] = StdErr

            r_max = ratio_max_min_flux(lc_g, mjd_key=mjd_key,
                                       mag_key=mag_key, emag_key=emag_key)
            featur['RatioMaxMinFlux'] = r_max

            chi_sq_prob_value = chi_sq_prob(lc_g, mjd_key=mjd_key,
                                            mag_key=mag_key, emag_key=emag_key)
            featur['ChiSqProb'] = chi_sq_prob_value

            n_points = n_points_out(lc_g, mjd_key=mjd_key,
                                    mag_key=mag_key, emag_key=emag_key)
            featur['Nout2sigma'] = n_points

            if False:
                try:
                    Amp, gamma = fitSF_mcmc(lc_g, mjd_key=mjd_key,
                                            mag_key=mag_key, emag_key=emag_key)
                    featur['SFamp'] = Amp[0]
                    featur['SFampStd1'] = Amp[1]
                    featur['SFampStd2'] = Amp[2]
                    featur['SFgamma'] = gamma[0]
                    featur['SFgammaStd1'] = gamma[1]
                    featur['SFgammaStd2'] = gamma[2]
                except:
                    continue

        frames.append(featur)
        count += 1
        print '---------------------------'
        # if count == 5: break

    fats_feat = pd.concat(frames, axis=0)
    fats_feat = fats_feat[np.sort(fats_feat.columns.values)]

    print fats_feat.columns.values

    if not os.path.exists('%s/features/%s' % (jorgepath, field)):
        print "Creating field folder"
        os.makedirs('%s/features/%s' % (jorgepath, field))
    if CCD == 'all':
        fats_feat.to_csv('%s/features/%s/%s_%s_galaxy_psf_app%i.csv' %
                         (jorgepath, field, field, CCD, n_app), compression='gzip')
    else:
        fats_feat.to_csv('%s/features/%s/%s_%s_prepro.csv' %
                         (jorgepath, field, field, CCD), compression='gzip')
    print 'It took', (datetime.now() - startTime), 'seconds'
