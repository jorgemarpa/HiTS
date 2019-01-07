import sys
import os
import glob
import re
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
# import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit
# import seaborn as sns
# sns.set(style="white", color_codes=True, context="poster")

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
sharepath = '/home/apps/astro/home/jmartinez'
thresh = 1.0      # threshold of catalog
minarea = 3      # min area for detection


def empty_aperture_all_CCD(field, epoch, n_aper=500,
                           plots=False, verbose=True):

    # warnings.filterwarnings("ignore")

    CHIPS = np.loadtxt('%s/info/ccds.txt' % (jorgepath), comments='#',
                       dtype=str)
    empty_ap_all = []
    GAIN = []
    errors = []
    cata_all = []
    empty_ap_ccd = []

    for CCD in CHIPS[:]:
        # if CCD == 'S7': continue

        if verbose:
            print 'CCD %s' % CCD
        #######################################################################
        imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (
            astropath, field, CCD, field, CCD, epoch)
        if not os.path.exists(imag_file):
            if verbose:
                print 'No image file: %s' % (imag_file)
            continue
        hdufits_ima = fits.open(imag_file)
        imag_data = hdufits_ima[0].data
        if verbose:
            print 'SCIENCE IMAGE SHAPE \t\t', imag_data.shape

        #######################################################################
        backg_file = "%s/fits/%s/%s/CHECKIMAGE/%s_%s_%s_image_crblaster_background_thresh%s_minarea%i_backsize64.fits.fz" % \
            (jorgepath, field, CCD, field, CCD, epoch, str(thresh), minarea)
        if not os.path.exists(backg_file):
            print 'No image file: %s' % (backg_file)
            sys.exit()
        hdufits_bac = fits.open(backg_file)
        back_data = hdufits_bac[1].data
        if verbose:
            print 'SCIENCE IMAGE SHAPE \t\t', back_data.shape

        ######################################################################
        segm_file = "%s/fits/%s/%s/CHECKIMAGE/%s_%s_%s_image_crblaster_segmentation_thresh%s_minarea%i_backsize64.fits.fz" % \
            (jorgepath, field, CCD, field, CCD, epoch, str(thresh), minarea)
        if not os.path.exists(segm_file):
            if verbose:
                print 'No segmentation image file: %s' % (segm_file)
            continue
        hdufits_seg = fits.open(segm_file)
        segm_data = hdufits_seg[1].data
        segm_mask = (segm_data > 0)
        if verbose:
            print 'SEGMENTATION IMAGE SHAPE \t', segm_data.shape

        #######################################################################
        files_epoch = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_cat.dat" % \
            (jorgepath, field, CCD, field, CCD, epoch, str(thresh),
             str(minarea))
        if not os.path.exists(files_epoch):
            if verbose:
                print 'No file: %s' % files_epoch
            continue
        cata = np.loadtxt(files_epoch, comments='#')
        errors.append(cata[:, 6])
        cata_all.append(cata)
        if verbose:
            print 'CATALOGUE SHAPE 	\t', cata.shape

        FWHM = hdufits_ima[0].header['FWHM']
        FWHM_ceil = np.ceil(FWHM) + 1
        if verbose:
            print 'FWHM = %f' % FWHM

        ima_back = imag_data - back_data
        n = 0
        empty_ap = []
        while n < n_aper:
            random_center = np.random.random_integers(
                0 + FWHM_ceil, 2048 - FWHM_ceil, 2) * (2, 1)
            if verbose:
                print 'Ramdom position ', random_center
            if verbose:
                print 'Object in patch?', np.invert(all(
                    (~segm_mask[random_center[0] -
                                FWHM_ceil:random_center[0] +
                                FWHM_ceil, random_center[1] -
                                FWHM_ceil:random_center[1] + FWHM_ceil]).flat))
            # if verbose:  print imag_data[random_center[0] -
            # FWHM_ceil:random_center[0] + FWHM_ceil, random_center[1] -
            # FWHM_ceil:random_center[1] + FWHM_ceil]
            if all((~segm_mask[random_center[0] - FWHM_ceil:random_center[0] +
                               FWHM_ceil,
                               random_center[1] - FWHM_ceil:random_center[1] +
                               FWHM_ceil]).flat):
                if verbose:
                    print 'Aperture %i' % (n + 1)
                empty_ap.append(flux(random_center[0],
                                     random_center[1], ima_back,
                                     int(np.round(FWHM / 2))))
                if verbose:
                    print 'Flux = %f' % empty_ap[n]
                n += 1
            if verbose:
                print '--------------------------------'

        empty_ap_all.append(empty_ap)
        empty_ap_ccd.append(np.asarray(empty_ap))
        if verbose:
            print '______________________________________________________'

    empty_ap_all = np.asarray(empty_ap_all).flatten()
    # empty_ap_all = empty_ap_all[empty_ap_all > 0]
    print np.std(empty_ap_all)
    if field[5:7] == '13':
        # for 04
        if np.std(empty_ap_all) > 200:
            mask_sigma = sigma_clip(
                empty_ap_all, sigma=1.2, iters=1, cenfunc=np.median)
            empty_ap_all = empty_ap_all[~mask_sigma.mask]
        # for 01 and 06 and 15
        elif np.std(empty_ap_all) > 50.0 and np.std(empty_ap_all) < 200.0:
            mask_sigma = sigma_clip(
                empty_ap_all, sigma=1, iters=1, cenfunc=np.median)
            empty_ap_all = empty_ap_all[~mask_sigma.mask]
        # for 15 and 16
        elif np.std(empty_ap_all) > 25.0 and np.std(empty_ap_all) < 50.0:
            mask_sigma = sigma_clip(
                empty_ap_all, sigma=3, iters=1, cenfunc=np.median)
            empty_ap_all = empty_ap_all[~mask_sigma.mask]
    print 'Median flux from empty apertures \t%.2f ADU' % (np.median(empty_ap_all))
    print 'Mean flux from empty apertures \t\t%.2f ADU' % (np.mean(empty_ap_all))
    print 'STD flux from empty apertures \t\t%.2f ADU' % (np.std(empty_ap_all))

    errors = [val for sublist in errors for val in sublist]
    errors = np.asarray(errors)
    print '_________________________________________________________________________'
    print '_________________________________________________________________________'

    if plots:
        colors = cm.rainbow(np.linspace(0, 1, 20))
        name = '%s_%s' % (field, epoch)
        print '%s/%s/%s/%s_empty_ap.png' % (webpath, field, epoch, name)
        fig, ax = plt.subplots(2, figsize=(12, 9))
        fig.suptitle(name, fontsize=15)

        ax[0].hist(empty_ap_all, bins=500, log=False, color='g', alpha=0.5,
                   histtype='stepfilled', normed=False,
                   label='Empty aperture, std %.2f' % (np.std(empty_ap_all)))
        ax[0].axvline(np.median(empty_ap_all), color='r',
                      label='Median %.2f' % np.median(empty_ap_all))
        ax[0].axvline(np.mean(empty_ap_all), color='m',
                      label='Mean %.2f' % np.mean(empty_ap_all))
        ax[0].legend(loc='upper right', fontsize='x-small')
        ax[0].set_ylabel('N')
        ax[0].set_xlabel('Flux [ADU]')

        for col, ap, ccd in zip(colors, empty_ap_ccd[40:60], CHIPS[40:60]):
            ax[1].hist(ap, bins=50, log=True, color=col, alpha=0.7,
                       histtype='step', normed=False, label='%s' % (ccd))
            # ax[1].axvline(np.median(ap), color = 'r',
            #               label = 'Median %.2f' % np.median(ap))
            # ax[1].axvline(np.mean(ap), color = 'm',
            #               label = 'Mean %.2f' % np.mean(ap))
        ax[1].legend(fontsize='xx-small', ncol=3, loc='center left',
                     bbox_to_anchor=(1, 0.5))
        ax[1].set_ylabel('N')
        ax[1].set_xlabel('Flux [ADU]')
        ax[1].set_xlim(0, 1000)

        plt.savefig('%s/%s/%s/%s_empty_ap.png' % (webpath,  field,
                                                  epoch,  name),
                    dpi=300, format='png', bbox_inches='tight')

    return np.std(empty_ap_all)


def empty_aperture_one_CCD(
        field, CCD, epoch, n_aper=1000, plots=False, verbose=True):

    warnings.filterwarnings("ignore")

    imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (astropath,
                                                                 field, CCD,
                                                                 field, CCD,
                                                                 epoch)
    if not os.path.exists(imag_file):
        print 'No image file: %s' % (imag_file)
        sys.exit()
    hdufits_ima = fits.open(imag_file)
    imag_data = hdufits_ima[0].data
    if verbose:
        print 'SCIENCE IMAGE SHAPE \t\t', imag_data.shape

    backg_file = "%s/fits/%s/%s/CHECKIMAGE/%s_%s_%s_image_crblaster_background_thresh%s_minarea%i_backsize64.fits.fz" % \
        (jorgepath, field, CCD, field, CCD, epoch, str(thresh), minarea)
    if not os.path.exists(backg_file):
        print 'No image file: %s' % (backg_file)
        sys.exit()
    hdufits_bac = fits.open(backg_file)
    back_data = hdufits_bac[1].data
    if verbose:
        print 'SCIENCE IMAGE SHAPE \t\t', back_data.shape

    segm_file = "%s/fits/%s/%s/CHECKIMAGE/%s_%s_%s_image_crblaster_segmentation_thresh%s_minarea%i_backsize64.fits.fz" % (
        jorgepath, field, CCD, field, CCD, epoch, str(thresh), minarea)
    if not os.path.exists(segm_file):
        print 'No segmentation image file: %s' % (segm_file)
        sys.exit()
    hdufits_seg = fits.open(segm_file)
    segm_data = hdufits_seg[1].data
    segm_mask = (segm_data > 0)
    if verbose:
        print 'SEGMENTATION IMAGE SHAPE \t', segm_data.shape

    # aperture_set = np.arange(1,50, 2)
    aperture_set = np.linspace(1, 50, 21)
    sigma_ap, mean_ap, area_ap = [], [], []
    ima_back = imag_data - back_data
    empty_ap_all = []

    for aperture in aperture_set:
        aperture = int(aperture)
        n = 0
        empty_ap = []
        while n < n_aper:
            random_center = np.random.random_integers(
                0 + aperture, 2048 - aperture, 2) * (2, 1)
            if verbose:
                print 'Ramdom position ', random_center
            if verbose:
                print 'Object in patch?', np.invert(all(
                    (~segm_mask[random_center[0] - aperture:random_center[0] +
                                aperture,
                                random_center[1] - aperture:random_center[1] +
                                aperture]).flat))
            # if verbose:  print imag_data[random_center[0] -
            # aperture:random_center[0] + aperture, random_center[1] -
            # aperture:random_center[1] + aperture]
            if all((~segm_mask[random_center[0] - aperture:random_center[0] +
                               aperture,
                               random_center[1] - aperture:random_center[1] +
                               aperture]).flat):
                if verbose:
                    print 'Aperture %i' % (n + 1)
                empty_ap.append(
                    flux(
                        random_center[0], random_center[1], ima_back, int(
                            np.round(
                                aperture / 2))))
                if verbose:
                    print 'Flux = %f\tArea = %f' % (empty_ap[n][0],
                                                    empty_ap[n][1])
                n += 1
            if verbose:
                print '--------------------------------'

        empty_ap = np.array(empty_ap)
        empty_ap_all.append(empty_ap[:, 0])
        mean_ap.append(np.mean(empty_ap[:, 0]))
        sigma_ap.append(np.std(empty_ap[:, 0]))
        area_ap.append(np.median(empty_ap[:, 1]))
        if verbose:
            print '__________________________________________________'

    linear_ap = np.sqrt(area_ap)
    sigma_ap = np.array(sigma_ap)
    # check for nans in arrays...
    if any(np.isnan(sigma_ap)):
        mask = np.isnan(sigma_ap)
        sigma_ap = sigma_ap[~mask]
        linear_ap = linear_ap[~mask]
    fit_bol = False

    try:
        params, params_cov = curve_fit(function, linear_ap, sigma_ap)
        fit_bol = True
    except (RuntimeError):
        print('Optimal parameters not found:',
              'Number of calls to function has reached maxfev = 600.')
        sigma1 = sigma_ap[(linear_ap == 1)][0]
        alpha = 1.
        beta = 1.
        return sigma1, alpha, beta

    if fit_bol:
        sigma1 = sigma_ap[(linear_ap == 1)][0]
        alpha = params[0] / sigma1
        beta = params[1]
        print 'sigma1 = %f' % (sigma1)
        print 'alpha = %f' % (alpha)
        print 'beta = %f' % (beta)

    if plots:

        if not os.path.exists("%s/%s" % (webpath, field)):
            print "Creating field folder"
            os.makedirs("%s/%s" % (webpath, field))
        if not os.path.exists("%s/%s/%s" % (webpath, field, CCD)):
            print "Creating CCD folder"
            os.makedirs("%s/%s/%s" % (webpath, field, CCD))

        name = '%s_%s_%s' % (field, CCD, epoch)
        fig, ax = plt.subplots(ncols=2, figsize=(13, 6))

        ax[0].hist(empty_ap_all[1], bins=20, log=False, color='k', alpha=1,
                   lw=2, histtype='step', normed=False,
                   label='N = %.3f, std = %.2f' %
                   (linear_ap[1], np.std(empty_ap_all[1])))
        ax[0].hist(empty_ap_all[10], bins=20, log=False, color='g', alpha=1,
                   lw=2, histtype='step', normed=False,
                   label='N = %.3f, std = %.2f' % (linear_ap[10],
                                                   np.std(empty_ap_all[10])))
        ax[0].hist(empty_ap_all[20], bins=25, log=False, color='b', alpha=1,
                   lw=2, histtype='step', normed=False,
                   label='N = %.3f, std = %.2f' % (linear_ap[20],
                                                   np.std(empty_ap_all[20])))
        # ax[0].axvline(np.median(empty_ap_all), color='r',
        #               label='Median %.2f' % np.median(empty_ap_all))
        # ax[0].axvline(np.mean(empty_ap_all), color='b',
        #               label='Mean %.2f' % np.mean(empty_ap_all))
        ax[0].legend(loc='upper right', fontsize='xx-small')
        ax[0].set_ylabel('N')
        ax[0].set_ylim(0, 185)
        ax[0].set_xlabel('Flux [ADU]')

        ax[1].plot(linear_ap, sigma_ap, 'ko', label='Measurement')
        if fit_bol:
            ax[1].plot(linear_ap, sigma1 * alpha * linear_ap**beta,
                       'r--', lw=.8, label=r'$\alpha = %.3f, \beta = %.3f$' %
                       (alpha,  beta))
        if fit_bol:
            ax[1].plot(1, sigma1, 'ro',
                       label=r'$\sigma_{1} = %.3f$' % (sigma1))
        ax[1].plot(linear_ap, linear_ap, 'k--', label=r'$ \propto N$')
        ax[1].plot(linear_ap, linear_ap**2, 'k-.', label=r'$ \propto N^2$')
        ax[1].set_ylim(-20, 1000)
        ax[1].legend(loc='upper left', fontsize='x-small')
        ax[1].set_ylabel(r'$\sigma_{N}$')
        ax[1].set_xlabel(r'$N$')

        plt.savefig('%s/%s/%s/sky_sigma_%s.pdf' % (webpath,  field,
                                                   CCD,  name),
                    dpi=600, format='pdf', bbox_inches='tight')

    if False:

        np.save('%s/empty_all.npy' % (jorgepath), empty_ap_all)
        np.save('%s/linear_ap.npy' % (jorgepath), linear_ap)
        np.save('%s/sigma_ap.npy' % (jorgepath), sigma_ap)
        np.save('%s/fit_coef.npy' % (jorgepath), [sigma1, alpha, beta])

    # returning alpha*sigma_1 , beta
    if fit_bol:
        return sigma1, alpha, beta


def flux(pos_ver, pos_hor, img, radius):

    counts = 0.0
    n_pix = 0.
    # if verbose:  print 'Radius %i' % radius
    ver_max, hor_max = img.shape
    for ver in range(pos_ver - radius, pos_ver + radius + 1):
        for hor in range(pos_hor - radius, pos_hor + radius + 1):
            r2 = radius**2
            if (ver - pos_ver)**2 + (hor -
                                     pos_hor)**2 <= r2 and 0 <= ver < ver_max and 0 <= hor < hor_max:
                counts += img[ver][hor]
                n_pix += 1
    return counts, n_pix


def function(x, alpha, beta):
    return alpha * x**(beta)


if __name__ == '__main__':
    field = sys.argv[1]
    CCD = sys.argv[2]
    epoch = sys.argv[3]
    nap = sys.argv[4]
    results = empty_aperture_one_CCD(
        field, CCD, epoch, n_aper=int(nap), verbose=True)
    print results
