#!/usr/bin/python

import sys
import os
import re
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
# from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree
from astropy.io import fits, ascii
from misc_func_leftraru import *
from astropy.table import Table, vstack
# from threading import Thread
import seaborn as sb
from astropy.stats import SigmaClip, sigma_clip, sigma_clipped_stats
from photutils import (Background2D, MedianBackground, find_peaks,
                       create_matching_kernel, SplitCosineBellWindow,
                       TopHatWindow, CosineBellWindow)
from photutils.utils import filter_data, calc_total_error
from photutils import aperture_photometry, CircularAperture
from kernel import *

matplotlib.use('Agg')
sb.set(style="white", color_codes=True, context="notebook", font_scale=1.4)

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'

thresh = 1.0      # threshold of catalog
minarea = 3      # min area for detection
deg2rad = 0.0174532925
rad2deg = 57.2957795
axisY = 4094
axisX = 2046
dx = 50
dx_stamp = 25

# match files: aflux | e_aflux | rms | order | sol_astrometry
# matchRADEC: afluxADUB | e_afluxADUB | rmsdeg | CRVAL1 | CRVAL2 | CRPIX1 |
#             CRPIX2 | CD11 | CD12 | CD21 | CD22 | nPV1 | nPV2 | order |
#             sol_astrometry_RADEC | PV(flatten)

# in fits header: shape = (X, Y) = (2048, 4096)
# importing image with pyfits: shape = (Y, X) = (row, col) = (4096, 2048)
# SE catalogue: shape = (X, Y) = (2048, 4096)

def preprocess_lc(lc, min_mag=15., mag_key=None):
    # time, mag, error = np.array(lc_g['MJD'].quantity), \
    #                   np.array(lc_g['MAG_KRON'].quantity),\
    #                   np.array(lc_g['MAGERR_KRON'].quantity)
    # remove saturated points
    print 'Original size: %i |' % len(lc),
    if mag_key:
        lc = lc[lc[mag_key] >= min_mag]
        filtered_data = sigma_clip(lc[mag_key], sigma=3,
                                   iters=1, cenfunc=np.median,
                                   copy=False)
    else:
        lc = lc[lc[:, 1] >= min_mag]
        filtered_data = sigma_clip(lc[:, 1], sigma=4,
                                   iters=1, cenfunc=np.median,
                                   copy=False)
    lc = lc[~filtered_data.mask]
    print ' clipped: %i |' % np.sum(filtered_data.mask),
    print 'Final size: %i' % len(lc)

    return lc


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def get_filter_nearest_stars(cata, xpix, ypix, nn=10):

    print '\tN original cata  : ', len(cata)
    # print cata.to_pandas()[['CLASS_STAR', 'FLAGS', 'ELONGATION',
    #                         'MAG_AUTO_ZP', 'FWHM_IMAGE']].describe()
    quality_mask = (cata['CLASS_STAR'] >= .9) & (cata['FLAGS'] == 0) & \
                   (1-1/cata['ELONGATION'] <= .5) & \
                   (cata['FWHM_IMAGE'] <= np.median(cata['FWHM_IMAGE']) +
                    0.5 * np.std(cata['FWHM_IMAGE'])) & \
                   (cata['MAG_AUTO_ZP'] >= 16.5) & \
                   (cata['MAG_AUTO_ZP'] <= 20) & \
                   (cata['X_IMAGE'] > dx) & (cata['X_IMAGE'] < axisX - dx) & \
                   (cata['Y_IMAGE'] > dx) & (cata['Y_IMAGE'] < axisY - dx)
    cata_gq = cata[quality_mask]
    print '\tN clean cata     : ', len(cata_gq)
    if len(cata_gq) == 0:
        print '# WARNING: No stars near the postion after cleaninng...'
        return None

    XY = np.transpose(np.array((cata_gq['X_IMAGE'], cata_gq['Y_IMAGE'])))
    tree_XY = cKDTree(XY)

    XY_obj = np.transpose(np.array((xpix, ypix)))
    dist, indx = tree_XY.query(XY_obj, k=nn, distance_upper_bound=1024)
    inf_mask = ~np.isinf(dist)
    nearest = cata_gq[indx[inf_mask]]

    print '\tN nearest objects: ', len(nearest)
    if len(nearest) == 0:
        print '# WARNING: No stars near the postion after cleaninng...'
        return None
    print '\tMean FWHM        : ', np.mean(nearest['FWHM_IMAGE']) * 0.27

    return nearest


def get_stars(image, stars_cata, scale=1.5, plot=True):

    print len(stars_cata)
    hal_size = int(np.max(stars_cata['FWHM_IMAGE']) * scale)
    hal_size = 23
    print hal_size
    stamps = []

    for k in range(len(stars_cata)):
        print '\tStar position: %f, %f' % (stars_cata['X_IMAGE'][k],
                                           stars_cata['Y_IMAGE'][k])
        print '\tStar size    : ', stars_cata['FWHM_IMAGE'][k]

        col_pix = int(stars_cata['X_IMAGE'][k])
        row_pix = int(stars_cata['Y_IMAGE'][k])

        stamp = image[row_pix - hal_size: row_pix + hal_size,
                      col_pix - hal_size: col_pix + hal_size].copy()
        stamps.append(stamp)

    stamps = np.array(stamps)
    print stamps.shape

    # out_stamps = open('/home/jmartinez/HiTS/temp/%s_%s_%s_stars.npy' %
    #                   (field, ccd, epoch), 'w')
    # pickle.dump(stamps, out_stamps)
    # out_stamps.close()

    if plot:
        plt.imshow(kernel, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return stamps


def get_kernel(image, stars_cata, plot=False):

    print len(stars_cata)
    hal_size = 10
    print hal_size
    stamps = []
    for k in range(len(stars_cata)):
        print '\tStar position: %f, %f' % (stars_cata['X_IMAGE'][k],
                                           stars_cata['Y_IMAGE'][k])
        print '\tStar size    : ', stars_cata['FWHM_IMAGE'][k]

        col_pix = int(stars_cata['X_IMAGE'][k])
        row_pix = int(stars_cata['Y_IMAGE'][k])

        stamp = image[row_pix - hal_size: row_pix + hal_size + 1,
                      col_pix - hal_size: col_pix + hal_size + 1].copy()
        stamp_bkg = image[row_pix - hal_size * 2: row_pix + hal_size * 2,
                          col_pix - hal_size * 2: col_pix + hal_size * 2].copy()

        # background
        sigma_clip = SigmaClip(sigma=3., iters=2)
        bkg_estimator = MedianBackground()
        bkg = Background2D(stamp_bkg, (hal_size * 2, hal_size * 2),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        stamp = stamp - bkg.background[hal_size: hal_size * 3 + 1,
                                       hal_size: hal_size * 3 + 1]
        # stamp[stamp < 0] = 0
        stamp /= stamp.sum()
        print '\tStar sum    :', stamp.sum()

        # plt.imshow(norm_stamp, cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        stamps.append(stamp)
    stamps = np.array(stamps)

    kernel = np.median(stamps, axis=0)
    kernel /= kernel.sum()
    print '\tPSF sum     :', kernel.sum()
    if plot:
        plt.imshow(kernel, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return kernel

# function to convert fluxes into magnitudes given fluxes and errors in
# ADU, the CCD number, the exposure time and the airmass of the
# observation

def ADU2mag(flux, e_flux, CCD, exptime, airmass, azero, kzero, err_Kg, err_Ag):

    mag = np.array(-2.5 * np.log10(flux) +
                   2.5 * np.log10(exptime) - azero[CCDn[CCD] - 1] -
                   kzero[CCDn[CCD] - 1] * airmass)
    e_mag = np.sqrt(((2.5 * e_flux) / (flux * np.log(10))) ** 2 +
                    err_Ag[CCDn[CCD]] ** 2 +
                    (airmass * err_Kg[CCDn[CCD]]) ** 2)
    return (mag, e_mag)


def ADU2mag_PS(flux, e_flux, exptime, ZP, e_ZP):

    mag = np.array(-2.5 * np.log10(flux) +
                   2.5 * np.log10(exptime) + ZP)
    e_mag = np.sqrt(((2.5 * e_flux) / (flux * np.log(10))) ** 2 +
                    e_ZP)
    return (mag, e_mag)


def get_photometry(image, mask=None, gain=4., pos=(dx_stamp, dx_stamp),
                   radii=10.):

    sigma_clip = SigmaClip(sigma=3., iters=2)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (10, 10), sigma_clip=sigma_clip,
                       bkg_estimator=bkg_estimator)
    print '\tBackground stats: %f, %f' % (bkg.background_median,
                                          bkg.background_rms_median)

    data = image - bkg.background
    # bkg_err = np.random.normal(bkg.background_median,
    #                            bkg.background_rms_median, image.shape)
    if False:
        plt.imshow(data, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()
    # error = calc_total_error(image, bkg_err, gain)
    back_mean, back_median, back_std = sigma_clipped_stats(data, mask, sigma=3,
                                                           iters=3,
                                                           cenfunc=np.median)
    print '\tBackground stats: %f, %f' % (back_median, back_std)
    tbl = find_peaks(data, np.minimum(back_std, bkg.background_rms_median) * 3,
                     box_size=5, subpixel=True)
    print tbl
    tree_XY = cKDTree(np.array([tbl['x_peak'], tbl['y_peak']]).T)
    dist, indx = tree_XY.query(pos, k=1, distance_upper_bound=5)
    if np.isinf(dist):
        print '\tNo source found in the asked position...'
        return None
    position = [tbl[indx]['x_centroid'], tbl[indx]['y_centroid']]
    print '\tObject position: ', position

    apertures = [CircularAperture(position, r=r) for r in radii]
    phot_table = aperture_photometry(data, apertures, mask=mask,
                                     method='subpixel', subpixels=5)
    for k, r in enumerate(radii):
        area = np.pi * r ** 2
        phot_table['aperture_flx_err_%i' %
                   k] = np.sqrt(area * bkg.background_rms_median ** 2 +
                                phot_table['aperture_sum_%i' % k] / gain)
    phot_table.remove_columns(['xcenter', 'ycenter'])
    phot_table['xcenter'] = position[0]
    phot_table['ycenter'] = position[1]
    return phot_table


def get_stamps_lc(field, CCD, FILTER, row_pix, col_pix, verbose=False):

    name = '%s_%s_%s_%s_%s' % (field, CCD, col_pix, row_pix, FILTER)
    print 'Saving stamps for... ', name
    epochs = np.loadtxt('%s/info/%s/%s_epochs_%s.txt' %
                        (jorgepath, field, field, FILTER),
                        dtype={'names': ('EPOCH', 'MJD'),
                               'formats': ('S2', 'f4')}, comments='#')

    stamps_lc, time_lc, centroid_lc = [], [], []

    for ll, epoch in enumerate(epochs):
        if verbose:
            print 'Working in epoch %s...' % (epoch[0])
        # fits
        imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (astropath,
                                                                     field,
                                                                     CCD,
                                                                     field,
                                                                     CCD,
                                                                     epoch[0])
        if not os.path.exists(imag_file):
            if verbose:
                print '\t\tNo image file: %s' % (imag_file)
            continue
        hdu = fits.open(imag_file)
        data = hdu[0].data

        # loading catalogs
        cat_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
            (jorgepath, field, CCD, field, CCD,
             epoch[0], str(thresh), str(minarea))
        if not os.path.exists(cat_file):
            if verbose:
                print '\t\tNo catalog file for worst epoch: %s' % (cat_file)
            continue
        cata = Table.read(cat_file, format='ascii')
        cata_XY = np.transpose(np.array((cata['X_IMAGE_REF'],
                                         cata['Y_IMAGE_REF'])))
        tree_XY = cKDTree(cata_XY)

        # quering asked position
        XY_obj = np.transpose(np.array((col_pix, row_pix)))
        dist, indx = tree_XY.query(XY_obj, k=1, distance_upper_bound=5)
        if np.isinf(dist):
            if verbose:
                print '\t\tNo match in epoch %s' % epoch[0]
                continue

        # position in non projected coordinates, i.e. loaded image
        row_pix2 = int(np.around(cata['Y_IMAGE'][indx]))
        col_pix2 = int(np.around(cata['X_IMAGE'][indx]))
        xx = cata['X_IMAGE'][indx] - col_pix2 + 10
        yy = cata['Y_IMAGE'][indx] - row_pix2 + 10
        print col_pix2, row_pix2
        print xx, yy
        stamp = data[row_pix2 - 10 - 1: row_pix2 + 10,
                     col_pix2 - 10 - 1: col_pix2 + 10]
        if False:
            plt.imshow(stamp, cmap='viridis', interpolation='nearest')
            plt.axvline(xx)
            plt.axhline(yy)
            plt.show()
        stamps_lc.append(stamp)
        time_lc.append(epoch)
        centroid_lc.append([xx, yy])

    stamps_lc = np.array(stamps_lc)
    time_lc = np.array(time_lc)
    centroid_lc = np.array(centroid_lc)
    print stamps_lc.shape, time_lc.shape
    to_save = {}
    to_save['stamp'] = stamps_lc
    to_save['time'] = time_lc
    to_save['centroidsXY'] = centroid_lc
    np.save('%s/lightcurves/stamps/%s_stamp.npy' % (jorgepath, name), to_save)
    if False:
        fig, ax = plt.subplots(ncols=len(stamps_lc), nrows=1,
                               figsize=(1.6 * len(stamps_lc), 2))
        for i in range(len(stamps_lc)):
            ax[i].imshow(stamps_lc[i], interpolation="nearest",
                         cmap='gray', origin='lower')
            ax[i].text(1, 1, "%8.2f" % time_lc[i]['MJD'],
                       fontsize=14, color='orange')
            ax[i].axes.get_xaxis().set_visible(False)
            ax[i].axes.get_yaxis().set_visible(False)
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.savefig('%s/lightcurves/stamps/%s_stamp.png' % (jorgepath, name),
                    tight_layout=True, pad_inches=0.01, facecolor='black',
                    bbox_inches='tight')
        plt.close(fig)

    print 'Done!'
    return


def run_lc_psf(field, CCD, FILTER, row_pix, col_pix, seeing_limit=1.5,
               verbose=False):
    name = '%s_%s_%04i_%04i_%s' % (field, CCD, col_pix, row_pix, FILTER)
    epochs = np.loadtxt('%s/info/%s/%s_epochs_%s.txt' %
                        (jorgepath, field, field, FILTER),
                        dtype={'names': ('EPOCH', 'MJD'),
                               'formats': ('S2', 'f4')}, comments='#')

    images, airmass, seeing, dqmask, gain, exptime = [], [], [], [], [], []

    print 'Loading catalogues...'
    print '\tEpoch |      MJD     |  SEEING  | AIRMASS'
    for ll, epoch in enumerate(epochs):
        # fits
        imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (astropath,
                                                                     field,
                                                                     CCD,
                                                                     field,
                                                                     CCD,
                                                                     epoch[0])
        if not os.path.exists(imag_file):
            if verbose:
                print '\t\tNo image file: %s' % (imag_file)
            continue
        hdu = fits.open(imag_file)
        seeing.append(hdu[0].header['FWHM'] * hdu[0].header['PIXSCAL1'])
        airmass.append(hdu[0].header['AIRMASS'])
        gain.append(hdu[0].header['GAINA'])
        exptime.append(hdu[0].header['EXPTIME'])
        data = hdu[0].data
        images.append(data)
        if verbose:
            print '\t   %s | %f | %f | %f' % (epoch[0], epoch[1],
                                              seeing[-1], airmass[-1])

        dqm_file = "%s/DATA/%s/%s/%s_%s_%s_dqmask.fits.fz" % (astropath, field,
                                                              CCD, field, CCD,
                                                              epoch[0])
        if not os.path.exists(dqm_file):
            if verbose:
                print '\t\tNo dqmask file: %s' % (dqm_file)
                dqmask.append(False)
        else:
            hdu = fits.open(dqm_file)
            dqmask.append(hdu[0].data)

    # clean bad observation conditions
    seeing = np.array(seeing)
    airmass = np.array(airmass)
    images = np.array(images)
    dqmask = np.array(dqmask)
    # mask_good = (seeing <= seeing_limit)
    mask_good = (epochs['MJD'] <= 57073.)
    if verbose:
        print 'Total number of images     : ', len(images)
        print 'Epochs with good conditions: ', len(images[mask_good])

    epochs = epochs[mask_good]
    images = images[mask_good]
    seeing = seeing[mask_good]
    airmass = airmass[mask_good]
    dqmask = dqmask[mask_good]

    # selecting worst epoch
    idx_worst = np.argmax(seeing)
    if verbose:
        print 'Worst epoch  : ', epochs[idx_worst]
        print 'Worst seeing : ', seeing[idx_worst]
        print 'Worst airmass: ', airmass[idx_worst]

    # select 100 stars closer to
    if verbose:
        print 'Searching for nearest & cleanest stars near position...'
    ref_cat_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
        (jorgepath, field, CCD, field, CCD,
         epochs[idx_worst][0], str(thresh), str(minarea))
    if not os.path.exists(ref_cat_file):
        if verbose:
            print '\t\tNo catalog file for worst epoch: %s' % (ref_cat_file)
        return

    ref_cata = Table.read(ref_cat_file, format='ascii')
    ref_tree_XY = cKDTree(np.transpose(np.array((ref_cata['X_IMAGE_REF'],
                                                 ref_cata['Y_IMAGE_REF']))))
    # quering asked position in worst image
    XY_worst = np.transpose(np.array((col_pix, row_pix)))
    dist_w, indx_w = ref_tree_XY.query(XY_worst, k=1, distance_upper_bound=5)
    print dist_w, indx_w
    row_pix_w = int(np.around(ref_cata['Y_IMAGE'][indx_w]))
    col_pix_w = int(np.around(ref_cata['X_IMAGE'][indx_w]))
    print col_pix_w, row_pix_w
    stamp_worst = images[idx_worst][row_pix_w - dx_stamp:
                                    row_pix_w + dx_stamp + 1,
                                    col_pix_w - dx_stamp:
                                    col_pix_w + dx_stamp + 1].copy()

    nearest_cata = get_filter_nearest_stars(ref_cata, col_pix_w, row_pix_w)
    if nearest_cata is None:
        return
    if verbose:
        print 'Creating kernel...'
    psf_worst = get_kernel(images[idx_worst], nearest_cata)
    # psf_worst1 = get_kernel(images[idx_worst],
    #                         nearest_cata[:int(len(nearest_cata)/2)])
    # psf_worst2 = get_kernel(images[idx_worst],
    #                         nearest_cata[int(len(nearest_cata)/2):])
    if verbose:
        print 'Kernel done!'
        print 'Going into time serie...'

    # aperture radii for photometry
    ap_radii = seeing[idx_worst] / 0.27 * np.array([0.5, 0.75, 1., 1.25, 1.5])
    print 'Aperture radii: ', ap_radii
    lc_yes, lc_no = [], []
    stamps_lc = []
    airmass2, exptime2, ZP = [], [], []

    for k, epoch in enumerate(epochs):

        if verbose:
            print 'Working in epoch %s...' % (epoch[0])

        # loading catalogs
        cat_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
            (jorgepath, field, CCD, field, CCD,
             epoch[0], str(thresh), str(minarea))
        if not os.path.exists(cat_file):
            if verbose:
                print '\t\tNo catalog file for worst epoch: %s' % (cat_file)
            continue
        cata = Table.read(cat_file, format='ascii')
        cata_XY = np.transpose(np.array((cata['X_IMAGE_REF'],
                                         cata['Y_IMAGE_REF'])))
        tree_XY = cKDTree(cata_XY)

        # quering asked position
        XY_obj = np.transpose(np.array((col_pix, row_pix)))
        dist, indx = tree_XY.query(XY_obj, k=1, distance_upper_bound=5)
        if np.isinf(dist):
            if verbose:
                print '\t\tNo match in epoch %s' % epoch[0]
                lc_no.append(epoch)
                continue

        # position in non projected coordinates, i.e. loaded image
        row_pix2 = int(np.around(cata['Y_IMAGE'][indx]))
        col_pix2 = int(np.around(cata['X_IMAGE'][indx]))
        err_mag = cata['MAGERR_AUTO_ZP'][indx]
        print row_pix2, col_pix2, err_mag
        stamp = images[k][row_pix2 - dx_stamp: row_pix2 + dx_stamp + 1,
                          col_pix2 - dx_stamp: col_pix2 + dx_stamp + 1].copy()
        stamp_mask = dqmask[k][row_pix2 - dx_stamp:
                               row_pix2 + dx_stamp + 1,
                               col_pix2 - dx_stamp:
                               col_pix2 + dx_stamp + 1].copy()

        if epoch[0] != epochs[idx_worst][0]:
            nearest_cata2 = get_filter_nearest_stars(cata, col_pix2, row_pix2)
            if nearest_cata2 is None:
                continue
            psf = get_kernel(images[k], nearest_cata2)
            part = 2
            matched_kernel = []
            for p in range(part):
                aux_psf = get_kernel(images[k],
                                     nearest_cata2[int(p * len(nearest_cata2) /
                                                       part):
                                                   int((p+1) *
                                                       len(nearest_cata2) /
                                                       part)])

                aux_kernel = create_matching_kernel(aux_psf, psf_worst,
                                                    window=TopHatWindow(0.62))
                matched_kernel.append(aux_kernel)
                # aux_kernel2 = create_matching_kernel(aux_psf, psf_worst2,
                #                                      window=TopHatWindow(0.7))
                # matched_kernel.append(aux_kernel2)

            matched_kernel = np.array(matched_kernel)
            print matched_kernel.shape
            matched_kernel = np.median(matched_kernel, axis=0)
            matched_kernel /= matched_kernel.sum()
            print '\tMatching kernel sum: ', matched_kernel.sum()
            convolved_kernel = filter_data(psf, matched_kernel,
                                           mode='nearest')

            convolved_stamp = filter_data(stamp, matched_kernel,
                                          mode='nearest')
        else:
            print '\tWorst epoch'
            matched_kernel = psf_worst.copy()
            stamp = stamp_worst.copy()
            convolved_stamp = stamp_worst.copy()
        stamps_lc.append([stamp, convolved_stamp, matched_kernel,
                          float(epoch[1]), convolved_kernel])

        data_point = get_photometry(convolved_stamp, mask=stamp_mask,
                                    gain=gain[k], pos=(dx_stamp, dx_stamp),
                                    radii=ap_radii)
        if data_point is None:
            print '\t\tNo match in epoch %s' % epoch[0]
            lc_no.append(epoch)
            continue
        data_point['mjd'] = float(epoch[1])
        data_point['epoch'] = epoch[0]
        data_point['aperture_mag_err_0_cat'] = err_mag

        lc_yes.append(data_point)
        airmass2.append(airmass[k])
        exptime2.append(exptime[k])

        ZP_PS = np.load('%s/info/%s/%s/ZP_%s_PS_%s_%s_%s.npy' %
                        (jorgepath, field, CCD, 'AUTO', field, CCD, epoch[0]))
        ZP.append([ZP_PS[0][0], ZP_PS[2][0]])

        if False:

            fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(18, 10))

            im2 = ax[0, 0].imshow(stamp,
                                  cmap='viridis', interpolation='nearest')
            ax[0, 0].set_title(epoch[0])
            fig.colorbar(im2, ax=ax[0, 0])

            im1 = ax[0, 1].imshow(stamp_worst,
                                  cmap='viridis', interpolation='nearest')
            ax[0, 1].set_title('Worst')
            fig.colorbar(im1, ax=ax[0, 1])

            im3 = ax[0, 2].imshow(convolved_stamp,
                                  cmap='viridis', interpolation='nearest')
            ax[0, 2].set_title('%s convolved' % epoch[0])
            for k in ap_radii:
                circle = plt.Circle([data_point['xcenter'][0],
                                     data_point['ycenter'][0]],
                                    k, color='r', fill=False)
                ax[0, 2].add_artist(circle)
            fig.colorbar(im3, ax=ax[0, 2])

            im4 = ax[0, 3].imshow(psf_worst - convolved_kernel,
                                  cmap='viridis', interpolation='nearest')
            ax[0, 3].set_title('kernel sustraction')
            fig.colorbar(im4, ax=ax[0, 3])

            im2 = ax[1, 0].imshow(psf, cmap='viridis',
                                  interpolation='nearest')
            ax[1, 0].set_title('%s' % (epoch[0]))
            fig.colorbar(im2, ax=ax[1, 0])

            im1 = ax[1, 1].imshow(psf_worst, cmap='viridis',
                                  interpolation='nearest')
            ax[1, 1].set_title('Worst')
            fig.colorbar(im1, ax=ax[1, 1])

            im3 = ax[1, 2].imshow(matched_kernel, cmap='viridis',
                                  interpolation='nearest')
            ax[1, 2].set_title('matching kernel')
            fig.colorbar(im3, ax=ax[1, 2])

            im4 = ax[1, 3].imshow(convolved_kernel, cmap='viridis',
                                  interpolation='nearest')
            ax[1, 3].set_title('convolved %s' % (epoch[0]))
            fig.colorbar(im4, ax=ax[1, 3])
            fig.tight_layout()
            # plt.savefig('%s/lightcurves/galaxy/%s/%s_%s_psf.png' %
            #             (jorgepath, field, name, epoch[0]),
            #             tight_layout=True, pad_inches=0.01,
            #             bbox_inches='tight')
            plt.show()

        # if k == 3: break

    if len(lc_yes) <= 10:
        print 'No LC for this source...'
        return None
    lc = vstack(lc_yes)
    # lc_raw = vstack(lc_raw)
    airmass2 = np.array(airmass2)
    exptime2 = np.array(exptime2)
    ZP = np.array(ZP)

    # magnitudes
    for k in range(len(ap_radii)):
        (mags, e1_mags) = ADU2mag_PS(lc['aperture_sum_%i' % k],
                                     lc['aperture_flx_err_%i' % k],
                                     exptime2, ZP[:, 0], ZP[:, 1])
        lc['aperture_mag_%i' % k] = mags
        lc['aperture_mag_err_%i' % k] = e1_mags

    lc_df = lc.to_pandas()
    lc_df.drop('id', axis=1, inplace=True)
    for k in range(len(ap_radii)):
        lc_df.rename(columns={'aperture_sum_%i' % k: 'aperture_flx_%i' % k},
                     inplace=True)
    print lc_df.columns.values
    print lc_df.shape
    f = open('%s/lightcurves/galaxy/%s/%s_psf.csv' %
                 (jorgepath, field, name), 'w')
    f.write('# Worst epoch    : %s\n' % epochs[idx_worst][0])
    f.write('# Worst MJD      : %s\n' % epochs[idx_worst][1])
    f.write('# Worst seeing   : %s\n' % seeing[idx_worst])
    f.write('# Aperture radii : seeing * [0.5, 0.75, 1., 1.25, 1.5]\n')
    f.write('# Aperture radii : %s\n' % str(ap_radii))
    lc_df.to_csv(f)
    f.close()
    # print lc_df[['aperture_flx_0', 'aperture_flx_err_0',
    #              'aperture_mag_0', 'aperture_mag_err_0',
    #              'aperture_mag_0_cat']]

    if True:
        brightest_idx = np.argmax(lc_df.aperture_flx_0)
        cmin = np.percentile(stamps_lc[brightest_idx][1].flatten(), 50)
        cmax = stamps_lc[brightest_idx][1].flatten().max()

        fig, ax = plt.subplots(ncols=len(stamps_lc), nrows=4,
                               figsize=(2. * len(stamps_lc), 10))
        for i in range(len(stamps_lc)):
            ax[0, i].imshow(stamp_worst, interpolation="nearest",
                            cmap='gray', origin='lower')
            ax[1, i].imshow(stamps_lc[i][0], interpolation="nearest",
                            cmap='gray', origin='lower')
            ax[2, i].imshow(stamps_lc[i][1], interpolation="nearest",
                            cmap='gray', origin='lower', clim=(cmin, cmax))
            circle = plt.Circle([lc_df['xcenter'][i],
                                 lc_df['ycenter'][i]],
                                ap_radii[0], color='r', lw=.5, fill=False)
            ax[2, i].add_artist(circle)
            circle = plt.Circle([lc_df['xcenter'][i],
                                 lc_df['ycenter'][i]],
                                ap_radii[1], color='r', lw=.5, fill=False)
            ax[2, i].add_artist(circle)
            cmin_k = np.percentile(stamps_lc[i][2].flatten(), 50)
            cmax_k = stamps_lc[i][2].flatten().max()
            ax[3, i].imshow(stamps_lc[i][2], interpolation="nearest",
                            cmap='gray', origin='lower', clim=(cmin_k, cmax_k))
            ax[0, i].text(1, 1, "%8.2f" % epochs[idx_worst][1],
                          fontsize=14, color='orange')
            ax[1, i].text(1, 1, "%8.2f" % stamps_lc[i][3],
                          fontsize=14, color='orange')
            for j in range(4):
                ax[j, i].axes.get_xaxis().set_visible(False)
                ax[j, i].axes.get_yaxis().set_visible(False)

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.savefig('%s/lightcurves/galaxy/%s/%s_psf_series.png' %
                    (jorgepath, field, name),
                    tight_layout=True, pad_inches=0.01, facecolor='black',
                    bbox_inches='tight')
        # plt.show()
        plt.close(fig)

    if False:
        plt.errorbar(lc_df['mjd'], lc_df['aperture_mag_0'],
                     yerr=lc_df['aperture_mag_err_0'], lw=0, label='psf',
                     elinewidth=1, c='r', marker='.', markersize=15)
        plt.xlabel('mjd')
        plt.ylabel('g flux')
        plt.legend(loc='best')
        plt.show()


def run_lc_catalog(field, CCD, FILTER, q_row_pix, q_col_pix, verbose=False):

    epochs = np.loadtxt('%s/info/%s/%s_epochs_%s.txt' %
                        (jorgepath, field, field, FILTER),
                        dtype={'names': ('EPOCH', 'MJD'),
                               'formats': ('S2', 'f4')}, comments='#')

    name = '%s_%s_%s_%s_%s' % (field, CCD, FILTER, q_col_pix, q_row_pix)
    print name
    q_row_pix = int(q_row_pix)
    q_col_pix = int(q_col_pix)

    time_series = []
    time_series_epoch = []
    images, gif_im, regions, yesyes = [], [], [], []

    print 'Loading catalogues...'
    for ll, epoch in enumerate(epochs):
        if verbose:
            print '\tEpoch %s' % epoch[0]

        # INFO epoch file

        # INFO_file = '%s/info/%s/%s_%s_%s.npy' % (jorgepath, field, field,
        #                                     epoch[0], FILTER)
        # if not os.path.exists(INFO_file):
        #     if verbose:
        #         print '\t\tNo file: %s' % (INFO_file)
        #     continue
        # INFO = np.load(INFO_file)

        # catalogues

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
                    (jorgepath, field, CCD, field, CCD,
                     epoch[0], str(thresh), str(minarea))
        if not os.path.exists(cata_file):
            if verbose:
                print '\t\tNo catalog file: %s' % (cata_file)
            continue
        cata = Table.read(cata_file, format='ascii')
        cata_XY = np.transpose(np.array((cata['X_IMAGE_REF'],
                                         cata['Y_IMAGE_REF'])))
        tree_XY = cKDTree(cata_XY)

        XY_obj = np.transpose(np.array((q_col_pix, q_row_pix)))
        indx = tree_XY.query(XY_obj, k=1, distance_upper_bound=5)[1]
        if indx == len(cata):
            if verbose:
                print '\t\tNo match in epoch %s' % epoch[0]
            # time_series_no.append(INFO)
            regions.append([24.5, 24.5, 0, 0, 0])
            yes_match = False
            continue
        else:
            print indx, cata['Y_IMAGE_REF'][indx], cata['X_IMAGE_REF'][indx],
            print cata['Y_IMAGE'][indx], cata['X_IMAGE'][indx]
            yes_match = True
            try:
                time_series.append([cata['MAG_AUTO_ZP'][indx],
                                    cata['MAGERR_AUTO_ZP'][indx],
                                    cata['FLUX_AUTO_AFLUX'][indx],
                                    cata['FLUXERR_AUTO_COR'][indx]])
            except KeyError:
                time_series.append([cata['MAG_AUTO_ZA'][indx],
                                    cata['MAGERR_AUTO_ZA'][indx],
                                    cata['FLUX_AUTO_AFLUX'][indx],
                                    cata['FLUXERR_AUTO_COR'][indx]])
            yesyes.append(ll)
            time_series_epoch.append(epoch)
            row_pix, col_pix = cata['Y_IMAGE'][indx], cata['X_IMAGE'][indx]

        # fits
        imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (astropath,
                                                                     field,
                                                                     CCD,
                                                                     field,
                                                                     CCD,
                                                                     epoch[0])
        if not os.path.exists(imag_file):
            if verbose:
                print '\t\tNo image file: %s' % (imag_file)
            continue
        hdufits = fits.open(imag_file)
        data = hdufits[0].data
        col_pix = int(col_pix)
        row_pix = int(row_pix)
        delta = int(25)
        delta2 = int(75)
        if yes_match:
            if cata['KRON_RADIUS'][indx] * cata['A_IMAGE'][indx] > 25:
                delta = int(cata['KRON_RADIUS'][indx] * cata['A_IMAGE'][indx])
                delta2 = int(cata['KRON_RADIUS'][indx] *
                             cata['A_IMAGE'][indx]) * 2

            if col_pix < delta2 or row_pix < delta2:
                delta2 = int(np.min([row_pix, col_pix]))

            images.append(data[row_pix - delta:row_pix +
                               delta, col_pix - delta:col_pix + delta])
            gif_im.append(data[row_pix - delta2:row_pix +
                               delta2, col_pix - delta2:col_pix + delta2])

            regions.append([delta, delta,
                            cata['KRON_RADIUS'][indx] * cata['A_IMAGE'][indx],
                            cata['KRON_RADIUS'][indx] * cata['B_IMAGE'][indx],
                            cata['THETA_IMAGE'][indx]])

        if verbose:
            print '_________________________________________________________'
    time_series = np.asarray(time_series)
    regions = np.asarray(regions)
    time_series_epoch = np.asarray(time_series_epoch)

    # sort by mjd
    idx_sort = np.argsort(time_series_epoch['MJD'])
    time_series_epoch = time_series_epoch[idx_sort]
    time_series = time_series[idx_sort]
    regions = regions[idx_sort]

    if verbose:
        print len(time_series)
        print len(time_series_epoch)
        print len(images)
        print len(gif_im)

    if True:
        if verbose:
            print 'Removing previous temporal files...'
        os.system('rm %s/figures/temp/gif_%s_*.png' % (jorgepath, name))

    figure_path = '%s/figures' % (jorgepath)

    if not os.path.exists("%s/%s" % (figure_path, field)):
        if verbose:
            print "Creating field folder"
        os.makedirs("%s/%s" % (figure_path, field))
    if not os.path.exists("%s/%s/%s" % (figure_path, field, CCD)):
        if verbose:
            print "Creating CCD folder"
        os.makedirs("%s/%s/%s" % (figure_path, field, CCD))

    # saving lc data
    if True:
        print 'Saving lc file...'
        to_save = np.rec.fromarrays((time_series_epoch['MJD'],
                                     time_series_epoch['EPOCH'],
                                     time_series[:, 0],
                                     time_series[:, 1],
                                     time_series[:, 2],
                                     time_series[:, 3]),
                                    dtype=[('MJD', float), ('EPOCH', float),
                                           ('mag', float),
                                           ('e_mag', float), ('ADU', float),
                                           ('e_ADU', float)])
        print to_save.shape
        np.save('%s/%s/%s/%s_data.npy' %
                (figure_path, field, CCD, name), to_save)

    if True:
        print 'Ploting LC'
        fig, ax = plt.subplots(2, figsize=(12, 9))
        ax[0].errorbar(time_series_epoch['MJD'] -
                       np.min(time_series_epoch['MJD']),
                       time_series[:, 0], yerr=time_series[:, 1],
                       fmt='o', color='b', label='DETECT_MAG', alpha=.7)
        ax[1].errorbar(time_series_epoch['EPOCH'].astype(int),
                       time_series[:, 0], yerr=time_series[:, 1],
                       fmt='o', color='b',
                       label='DETECT_MAG', alpha=.7)
        ax[0].legend(loc='upper right', fontsize='xx-small')
        ax[1].legend(loc='upper right', fontsize='xx-small')
        ax[0].set_xlabel('MJD + %.0f' % (np.min(time_series_epoch['MJD'])))
        ax[0].set_ylabel(r'$mag_{g}$')
        ax[1].set_xlabel('Time Index')
        ax[1].set_ylabel(r'$mag_{g}$')
        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        ax[0].grid(True)
        ax[1].grid(True)

    plt.savefig('%s/%s/%s/%s_lightcurve.png' %
                (figure_path, field, CCD, name), tight_layout=True,
                bbox_inches='tight', dpi=300)

    if len(images) > 0:
        print 'Ploting seq...'
        # fig, ax = plt.subplots(1, len(images), sharey = True,
        #                        figsize = (1. * len(images), 1.))
        n_cols_fig = int(len(images) / 2.)
        impar = False
        if len(images) % 2. == 1.:
            n_cols_fig += 1
            impar = True
        fig, ax = plt.subplots(2, n_cols_fig, sharey=True,
                               figsize=(1. * len(images), 4.5))
        print ax.shape
        for k, axx in enumerate(ax.ravel()):
            if k + 1 == len(ax.ravel()) and impar:
                axx.axis('off')
                break
            image_log = images[k]
            vmin = np.percentile(image_log, 10)
            vmax = np.percentile(image_log, 90)
            axx.imshow(image_log, interpolation='nearest', cmap='viridis',
                       vmin=vmin, vmax=vmax)
            axx.set_xlabel('%s' % epochs['MJD'][k])
            # if k in yesyes:
            #	ells = Ellipse(xy = regions[k, 0:2], width = regions[k, 2],
            #                  height = regions[k, 3], angle = regions[k, 4],
            #                  color = 'g', fill = False)
            #	ax[k].add_artist(ells)
            ax2 = axx.twiny()
            ax2.set_xlabel('%s' % epochs['EPOCH'][k], fontsize=8)

            if True:
                gif_image_log = gif_im[k]
                f1 = plt.figure()
                gif = f1.add_subplot(111)
                try:
                    vmin = np.percentile(gif_image_log, 10)
                    vmax = np.percentile(gif_image_log, 90)
                    gif.imshow(gif_image_log, interpolation='nearest',
                               cmap='viridis', vmin=vmin, vmax=vmax)
                except (IndexError):
                    gif.imshow(gif_image_log, interpolation='nearest',
                               cmap='viridis', vmin=vmin, vmax=vmax)
                gif.set_xlabel('%s' % epochs['MJD'][k], fontsize=16)
                gif.text(7, 16, '%s' % epochs['MJD'][k],
                         fontdict={'color': 'g', 'size': 40})
                gif.axes.get_xaxis().set_visible(False)
                gif.axes.get_yaxis().set_visible(False)
                f1.savefig(
                    '%s/figures/temp/gif_%s_%i.png' %
                    (jorgepath, name, k), bbox_inches='tight')
                plt.close(f1)

        fig.subplots_adjust(hspace=.3, wspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=False)
        plt.setp([a.get_yticklabels() for a in fig.axes[:]], visible=False)

        plt.savefig('%s/%s/%s/%s_sequence.png' %
                    (figure_path, field, CCD, name), dpi=300,
                    bbox_inches='tight', pad_inches=0.3)

    print 'Creating GIF animation...'

    os.system('convert -delay 20 -loop 0 %s/figures/temp/gif_%s_*.png %s/%s/%s/%s_animation.gif'
              % (jorgepath, name, figure_path, field, CCD, name))
    if True:
        print 'Removing temporal files...'
        os.system('rm %s/figures/temp/gif_%s_*.png' % (jorgepath, name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help="mode: xypix/id/list",
                        required=False, default='xypix', type=str)
    parser.add_argument('-p', '--phot_mode',
                        help="photometry mode: cat/psf/stamp",
                        required=False, default='cat', type=str)
    parser.add_argument('-F', '--field', help="HiTS field",
                        required=False, default='Blind15A_01', type=str)
    parser.add_argument('-C', '--ccd', help="HiTS ccd",
                        required=False, default='N1', type=str)
    parser.add_argument('-x', '--xcoord', help="x coordinate",
                        required=False, default=0, type=int)
    parser.add_argument('-y', '--ycoord', help="y coordinate",
                        required=False, default=0, type=int)
    parser.add_argument('-b', '--band', help="filter band",
                        required=False, default='g', type=str)
    parser.add_argument('-i', '--id',
                        help="object id or file name with list of ids",
                        required=False, default='', type=str)
    args = parser.parse_args()
    FILTER = args.band

    if args.mode == 'xypix':
        field = args.field
        ccd = args.ccd
        row_pix = int(args.ycoord)
        col_pix = int(args.xcoord)
        print field, ccd,
        print row_pix, col_pix

        if args.phot_mode == 'cat':
            run_lc_catalog(field, ccd, FILTER, row_pix, col_pix, verbose=True)
        elif args.phot_mode == 'psf':
            run_lc_psf(field, ccd, FILTER, row_pix, col_pix, verbose=True)
        elif args.phot_mode == 'stamp':
            get_stamps_lc(field, ccd, FILTER, row_pix, col_pix, verbose=True)
        else:
            print 'Wrong photometry mode...'
            sys.exit()

    if args.mode == 'id':
        print args.id
        field, ccd, col_pix, row_pix = re.findall(
            r'(\w+\d+\w?\_\d\d?)\_(\w\d+?)\_(\d+)\_(\d+)', args.id)[0]
        row_pix = int(row_pix)
        col_pix = int(col_pix)
        print field, ccd, row_pix, col_pix

        if args.phot_mode == 'cat':
            run_lc_catalog(field, ccd, FILTER, row_pix, col_pix, verbose=True)
        elif args.phot_mode == 'psf':
            run_lc_psf(field, ccd, FILTER, row_pix, col_pix, verbose=True)
        elif args.phot_mode == 'stamp':
            get_stamps_lc(field, ccd, FILTER, row_pix, col_pix, verbose=True)
        else:
            print 'Wrong photometry mode...'
            sys.exit()

    if args.mode == 'list':

        ID_table = pd.read_csv(args.id, compression='gzip')
        IDs = ID_table.internalID.values
        fail = []
        for kk, ids in enumerate(IDs):
            # if kk == 10: break
            field, ccd, col_pix, row_pix = re.findall(
                r'(\w+\d+\w?\_\d\d?)\_(\w\d+?)\_(\d+)\_(\d+)', ids)[0]
            row_pix = int(row_pix)
            col_pix = int(col_pix)
            print kk,
            print field, ccd,
            print col_pix, row_pix
            if (col_pix < dx) or (row_pix < dx) or \
               (axisX - col_pix < dx) or (axisY - row_pix < dx):
                continue
            if ccd == 'S7':
                continue

            try:
                if args.phot_mode == 'cat':
                    run_lc_catalog(field, ccd, FILTER, row_pix, col_pix,
                                   verbose=True)
                elif args.phot_mode == 'psf':
                    run_lc_psf(field, ccd, FILTER, row_pix, col_pix,
                               verbose=True)
                elif args.phot_mode == 'stamp':
                    get_stamps_lc(field, ccd, FILTER, row_pix, col_pix,
                                  verbose=True)
                else:
                    print 'Wrong photometry mode...'
                    sys.exit()
            except:
                fail.append(ids)
                print '_____________________'
                continue
            print '_____________________'
        print 'Fail: ', fail
        thefile = open('/home/jmartinez/HiTS/HiTS-Leftraru/temp/%s_fail.txt'
                       % (field), 'w')
        thefile.writelines( "%s\n" % item for item in fail)
        thefile.close()
        print 'Done!'


#######
# resize kernel to match the other
# print '\t', kernel.shape, psf_worst.shape
# diff = np.abs(kernel.shape[0] - psf_worst.shape[0])
# if kernel.shape[0] < psf_worst.shape[0]:
#     kernel = np.pad(kernel, diff/2, pad_with, padder=0)
# elif kernel.shape[0] > psf_worst.shape[0]:
#     psf_worst = np.pad(psf_worst, diff/2, pad_with, padder=0)
# print '\t', kernel.shape, psf_worst.shape
