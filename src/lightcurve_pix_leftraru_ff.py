#!/usr/bin/python

import sys
import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.table import Table, vstack
# from threading import Thread
import seaborn as sb
from astropy.stats import SigmaClip, sigma_clip, sigma_clipped_stats
from photutils import (Background2D, MedianBackground, find_peaks,
                       create_matching_kernel, SplitCosineBellWindow,
                       TopHatWindow, CosineBellWindow)
from photutils.utils import filter_data, calc_total_error
from photutils import aperture_photometry, CircularAperture
from kernel import kernel

plt.switch_backend('agg')

warnings.filterwarnings("ignore")
sb.set(style="white", color_codes=True, context="notebook", font_scale=1.4)

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'

thresh = 1.0      # threshold of catalog
minarea = 3      # min area for detection
deg2rad = 0.0174532925
rad2deg = 57.2957795
axisY = 4094
axisX = 2046
dx = 75
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


def get_stars(image, stars_cata, save=False,
              field='Blind15A_03', ccd='N1', epoch='ref'):

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
        stamp_bkg = image[row_pix - hal_size * 2: row_pix + hal_size * 2,
                          col_pix - hal_size * 2: col_pix + hal_size * 2].copy()

        # background
        sigma_clip = SigmaClip(sigma=3., iters=2)
        bkg_estimator = MedianBackground()
        bkg = Background2D(stamp_bkg, (hal_size * 2, hal_size * 2),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        stamp -= bkg.background[hal_size: hal_size * 3,
                                hal_size: hal_size * 3]
        stamps.append(stamp)

    stamps = np.array(stamps)
    print stamps.shape
    if save:
        out_stamps = open('/home/jmartinez/HiTS/temp/%s_%s_%s_stars.npy' %
                          (field, ccd, epoch), 'w')
        pickle.dump(stamps, out_stamps)
        out_stamps.close()

    return stamps


def get_psf(image, stars_cata, plot=True,
               field='Blind15A_03', ccd='N1', epoch='ref'):

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

        stamp -= bkg.background[hal_size: hal_size * 3 + 1,
                                hal_size: hal_size * 3 + 1]
        stamp /= stamp.sum()
        print '\tStar sum    :', stamp.sum()

        stamps.append(stamp)
    stamps = np.array(stamps)

    kernel = np.median(stamps, axis=0)
    kernel /= kernel.sum()
    print '\tPSF sum     :', kernel.sum()
    if plot:
        plt.imshow(kernel, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        # plt.show()

    return kernel


def create_matching_kernel_ff(stars_source, stars_target, allstars=True):
    kern = kernel(81)
    nn = int(np.minimum(len(stars_source), len(stars_target)))

    starpairs = np.stack([stars_target[:nn], stars_source[:nn]])
    _, nstars, nside, _ = np.shape(starpairs)
    npsf = nside - kern.nf

    pairs = []
    for i in range(nstars):
        star1 = starpairs[1][i]
        star2 = starpairs[0][i]
        pairs.append([star1,
                      star2])
        if i > 0 and not allstars:
            continue

    sol = kern.solve(npsf, pairs)
    # sol[sol < 0] = 0
    sol /= sol.sum()
    print sol.sum()
    print sol.shape

    return sol


def ADU2mag_PS(flux, e_flux, exptime, ZP, e_ZP):

    mag = np.array(-2.5 * np.log10(flux) +
                   2.5 * np.log10(exptime) + ZP)
    e_mag = np.sqrt(((2.5 * e_flux) / (flux * np.log(10))) ** 2 +
                    e_ZP**2)
    return (mag, e_mag*2/3)


def get_photometry(image, mask=None, gain=4., pos=(dx_stamp, dx_stamp),
                   radii=10., sigma1=None, alpha=None, beta=None, iter=0):

    print iter
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
        # plt.show()
    # error = calc_total_error(image, bkg_err, gain)
    if True:
        back_mean, back_median, back_std = sigma_clipped_stats(data, mask,
                                                               sigma=3,
                                                               iters=3,
                                                               cenfunc=np.median)
        print '\tBackground stats: %f, %f' % (back_median, back_std)
        tbl = find_peaks(data,
                         np.minimum(back_std, bkg.background_rms_median) * 3,
                         box_size=5, subpixel=True)
        if len(tbl) == 0:
            print '\tNo detection...'
            return None
        tree_XY = cKDTree(np.array([tbl['x_centroid'], tbl['y_centroid']]).T)
        if iter == 0:
            d = 9
        else:
            d = 5
        dist, indx = tree_XY.query(pos, k=2, distance_upper_bound=d)
        print tbl
        print dist, indx

        if np.isinf(dist).all():
            print '\tNo source found in the asked position... ',
            print 'using given position...'
            position = pos
            # return None
        else:
            if len(tbl) >= 2 and not np.isinf(dist[1]):
                if tbl[indx[1]]['fit_peak_value'] > \
                    tbl[indx[0]]['fit_peak_value']:
                    indx = indx[1]
                else:
                    indx = indx[0]
            else:
                indx = indx[0]
            position = [tbl[indx]['x_centroid'], tbl[indx]['y_centroid']]
    else:
        position = pos

    print '\tObject position: ', position

    apertures = [CircularAperture(position, r=r) for r in radii]
    try:
        phot_table = aperture_photometry(data, apertures, mask=mask,
                                         method='subpixel', subpixels=5)
    except IndexError:
        phot_table = aperture_photometry(data, apertures,
                                         method='subpixel', subpixels=5)
    for k, r in enumerate(radii):
        area = np.pi * r ** 2
        phot_table['aperture_flx_err_%i' %
                   k] = np.sqrt(sigma1**2 * alpha**2 * area**beta +
                                phot_table['aperture_sum_%i' % k][0] / gain)
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
            # plt.show()
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


def run_lc_psf(field, CCD, FILTER, row_pix, col_pix, seeing_limit=1.8,
               verbose=False, radec=False):
    '''
    row_pix: can be Y pix or ra
    col_pix: can be x pix or dec
    '''
    name = '%s_%s_%04i_%04i_%s' % (field, CCD, col_pix, row_pix, FILTER)
    epochs = np.loadtxt('%s/info/%s/%s_epochs_%s.txt' %
                        (jorgepath, field, field, FILTER),
                        dtype={'names': ('EPOCH', 'MJD'),
                               'formats': ('S2', 'float')}, comments='#')
    print '%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, FILTER)

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
                dqmask.append(None)
        else:
            hdu = fits.open(dqm_file)
            dqmask.append(hdu[0].data)

    # clean bad observation conditions
    seeing = np.array(seeing)
    airmass = np.array(airmass)
    images = np.array(images)
    dqmask = np.array(dqmask)
    mask_good = (seeing <= seeing_limit) & (airmass < 2.) & \
        (epochs['MJD'] <= 57073.)
    # mask_good = (epochs['MJD'] <= 57073.)
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
    if radec:
        x_kw = 'RA'
        y_kw = 'DEC'
        dist_tree = 0.0002
    else:
        x_kw = 'X_IMAGE_REF'
        y_kw = 'Y_IMAGE_REF'
        dist_tree = 9
    ref_tree_XY = cKDTree(np.transpose(np.array((ref_cata[x_kw],
                                                 ref_cata[y_kw]))))
    # quering asked position in worst image
    XY_worst = np.transpose(np.array((col_pix, row_pix)))
    print XY_worst
    dist_w, indx_w = ref_tree_XY.query(XY_worst, k=1,
                                       distance_upper_bound=dist_tree)
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
    psf_worst = get_psf(images[idx_worst], nearest_cata, plot=False)
    stars_worst = get_stars(images[idx_worst], nearest_cata,
                            field=field, ccd=CCD, epoch='ref')
    if verbose:
        print 'Kernel done!'
        print 'Going into time serie...'

    # aperture radii for photometry
    ap_radii = seeing[idx_worst] / 0.27 * np.array([0.5, 0.75, 1., 1.25, 1.5])
    print 'Aperture radii: ', ap_radii
    lc_yes, lc_no = [], []
    # lc_raw = []
    stamps_lc = []
    airmass2, exptime2, ZP = [], [], []
    pos_persis = []

    for k, epoch in enumerate(epochs):

        if verbose:
            print 'Working in epoch %s...' % (epoch[0])

        # loading catalogs
        cat_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
            (jorgepath, field, CCD, field, CCD,
             epoch[0], str(thresh), str(minarea))
        if not os.path.exists(cat_file):
            if verbose:
                print '\t\tNo catalog file for epoch: %s' % (cat_file)
            continue
        cata = Table.read(cat_file, format='ascii')
        cata_XY = np.transpose(np.array((cata[x_kw],
                                         cata[y_kw])))
        tree_XY = cKDTree(cata_XY)

        # quering asked position
        XY_obj = np.transpose(np.array((col_pix, row_pix)))
        dist, indx = tree_XY.query(XY_obj, k=1, distance_upper_bound=dist_tree)
        print dist
        if np.isinf(dist):
            if verbose:
                print '\t\tNo match in epoch %s' % epoch[0]
                lc_no.append(epoch)
                continue

        # position in non projected coordinates, i.e. loaded image
        row_pix2 = int(cata['Y_IMAGE'][indx])
        col_pix2 = int(cata['X_IMAGE'][indx])
        err_mag = cata['MAGERR_AUTO_ZP'][indx]
        xx = cata['X_IMAGE'][indx] - float(col_pix2) + dx_stamp
        yy = cata['Y_IMAGE'][indx] - float(row_pix2) + dx_stamp

        stamp = images[k][row_pix2 - dx_stamp:
                          row_pix2 + dx_stamp + 1,
                          col_pix2 - dx_stamp:
                          col_pix2 + dx_stamp + 1].copy()
        if (dqmask[k] is not None):
            print 'Ok'
            stamp_mask = dqmask[k][row_pix2 - dx_stamp:
                                   row_pix2 + dx_stamp + 1,
                                   col_pix2 - dx_stamp:
                                   col_pix2 + dx_stamp + 1].copy()
        else:
            print 'No'
            stamp_mask = None

        if epoch[0] != epochs[idx_worst][0]:
            nearest_cata2 = get_filter_nearest_stars(cata, col_pix2, row_pix2)
            if nearest_cata2 is None:
                continue
            psf = get_psf(images[k], nearest_cata2, plot=False)
            stars = get_stars(images[k], nearest_cata2,
                              field=field, ccd=CCD, epoch=epoch[0])

            # resize psf to match the other
            # print '\t', psf.shape, psf_worst.shape
            # diff = np.abs(psf.shape[0] - psf_worst.shape[0])
            # if psf.shape[0] < psf_worst.shape[0]:
            #     psf = np.pad(psf, diff/2, pad_with, padder=0)
            # elif psf.shape[0] > psf_worst.shape[0]:
            #    psf_worst = np.pad(psf_worst, diff/2, pad_with,
            #                          padder=0)
            # print '\t', psf.shape, psf_worst.shape

            matched_kernel = create_matching_kernel_ff(stars, stars_worst)

            print '\tMatching kernel sum: ', matched_kernel.sum()
            convolved_psf = filter_data(psf, matched_kernel, mode='nearest')

            convolved_stamp = filter_data(stamp, matched_kernel,
                                          mode='nearest')
        else:
            print '\tWorst epoch'
            matched_kernel = psf_worst.copy()
            stamp = stamp_worst.copy()
            convolved_stamp = stamp_worst.copy()

        pix_corr_file = '%s/info/%s/%s/pixcorrcoef_%s_%s_%s_%i.npy' % \
            (jorgepath, field, CCD, field, CCD, epoch[0], 500)
        temp = np.load(pix_corr_file)
        sigma1, alpha, beta = temp[0], temp[1], temp[2]
        # print 'sigma1 = %f' % (sigma1)
        # print 'alpha = %f' % (alpha)
        # print 'beta = %f' % (beta)
        if k > 0:
            xx, yy = pos_persis[0], pos_persis[1]
        data_point = get_photometry(convolved_stamp, mask=stamp_mask,
                                    gain=gain[k], pos=(xx, yy),
                                    radii=ap_radii, sigma1=sigma1,
                                    alpha=alpha, beta=beta, iter=k)

        if data_point is None:
            print '\t\tNo match in epoch %s' % epoch[0]
            lc_no.append(epoch)
            continue

        stamps_lc.append([stamp, convolved_stamp, matched_kernel,
                          float(epoch[1]), convolved_psf])
        data_point['mjd'] = float(epoch[1])
        data_point['epoch'] = epoch[0]
        data_point['aperture_mag_err_0_cat'] = err_mag
        data_point['ra'] = cata['RA'][indx]
        data_point['dec'] = cata['DEC'][indx]

        if k == 0:
            pos_persis = [data_point['xcenter'][0], data_point['ycenter'][0]]

        lc_yes.append(data_point)
        airmass2.append(airmass[k])
        exptime2.append(exptime[k])

        ZP_PS = np.load('%s/info/%s/%s/ZP_%s_PS_%s_%s_%s.npy' %
                        (jorgepath, field, CCD, 'AUTO', field, CCD, epoch[0]))
        ZP.append([ZP_PS[0][0], ZP_PS[2][0]])

        if True:

            fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(18, 10))

            im2 = ax[0, 0].imshow(stamp,
                                  cmap='viridis', interpolation='nearest')
            ax[0, 0].set_title(epoch[0])
            fig.colorbar(im2, ax=ax[0, 0], fraction=0.046, pad=0.04)

            im1 = ax[0, 1].imshow(stamp_worst,
                                  cmap='viridis', interpolation='nearest')
            ax[0, 1].set_title('Worst')
            fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

            im3 = ax[0, 2].imshow(convolved_stamp,
                                  cmap='viridis', interpolation='nearest')
            ax[0, 2].set_title('%s convolved' % epoch[0])
            for k in ap_radii:
                circle = plt.Circle([data_point['xcenter'][0],
                                     data_point['ycenter'][0]],
                                    k, color='r', fill=False)
                ax[0, 2].add_artist(circle)
            fig.colorbar(im3, ax=ax[0, 2], fraction=0.046, pad=0.04)

            im4 = ax[0, 3].imshow(psf_worst - convolved_psf,
                                  cmap='viridis', interpolation='nearest')
            ax[0, 3].set_title('kernel sustraction')
            fig.colorbar(im4, ax=ax[0, 3], fraction=0.046, pad=0.04)

            im2 = ax[1, 0].imshow(psf, cmap='viridis',
                                  interpolation='nearest')
            ax[1, 0].set_title('%s' % (epoch[0]))
            fig.colorbar(im2, ax=ax[1, 0], fraction=0.046, pad=0.04)

            im1 = ax[1, 1].imshow(psf_worst, cmap='viridis',
                                  interpolation='nearest')
            ax[1, 1].set_title('Worst')
            fig.colorbar(im1, ax=ax[1, 1], fraction=0.046, pad=0.04)

            im3 = ax[1, 2].imshow(matched_kernel, cmap='viridis',
                                  interpolation='nearest')
            ax[1, 2].set_title('matching kernel')
            fig.colorbar(im3, ax=ax[1, 2], fraction=0.046, pad=0.04)

            im4 = ax[1, 3].imshow(convolved_psf, cmap='viridis',
                                  interpolation='nearest')
            ax[1, 3].set_title('convolved %s' % (epoch[0]))
            fig.colorbar(im4, ax=ax[1, 3], fraction=0.046, pad=0.04)
            fig.tight_layout()
            plt.savefig('%s/lightcurves/galaxy/kernels/%s_%s_psf.png' %
                        (jorgepath, name, epoch[0]),
                        tight_layout=True, pad_inches=0.01,
                        bbox_inches='tight')
            # plt.show()

        # if k == 3: break

    if len(lc_yes) <= 10:
        print 'No LC for this source...'
        return None
    lc = vstack(lc_yes)
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
    print len(stamps_lc)
    print lc_df[['ra', 'dec']]
    # print lc_df[['aperture_flx_0', 'aperture_flx_err_0',
    #              'aperture_flx_err_1', 'aperture_flx_err_2',
    #              'aperture_mag_err_0', 'aperture_mag_err_1',
    #              'aperture_mag_err_2', 'aperture_mag_err_0_cat']]

    if not os.path.exists('%s/lightcurves/galaxy/%s' % (jorgepath, field)):
        print "Creating field folder"
        os.makedirs('%s/lightcurves/galaxy/%s' % (jorgepath, field))

    if True:
        print 'saving table...'
        f = open('%s/lightcurves/galaxy/%s/%s_psf_ff.csv' %
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
        print 'saving figure...'
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
                            cmap='gray', origin='lower')
            circle = plt.Circle([lc_df['xcenter'][i],
                                 lc_df['ycenter'][i]],
                                ap_radii[0], color='r', lw=.5, fill=False)
            ax[2, i].add_artist(circle)
            circle = plt.Circle([lc_df['xcenter'][i],
                                 lc_df['ycenter'][i]],
                                ap_radii[1], color='r', lw=.5, fill=False)
            ax[2, i].add_artist(circle)
            cmin_k = np.percentile(stamps_lc[i][2].flatten(), 10)
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
        plt.savefig('%s/lightcurves/galaxy/%s/%s_psf_series_ff.png' %
                    (jorgepath, field, name),
                    tight_layout=True, pad_inches=0.01, facecolor='black',
                    bbox_inches='tight')
        # plt.show()
        plt.close(fig)

    if False:
        plt.plot(lc_df['mjd'],
                 lc_df['aperture_mag_0'],
                 '.k')
        plt.xlabel('cat')
        plt.ylabel('cat - here')
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help="mode: xypix/radec/id/list",
                        required=False, default='xypix', type=str)
    parser.add_argument('-p', '--phot_mode',
                        help="photometry mode: cat/psf/stamp",
                        required=False, default='cat', type=str)
    parser.add_argument('-F', '--field', help="HiTS field",
                        required=False, default='Blind15A_01', type=str)
    parser.add_argument('-C', '--ccd', help="HiTS ccd",
                        required=False, default='N1', type=str)
    parser.add_argument('-x', '--xcoord', help="x coordinate",
                        required=False, default=0, type=float)
    parser.add_argument('-y', '--ycoord', help="y coordinate",
                        required=False, default=0, type=float)
    parser.add_argument('-b', '--band', help="filter band",
                        required=False, default='g', type=str)
    parser.add_argument('-i', '--id',
                        help="object id or file name with list of ids",
                        required=False, default='', type=str)
    args = parser.parse_args()
    FILTER = args.band

    if args.mode in 'xypix':
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

    if args.mode in 'radec':
        field = args.field
        ccd = args.ccd
        row_pix = args.ycoord
        col_pix = args.xcoord
        print field, ccd, row_pix, col_pix

        if args.phot_mode == 'psf':
            run_lc_psf(field, ccd, FILTER, row_pix, col_pix,
                       verbose=True, radec=True)
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
        print field, ccd,
        print row_pix, col_pix

        if args.phot_mode == 'psf':
            run_lc_psf(field, ccd, FILTER, row_pix, col_pix, verbose=True)
        elif args.phot_mode == 'stamp':
            get_stamps_lc(field, ccd, FILTER, row_pix, col_pix, verbose=True)
        else:
            print 'Wrong photometry mode...'
            sys.exit()

    if args.mode == 'list':

        print args.id
        ID_table = pd.read_csv(args.id, compression='gzip')
        ID_table.set_index('internalID', inplace=True)
        print ID_table.shape
        # sys.exit()
        IDs = ID_table.index.values
        fail = []
        for kk, ids in enumerate(IDs):
            # if kk == 10: break
            field, ccd, col_pix, row_pix = re.findall(
                r'(\w+\d+\w?\_\d\d?)\_(\w\d+?)\_(\d+)\_(\d+)', ids)[0]
            row_pix = int(row_pix)
            col_pix = int(col_pix)
            # col_pix = ID_table.loc[ids, 'raMedian']
            # row_pix = ID_table.loc[ids, 'decMedian']
            print kk, field, ccd, col_pix, row_pix
            if (col_pix < dx) or (row_pix < dx) or \
               (axisX - col_pix < dx) or (axisY - row_pix < dx):
                continue
            if ccd == 'S7':
                continue

            try:
                if args.phot_mode == 'psf':
                    run_lc_psf(field, ccd, FILTER, row_pix, col_pix,
                               verbose=True, radec=False)
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
            # break
        print 'Fail: ', fail
        thefile = open('/home/jmartinez/HiTS/HiTS-Leftraru/temp/%s_fail.txt'
                       % (field), 'w')
        thefile.writelines( "%s\n" % item for item in fail)
        thefile.close()
        print 'Done!'
