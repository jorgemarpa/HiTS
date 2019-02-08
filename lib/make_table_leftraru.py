#!/usr/bin/python

# cript to calculate and plots statistics in lightcurves
# for a given field all CCD

import sys
import os
import getopt
from datetime import datetime
import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import warnings
from astropy.table import Table
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from functools import reduce
# from misc_func_leftraru import *

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'
sharepath = '/home/apps/astro/home/jmartinez'

thresh = 1.0      # threshold of catalog
minarea = 3      # min area for detection

# 0 = number
# 1,2 = X,Y
# 3,4 = RA,DEC
# 5,6 = FLUX-ERR_AUTO
# 7, 8 = MAG-ERR_AUTO
# 9 = FLUX_RADIUS
# 10 = FWHM
# 11 = CLASS_STAR
# 12 = ELONGATION
# 13 = FLAG
# 14 = A_IMAGE
# 15 = B_IMAGE
# 16 = THETA_IMAGE
# 17 = KRON_RADIUS


def get_values(field, CCD, occ=0, band='g'):

    warnings.filterwarnings("ignore")

#########################################################################
#########################################################################

    if band in ['u', 'i']:
        occ = 1

    epochs_file = '%s/info/%s/%s_epochs_%s.txt' % (
        jorgepath, field, field, band)
    if not os.path.exists(epochs_file):
        print '..No epochs file: %s' % (epochs_file)
        sys.exit()
    epochs = np.loadtxt(epochs_file, comments='#', dtype=str)
    if len(epochs) == 0:
        print 'No observations in this filter...'
        return None
    if epochs.shape == (2,):
        epochs = epochs.reshape(1, 2)

    epoch_c = []
    to_remove = []
    print '\tLoading catalogues files'
    for i, epoch in enumerate(epochs):
        print '\t\tEpoch %s' % epoch[0]

        # catalogues

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat"\
            % (jorgepath, field, CCD, field, CCD, epoch[0], str(thresh), str(minarea))
        if not os.path.exists(cata_file):
            print 'No catalog file: %s' % (cata_file)
            to_remove.append(i)
            continue
        cata = Table.read(cata_file, format='ascii')

        # epoch_c has all the catalogues, each element of epoch_c contain the
        # catalogue of a given epoch
        epoch_c.append(cata)

    if len(epoch_c) < occ:
        print '...Not enough catalogues'
        print '______________________________________________________________'
        return None

    indx_file = "%s/lightcurves/%s/%s/%s_%s_%s_master_index.txt" % (jorgepath,
                                                                    field, CCD,
                                                                    field,
                                                                    CCD, band)
    if not os.path.exists(indx_file):
        print '...No master index file: %s' % (indx_file)
        return None
    master_cat = np.loadtxt(indx_file, comments='#', dtype='i4')

    if master_cat.shape == (len(master_cat),):
        master_cat = master_cat.reshape(len(master_cat), 1)
    print '\tNumber of epochs: %i, effective: %i' % (epochs.shape[0],
                                                     len(master_cat[0]))
    epochs = np.delete(epochs, to_remove, axis=0)

    if band == 'u':
        corr = 'ZA'
    else:
        corr = 'ZP'

    ##########################################################################
    #                                objects > .8                            #
    ##########################################################################

    median_mag, std_mag, median_err = [], [], []
    flx, flx_err, flx_std = [], [], []
    median_mag_ap1, std_mag_ap1, median_err_ap1 = [], [], []
    flx_ap1, flx_err_ap1, flx_std_ap1 = [], [], []
    median_mag_ap2, std_mag_ap2, median_err_ap2 = [], [], []
    flx_ap2, flx_err_ap2, flx_std_ap2 = [], [], []
    median_mag_ap3, std_mag_ap3, median_err_ap3 = [], [], []
    flx_ap3, flx_err_ap3, flx_std_ap3 = [], [], []
    median_mag_ap4, std_mag_ap4, median_err_ap4 = [], [], []
    flx_ap4, flx_err_ap4, flx_std_ap4 = [], [], []
    median_mag_ap5, std_mag_ap5, median_err_ap5 = [], [], []
    flx_ap5, flx_err_ap5, flx_std_ap5 = [], [], []
    pix_coord, radec_coord, radec_std_err = [], [], []
    obj_id = []
    show, flux_rad, fwhm, ellip = [], [], [], []
    kron_rad, flag, class_star = [], [], []

    for obj in range(len(master_cat)):
        print '\r\t\t Object %i of %i' % (obj, len(master_cat)),

        num_obs = np.where(master_cat[obj, :] > 0)
        if len(num_obs[0]) >= occ:
            show.append(len(num_obs[0]))
            err_lc, mag_lc, flx_lc, flx_e_lc = [], [], [], []
            err_lc_ap1, mag_lc_ap1, flx_lc_ap1, flx_e_lc_ap1 = [], [], [], []
            err_lc_ap2, mag_lc_ap2, flx_lc_ap2, flx_e_lc_ap2 = [], [], [], []
            err_lc_ap3, mag_lc_ap3, flx_lc_ap3, flx_e_lc_ap3 = [], [], [], []
            err_lc_ap4, mag_lc_ap4, flx_lc_ap4, flx_e_lc_ap4 = [], [], [], []
            err_lc_ap5, mag_lc_ap5, flx_lc_ap5, flx_e_lc_ap5 = [], [], [], []
            pix_x, pix_y, ra, dec = [], [], [], []
            frad, fw, ell, krad, flg, clst = [], [], [], [], [], []
            for tim in range(len(master_cat[obj, :])):
                pos = master_cat[obj, tim]
                if pos >= 0:
                    flx_lc.append(epoch_c[tim]['FLUX_AUTO'][pos])
                    mag_lc.append(epoch_c[tim]['MAG_AUTO_%s' % corr][pos])
                    flx_e_lc.append(epoch_c[tim]['FLUXERR_AUTO_COR'][pos])
                    err_lc.append(epoch_c[tim]['MAGERR_AUTO_%s' % corr][pos])

                    flx_lc_ap1.append(epoch_c[tim]['FLUX_APER'][pos])
                    mag_lc_ap1.append(
                        epoch_c[tim]['MAG_APER_%s' % corr][pos])
                    flx_e_lc_ap1.append(
                        epoch_c[tim]['FLUXERR_APER_COR'][pos])
                    err_lc_ap1.append(
                        epoch_c[tim]['MAGERR_APER_%s' % corr][pos])

                    flx_lc_ap2.append(epoch_c[tim]['FLUX_APER_1'][pos])
                    mag_lc_ap2.append(
                        epoch_c[tim]['MAG_APER_1_%s' % corr][pos])
                    flx_e_lc_ap2.append(
                        epoch_c[tim]['FLUXERR_APER_1_COR'][pos])
                    err_lc_ap2.append(
                        epoch_c[tim]['MAGERR_APER_1_%s' % corr][pos])

                    flx_lc_ap3.append(epoch_c[tim]['FLUX_APER_2'][pos])
                    mag_lc_ap3.append(
                        epoch_c[tim]['MAG_APER_2_%s' % corr][pos])
                    flx_e_lc_ap3.append(
                        epoch_c[tim]['FLUXERR_APER_2_COR'][pos])
                    err_lc_ap3.append(
                        epoch_c[tim]['MAGERR_APER_2_%s' % corr][pos])

                    flx_lc_ap4.append(epoch_c[tim]['FLUX_APER_3'][pos])
                    mag_lc_ap4.append(
                        epoch_c[tim]['MAG_APER_3_%s' % corr][pos])
                    flx_e_lc_ap4.append(
                        epoch_c[tim]['FLUXERR_APER_3_COR'][pos])
                    err_lc_ap4.append(
                        epoch_c[tim]['MAGERR_APER_3_%s' % corr][pos])

                    flx_lc_ap5.append(epoch_c[tim]['FLUX_APER_4'][pos])
                    mag_lc_ap5.append(
                        epoch_c[tim]['MAG_APER_4_%s' % corr][pos])
                    flx_e_lc_ap5.append(
                        epoch_c[tim]['FLUXERR_APER_4_COR'][pos])
                    err_lc_ap5.append(
                        epoch_c[tim]['MAGERR_APER_4_%s' % corr][pos])
                    # err_lc.append(epoch_c[tim]['MAGERR_AUTO_ZACI'][pos])

                    pix_x.append(epoch_c[tim]['X_IMAGE_REF'][pos])
                    pix_y.append(epoch_c[tim]['Y_IMAGE_REF'][pos])
                    ra.append(epoch_c[tim]['RA'][pos])
                    dec.append(epoch_c[tim]['DEC'][pos])

                    frad.append(epoch_c[tim]['FLUX_RADIUS'][pos])
                    fw.append(epoch_c[tim]['FWHM_IMAGE'][pos])
                    ell.append(1 - 1 / epoch_c[tim]['ELONGATION'][pos])
                    krad.append(epoch_c[tim]['KRON_RADIUS'][pos])
                    flg.append(epoch_c[tim]['FLAGS'][pos])
                    clst.append(epoch_c[tim]['CLASS_STAR'][pos])

            median_mag.append(np.median(mag_lc))
            std_mag.append(np.std(mag_lc))
            median_err.append(np.median(err_lc))
            flx.append(np.median(flx_lc))
            flx_err.append(np.median(flx_e_lc))
            flx_std.append(np.std(flx_e_lc))

            median_mag_ap1.append(np.median(mag_lc_ap1))
            std_mag_ap1.append(np.std(mag_lc_ap1))
            median_err_ap1.append(np.median(err_lc_ap1))
            flx_ap1.append(np.median(flx_lc_ap1))
            flx_err_ap1.append(np.median(flx_e_lc_ap1))
            flx_std_ap1.append(np.std(flx_e_lc_ap1))

            median_mag_ap2.append(np.median(mag_lc_ap2))
            std_mag_ap2.append(np.std(mag_lc_ap2))
            median_err_ap2.append(np.median(err_lc_ap2))
            flx_ap2.append(np.median(flx_lc_ap2))
            flx_err_ap2.append(np.median(flx_e_lc_ap2))
            flx_std_ap2.append(np.std(flx_e_lc_ap2))

            median_mag_ap3.append(np.median(mag_lc_ap3))
            std_mag_ap3.append(np.std(mag_lc_ap3))
            median_err_ap3.append(np.median(err_lc_ap3))
            flx_ap3.append(np.median(flx_lc_ap3))
            flx_err_ap3.append(np.median(flx_e_lc_ap3))
            flx_std_ap3.append(np.std(flx_e_lc_ap3))

            median_mag_ap4.append(np.median(mag_lc_ap4))
            std_mag_ap4.append(np.std(mag_lc_ap4))
            median_err_ap4.append(np.median(err_lc_ap4))
            flx_ap4.append(np.median(flx_lc_ap4))
            flx_err_ap4.append(np.median(flx_e_lc_ap4))
            flx_std_ap4.append(np.std(flx_e_lc_ap4))

            median_mag_ap5.append(np.median(mag_lc_ap5))
            std_mag_ap5.append(np.std(mag_lc_ap5))
            median_err_ap5.append(np.median(err_lc_ap5))
            flx_ap5.append(np.median(flx_lc_ap5))
            flx_err_ap5.append(np.median(flx_e_lc_ap5))
            flx_std_ap5.append(np.std(flx_e_lc_ap5))

            pix_coord.append(np.array([np.median(pix_x), np.median(pix_y)]))
            radec_coord.append(np.array([np.median(ra), np.median(dec)]))
            radec_std_err.append(np.array([np.std(ra), np.std(dec)]))
            obj_id.append('%s_%s_%04i_%04i' %
                          (field, CCD, pix_coord[-1][0], pix_coord[-1][1]))

            flux_rad.append(np.median(frad))
            fwhm.append(np.median(fw))
            ellip.append(np.median(ell))
            kron_rad.append(np.median(krad))
            flag.append(np.max(flg))
            class_star.append(np.max(clst))

    median_mag = np.asarray(median_mag)
    std_mag = np.asarray(std_mag)
    median_err = np.asarray(median_err)
    flx = np.asarray(flx)
    flx_err = np.asarray(flx_err)
    flx_std = np.asarray(flx_std)

    median_mag_ap1 = np.asarray(median_mag_ap1)
    std_mag_ap1 = np.asarray(std_mag_ap1)
    median_err_ap1 = np.asarray(median_err_ap1)
    flx_ap1 = np.asarray(flx_ap1)
    flx_err_ap1 = np.asarray(flx_err_ap1)
    flx_std_ap1 = np.asarray(flx_std_ap1)

    median_mag_ap2 = np.asarray(median_mag_ap2)
    std_mag_ap2 = np.asarray(std_mag_ap2)
    median_err_ap2 = np.asarray(median_err_ap2)
    flx_ap2 = np.asarray(flx_ap2)
    flx_err_ap2 = np.asarray(flx_err_ap2)
    flx_std_ap2 = np.asarray(flx_std_ap2)

    median_mag_ap3 = np.asarray(median_mag_ap3)
    std_mag_ap3 = np.asarray(std_mag_ap3)
    median_err_ap3 = np.asarray(median_err_ap3)
    flx_ap3 = np.asarray(flx_ap3)
    flx_err_ap3 = np.asarray(flx_err_ap3)
    flx_std_ap3 = np.asarray(flx_std_ap3)

    median_mag_ap4 = np.asarray(median_mag_ap4)
    std_mag_ap4 = np.asarray(std_mag_ap4)
    median_err_ap4 = np.asarray(median_err_ap4)
    flx_ap4 = np.asarray(flx_ap4)
    flx_err_ap4 = np.asarray(flx_err_ap4)
    flx_std_ap4 = np.asarray(flx_std_ap4)

    median_mag_ap5 = np.asarray(median_mag_ap5)
    std_mag_ap5 = np.asarray(std_mag_ap5)
    median_err_ap5 = np.asarray(median_err_ap5)
    flx_ap5 = np.asarray(flx_ap5)
    flx_err_ap5 = np.asarray(flx_err_ap5)
    flx_std_ap5 = np.asarray(flx_std_ap5)

    pix_coord = np.asarray(pix_coord)
    radec_coord = np.asarray(radec_coord)
    radec_std_err = np.asarray(radec_std_err)

    show = np.asarray(show, dtype=int)
    flux_rad = np.asarray(flux_rad)
    fwhm = np.asarray(fwhm)
    ellip = np.asarray(ellip)
    kron_rad = np.asarray(kron_rad)
    flag = np.asarray(flag)
    class_star = np.asarray(class_star)

    # print 'show ', show.shape
    # print 'median_mag ', median_mag.shape
    # print 'mad_mag ', mad_mag.shape
    # print 'mean_mag ', mean_mag.shape
    # print 'std_mag ', std_mag.shape
    # print 'mean_err ', mean_err.shape
    # print 'median_err ', median_err.shape
    # print 'pix_coord ', pix_coord.shape
    # print 'radec_coord ', radec_coord.shape

    print '\n\tNumber of objects: %i' % (len(show))
    print '___________________________________________________________________'

    dicty = {'internalID': obj_id,
             'X': pix_coord[:, 0],
             'Y': pix_coord[:, 1],
             'raMedian': radec_coord[:, 0],
             'decMedian': radec_coord[:, 1],
             'raMedianStd': radec_std_err[:, 0],
             'decMedianStd': radec_std_err[:, 1],
             '%sN' % (band): show,
             '%sFluxRadius' % (band): flux_rad,
             '%sFWHM' % (band): fwhm,
             '%sEllipticity' % (band): ellip,
             '%sKronRadius' % (band): kron_rad,
             '%sFlags' % (band): flag,
             '%sClassStar' % (band): class_star,
             '%sKronFlux' % (band): flx,
             '%sKronFluxErr' % (band): flx_err,
             '%sKronFluxStd' % (band): flx_std,
             '%sMedianKronMag' % (band): median_mag,
             '%sMedianKronMagStd' % (band): std_mag,
             '%sMedianKronMagErr' % (band): median_err,
             '%sAp1Flux' % (band): flx_ap1,
             '%sAp1FluxErr' % (band): flx_err_ap1,
             '%sAp1FluxStd' % (band): flx_std_ap1,
             '%sMedianAp1Mag' % (band): median_mag_ap1,
             '%sMedianAp1MagStd' % (band): std_mag_ap1,
             '%sMedianAp1MagErr' % (band): median_err_ap1,
             '%sAp2Flux' % (band): flx_ap2,
             '%sAp2FluxErr' % (band): flx_err_ap2,
             '%sAp2FluxStd' % (band): flx_std_ap2,
             '%sMedianAp2Mag' % (band): median_mag_ap2,
             '%sMedianAp2MagStd' % (band): std_mag_ap2,
             '%sMedianAp2MagErr' % (band): median_err_ap2,
             '%sAp3Flux' % (band): flx_ap3,
             '%sAp3FluxErr' % (band): flx_err_ap3,
             '%sAp3FluxStd' % (band): flx_std_ap3,
             '%sMedianAp3Mag' % (band): median_mag_ap3,
             '%sMedianAp3MagStd' % (band): std_mag_ap3,
             '%sMedianAp3MagErr' % (band): median_err_ap3,
             '%sAp4Flux' % (band): flx_ap4,
             '%sAp4FluxErr' % (band): flx_err_ap4,
             '%sAp4FluxStd' % (band): flx_std_ap4,
             '%sMedianAp4Mag' % (band): median_mag_ap4,
             '%sMedianAp4MagStd' % (band): std_mag_ap4,
             '%sMedianAp4MagErr' % (band): median_err_ap4,
             '%sAp5Flux' % (band): flx_ap5,
             '%sAp5FluxErr' % (band): flx_err_ap5,
             '%sAp5FluxStd' % (band): flx_std_ap5,
             '%sMedianAp5Mag' % (band): median_mag_ap5,
             '%sMedianAp5MagStd' % (band): std_mag_ap5,
             '%sMedianAp5MagErr' % (band): median_err_ap5}
    data_frame = pd.DataFrame(dicty)

    col_names = ['internalID', 'X', 'Y', 'raMedian', 'decMedian',
                 'raMedianStd', 'decMedianStd', '%sN' % (band),
                 '%sFluxRadius' % (band), '%sFWHM' % (band),
                 '%sEllipticity' % (band), '%sKronRadius' % (band),
                 '%sFlags' % (band), '%sClassStar' % (band),
                 '%sKronFlux' % (band), '%sKronFluxErr' % (band),
                 '%sKronFluxStd' % (band),
                 '%sMedianKronMag' % (band), '%sMedianKronMagStd' % (band),
                 '%sMedianKronMagErr' % (band),
                 '%sAp1Flux' % (band), '%sAp1FluxErr' % (band),
                 '%sAp1FluxStd' % (band),
                 '%sMedianAp1Mag' % (band), '%sMedianAp1MagStd' % (band),
                 '%sMedianAp1MagErr' % (band),
                 '%sAp2Flux' % (band), '%sAp2FluxErr' % (band),
                 '%sAp2FluxStd' % (band),
                 '%sMedianAp2Mag' % (band), '%sMedianAp2MagStd' % (band),
                 '%sMedianAp2MagErr' % (band),
                 '%sAp3Flux' % (band), '%sAp3FluxErr' % (band),
                 '%sAp3FluxStd' % (band),
                 '%sMedianAp3Mag' % (band), '%sMedianAp3MagStd' % (band),
                 '%sMedianAp3MagErr' % (band),
                 '%sAp4Flux' % (band), '%sAp4FluxErr' % (band),
                 '%sAp4FluxStd' % (band),
                 '%sMedianAp4Mag' % (band), '%sMedianAp4MagStd' % (band),
                 '%sMedianAp4MagErr' % (band),
                 '%sAp5Flux' % (band), '%sAp5FluxErr' % (band),
                 '%sAp5FluxStd' % (band),
                 '%sMedianAp5Mag' % (band), '%sMedianAp5MagStd' % (band),
                 '%sMedianAp5MagErr' % (band)]

    return data_frame[col_names]


def remove_non_unique_by_distance(index, dist, mask_yes, exc):
    unq, unq_idx = np.unique(index, return_inverse=True)
    unq_cnt = np.bincount(unq_idx)
    cnt_mask = unq_cnt > 1
    cnt_idx, = np.nonzero(cnt_mask)
    idx_mask = np.in1d(unq_idx, cnt_idx)
    idx_idx, = np.nonzero(idx_mask)
    srt_idx = np.argsort(unq_idx[idx_mask])
    dup_idx = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])

    to_remove = []
    for d_i in dup_idx:
        if index[d_i][0] == exc:
            continue
        idx_min = np.argmin(dist[d_i])
        idx_max = np.argmax(dist[d_i])
        to_remove.append(d_i[idx_min])
        mask_yes[d_i[idx_max]] = False
    return mask_yes


def merge_catalogs(main, other, on='pixel', how='outer'):

    main_XY = np.transpose(np.array([main.X, main.Y]))
    tree = cKDTree(main_XY)

    other_XY = np.transpose(np.array([other.X, other.Y]))

    print 'main: ', main_XY.shape
    print 'other: ', other_XY.shape

    dist, idx = tree.query(other_XY, k=1, distance_upper_bound=3)
    mask_yes = (idx != len(main_XY))            # mask with match on main
    if len(idx[mask_yes]) != len(np.unique(idx[mask_yes])):
        print 'Exceed by: ', len(idx[mask_yes]) - len(np.unique(idx[mask_yes]))
        mask_yes = remove_non_unique_by_distance(idx, dist, mask_yes,
                                                 len(main_XY))
        idx_yes = idx[mask_yes]
    else:
        idx_yes = idx[mask_yes]

    other_no = other[~mask_yes]
    other_yes = other[mask_yes]
    print 'other_no: ', other_no.shape
    print 'other_yes: ', other_yes.shape

    idx_no = [x for x in range(len(main_XY)) if x not in idx_yes]

    main_yes = main.iloc[idx_yes]
    main_no = main.iloc[idx_no]
    print 'main_yes: ', main_yes.shape
    print 'main_no: ', main_no.shape

    other_yes.drop('internalID', axis=1, inplace=True)
    other_yes.drop('X', axis=1, inplace=True)
    other_yes.drop('Y', axis=1, inplace=True)
    other_yes.drop('raMedian', axis=1, inplace=True)
    other_yes.drop('raMedianStd', axis=1, inplace=True)
    other_yes.drop('decMedian', axis=1, inplace=True)
    other_yes.drop('decMedianStd', axis=1, inplace=True)

    cols = list(main_yes.columns.values) + list(other_yes.columns.values)
    merged_cata = pd.DataFrame(np.hstack([main_yes.values, other_yes.values]),
                               columns=cols)
    merged_cata = pd.concat([merged_cata, main_no, other_no])
    print 'Before drop duplicates: ', merged_cata.shape
    merged_cata.drop_duplicates(subset=['internalID', 'X', 'Y'], inplace=True)
    print 'After drop duplicates : ', merged_cata.shape
    print 'done\n___________________________\n'

    return merged_cata[cols]


def make_table(field, CHIPS, occ, filters='ugri'):

    frames_all = []
    print 'Field: ', field
    occ_in = occ

    for ccd in CHIPS[:]:
        if CCD != 'all':
            ccd = CCD
        print 'CCD: %s' % ccd

        # table for all bands
        band_table = []
        for m, filt in enumerate(filters):
            # if filt == 'g': continue
            print 'Band: ', filt
            if filt in ['r', 'i']:
                occ = 1
            frame = get_values(field, ccd, occ, band=filt)
            if isinstance(frame, pd.DataFrame):
                print 'Table shape: ', frame.shape
                frame['%sN' % (filt)].fillna(value=0, inplace=True)
                band_table.append(frame)

        if len(band_table) == 0:
            print 'No tables to merge'
            sys.exit()
        # merge all band catalogs available
        df_final = reduce(lambda left, right:
                          merge_catalogs(left, right), band_table)
        df_final.drop_duplicates(subset=['internalID', 'X', 'Y'], inplace=True)
        # filling nans = 0 for number os observations
        for filt in filters:
            try:
                df_final['%sN' % (filt)].fillna(value=0, inplace=True)
            except BaseException:
                print 'No filter %s to replace nans...' % (filt)
        df_final.fillna(value=-999, inplace=True)
        # add survey ID name
        ID_all = []
        for i, [ra, dec] in enumerate(zip(df_final.raMedian.values,
                                          df_final.decMedian.values)):
            print '\r %i' % i,
            c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
            # print c.ra.hms
            # print c.dec.dms
            ID = 'HiTS%02i%02i%02i' % (
                int(c.ra.hms[0]), int(c.ra.hms[1]), int(c.ra.hms[2]))
            if dec > 0:
                ID = '%s+%02i%02i%02i' % (ID,
                                          int(c.dec.dms[0]),
                                          int(c.dec.dms[1]),
                                          int(c.dec.dms[2]))
            else:
                ID = '%s-%02i%02i%02i' % (ID,
                                          abs(int(c.dec.dms[0])),
                                          abs(int(c.dec.dms[1])),
                                          abs(int(c.dec.dms[2])))
            ID_all.append(ID)

        df_final['ID'] = ID_all
        col_names_ = df_final.columns.tolist()
        col_names_.remove('ID')
        col_names_.insert(0, 'ID')
        frames_all.append(df_final[col_names_])
        # break if only one CCD
        if CCD != 'all':
            break
    print '\n'
    print '___________________________________________________________________'

    data_frame = pd.concat(frames_all, axis=0)
    print data_frame.columns.values
    print data_frame.head()
    occ = occ_in

    print 'Number of objects with %i or more detections: %i' % (
        occ, data_frame.shape[0])
    print 'Number of objects with %i or more detections (drop_duplicates): %i'\
        % (occ, data_frame.drop_duplicates(subset=['internalID']).shape[0])

    data_frame.set_index('internalID', inplace=True)
    data_frame.sort_index(inplace=True)

    if data_frame.shape[0] == 0:
        print 'No objects to calculate statistics'
        return

    if CCD == 'all':
        if not os.path.exists('%s/tables/%s' % (jorgepath, field)):
            print "Creating field folder..."
            os.makedirs('%s/tables/%s' % (jorgepath, field))
        data_frame.to_csv('%s/tables/%s/%s_HiTS_n%i_table.csv' %
                          (jorgepath, field, field, int(occ)))
    else:
        data_frame.to_csv('%s/catalogues/%s/%s/%s_%s_HiTS_n%i_table.csv'
                          % (jorgepath, field, CCD, field,
                             CCD, int(occ)))


if __name__ == '__main__':
    startTime = datetime.now()
    field = ''
    CCD = 'all'
    occ = 5
    filters = 'gri'

    if len(sys.argv) == 1:
        print help
        sys.exit()

    # read command line option.
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'F:C:o:b:')
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
            filters = str(a)
        else:
            continue

    if CCD == 'all':
        CHIPS = np.loadtxt(
            '%s/info/ccds.txt' %
            (jorgepath), comments='#', dtype=str)
    else:
        CHIPS = CCD

    make_table(field, CHIPS, occ, filters)

    print 'It took', (datetime.now() - startTime), 'seconds'
    print '___________________________________________________________________'
