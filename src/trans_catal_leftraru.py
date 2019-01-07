# this script revise all the catalogues and convert pixel coordinates
# and flux to reference frame also conver to WSC and add MDJ to the name

import sys
import os
import glob
import getopt
import warnings
import numpy as np
import pickle
from misc_func_leftraru import *
from scipy.spatial import cKDTree
import scipy.stats as stats
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from empty_aperture_leftraru import empty_aperture_one_CCD
from astropy import units as u
from create_lightcurves_leftraru import run_crossmatch_lc
from datetime import datetime
from doastrometry import WCSsol
from astropy.coordinates import SkyCoord
from astropy import units as u

# import matplotlib.pyplot as plt
# import seaborn as sb
# sb.set(style="ticks", color_codes=True, context="poster")

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
thresh = 1.0      # threshold of catalog
minarea = 3      # min area for detection
deg2rad = 0.0174532925
rad2deg = 57.2957795
occ = 0.00001
g_ref = '02'
r_ref = '01'
dis_arcsec = 1.2
method = ['AUTO', 'APER', 'APER_1', 'APER_2', 'APER_3', 'APER_4']


def mask_ratio(x, y, n=3.):
    ratio = x / y
    ratio_MAD = MAD(ratio)
    mask = (
        ratio > np.median(ratio) -
        n *
        ratio_MAD) & (
        ratio < np.median(ratio) +
        n *
        ratio_MAD)
    return mask


def mask_diff(x, y, n=3.):
    diff = x - y
    diff_MAD = MAD(diff)
    mask = (
        diff > np.median(diff) -
        n *
        diff_MAD) & (
        diff < np.median(diff) +
        n *
        diff_MAD)
    return mask


def coordinates_correction(field, CCD, epochs, a_ref='02'):

    try:
        print 'Using SCAMP solution...'
        aux = fits.Header.fromtextfile(
            '%s/info/%s/%s/scamp_astrometry_%s_%s_%s.dat' %
            (jorgepath, field, CCD, field, CCD, a_ref))
        solpars = {}
        solpars['CRPIX'] = np.array([aux['CRPIX1'], aux['CRPIX2']])
        solpars['CRVAL'] = np.array([aux['CRVAL1'], aux['CRVAL2']])
        solpars['CD'] = np.array([[aux['CD1_1'], aux['CD1_2']],
                                  [aux['CD2_1'], aux['CD2_2']]])
        solpars['PV'] = np.array([[aux['PV1_0'], aux['PV1_1'], aux['PV1_2'],
                                   float(0.0), aux['PV1_4'], aux['PV1_5'],
                                   aux['PV1_6'], aux['PV1_7'], aux['PV1_8'],
                                   aux['PV1_9'], aux['PV1_10']],
                                  [aux['PV2_0'], aux['PV2_1'], aux['PV2_2'],
                                   float(0.0), aux['PV2_4'], aux['PV2_5'],
                                   aux['PV2_6'], aux['PV2_7'], aux['PV2_8'],
                                   aux['PV2_9'], aux['PV2_10']]])
    except :
        print 'SCAMP solution fails, using doastro sol...'
        if True:
            radec_file = np.sort(glob.glob(
                    "%s/SHARED/%s/%s/CALIBRATIONS/matchRADEC_%s_%s_*-02.npy" %
                    (astropath, field, CCD, field, CCD)), kind='mergesort')
            mRADEC_all = []
            for npy in radec_file:
                mRADEC_all.append(np.load(npy))
            mRADEC_all = np.asarray(mRADEC_all)
            mRADEC = mRADEC_all[np.argmin(mRADEC_all[:, 2])]
            solpars = {}
            solpars['CRPIX'] = np.array([mRADEC[5], mRADEC[6]])
            solpars['CRVAL'] = np.array([mRADEC[3], mRADEC[4]])
            solpars['CD'] = np.reshape(mRADEC[7:11], (2, 2))
            solpars['PV'] = np.reshape(
                mRADEC[26:], (int(mRADEC[11]), int(mRADEC[12])))
        else:
            solpars = pickle.load(open(
                "%s/SHARED/%s/%s/CALIBRATIONS/astrometry_%s_%s_%s.pkl"
                % (astropath, field, CCD, field, CCD, a_ref), 'rb'))
    print solpars
    epochs_to_remove = []

    print 'PIXEL AND WCS COVERTION...'

    ra_center, dec_center = [], []
    for epoch in epochs[:, 0]:
        print 'Loading epoch %s' % (epoch)

        # loading catalogs, fits and match fil

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_cat.dat" \
            % (jorgepath, field, CCD, field, CCD, epoch,
               str(thresh), str(minarea))
        if not os.path.exists(cata_file):
            print '\t\tNo catalog file: %s' % (cata_file)
            continue
        cata = Table.read(cata_file, format='ascii')
        cata = cata[(cata['FLUX_AUTO'] > 0)]
        # sys.exit()

        if epoch == a_ref:
            new_X = cata['X_IMAGE']
            new_Y = cata['Y_IMAGE']
            print '\t\tEpoch is reference, no transformation in pixel\
                   coordinates to apply'
        else:
            # Transformation of coordinates and

            print '\t\tApplying transformation to pix coordinates...'

            mtch_file = '%s/info/%s/%s/match_%s_%s_%s-%s.npy' % \
                        (jorgepath, field, CCD, field, CCD, epoch, a_ref)
            if not os.path.exists(mtch_file):
                print '\t\t\tNo jm-match file: %s' % (mtch_file),
                print 'passing to ff-match file'
                mtch_file = '%s/SHARED/%s/%s/CALIBRATIONS/match_%s_%s_%s-02.npy'\
                 % (astropath, field, CCD, field, CCD, epoch)
                if not os.path.exists(mtch_file):
                    print '\t\t\tNo jm either ff match files...'
                    epochs_to_remove.append(epoch)
                    continue
                else:
                    match_coef = np.load(mtch_file)[2:]
            else:
                match_coef = np.load(mtch_file)
            # print match_coef

            new_X = np.zeros(len(cata['X_IMAGE']))
            new_Y = np.zeros(len(cata['Y_IMAGE']))
            for k in range(len(cata)):
                new_X[k], new_Y[k] = applyinversetransformation(
                    match_coef[1], cata['X_IMAGE'][k],
                    cata['Y_IMAGE'][k], match_coef[2:])

            print '\t\tTransformation Done!'

        # Transformation from pix to RADEC ###

        print '\tApplying transformation from pix to RA-DEC'
        # New astrometric solution
        sol = WCSsol(0, 0, np.array(new_X), np.array(new_Y),
                     solpars["CRPIX"], solpars["CRVAL"], solpars["CD"],
                     solpars["PV"])
        RA2 = sol.RA()
        DEC2 = sol.DEC()
        ra_center.append(np.median(RA2))
        dec_center.append(np.median(DEC2))
        print '\tTransformation Done!'

        # cata.remove_columns(['X_IMAGE', 'Y_IMAGE', 'X_WORLD', 'Y_WORLD'])
        cata.add_column(Column(new_X, name='X_IMAGE_REF', unit=u.pix), index=1)
        cata.add_column(Column(new_Y, name='Y_IMAGE_REF', unit=u.pix), index=2)
        cata.add_column(Column(RA2, name='RA', unit=u.degree), index=3)
        cata.add_column(Column(DEC2, name='DEC', unit=u.degree), index=4)

        print '\tNew columns: ', cata.colnames

        new_cata_file = cata_file.replace('cat.dat', 'final-scamp.dat')
        cata.write(new_cata_file, format='ascii.commented_header',
                   delimiter='\t', overwrite=True)

        print '_______________________________________________________________'
    print [np.median(ra_center), np.median(dec_center)]
    print 'Transformation finish in all epochs'
    print '###################################################################'

    return epochs_to_remove, [np.median(ra_center), np.median(dec_center)]


def correct_pix_correlation(field, CCD, epochs):

    print 'CORRECTING ERRORS FOR PIXEL CORRELATION'

    for epoch in epochs[:, 0]:
        print 'Loading epoch %s' % (epoch)

        # loading catalogs, fits and match fil

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
            (jorgepath, field, CCD, field, CCD, epoch,
             str(thresh), str(minarea))
        if not os.path.exists(cata_file):
            print '\t\tNo catalog file: %s' % (cata_file)
            continue
        cata = Table.read(cata_file, format='ascii')
        cata = cata[(cata['FLUX_AUTO'] > 0)]
        print len(cata)

        # correction to flux errors by pixel correlat
        imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (
            astropath, field, CCD, field, CCD, epoch)
        if not os.path.exists(imag_file):
            print '\t\tNo image file: %s' % (imag_file)
            continue

        hdufits = fits.open(imag_file)
        GAIN = float(hdufits[0].header['GAINA'])

        n_aper = 500
        pix_corr_file = '%s/info/%s/%s/pixcorrcoef_%s_%s_%s_%i.npy' % \
            (jorgepath, field, CCD, field, CCD, epoch, n_aper)
        if not os.path.exists(pix_corr_file):
            print '\t\tMeasuring pixel correlation...'
            sigma1, alpha, beta = empty_aperture_one_CCD(
                field, CCD, epoch, n_aper=n_aper, verbose=False)
            auxxx = np.array((sigma1, alpha, beta))
            np.save(pix_corr_file, auxxx)
        else:
            print '\t\Loading pixel correlation...'
            temp = np.load(pix_corr_file)
            sigma1, alpha, beta = temp[0], temp[1], temp[2]
            print 'sigma1 = %f' % (sigma1)
            print 'alpha = %f' % (alpha)
            print 'beta = %f' % (beta)

        for k, met in enumerate(method):
            print 'Type of photometry: %s' % (met)
            if sigma1 is not None:
                if met == 'AUTO':
                    num_pix = np.pi * cata['A_IMAGE'] * \
                        cata['B_IMAGE'] * cata['KRON_RADIUS']**2
                else:
                    num_pix = np.pi * \
                        (float(k) * np.median(cata['FWHM_IMAGE']))**2
                new_flux_err = np.sqrt(sigma1**2 * alpha**2 *
                                       num_pix**beta +
                                       cata['FLUX_%s' % (met)] / GAIN)
            else:
                new_flux_err = cata['FLUXERR_%s' % (met)]

            cata.add_column(
                Column(
                    new_flux_err,
                    name='FLUXERR_%s_COR' %
                    (met),
                    unit=u.count),
                index=7)

        print '\tNew columns: ', cata.colnames
        cata.write(cata_file, format='ascii.commented_header',
                   delimiter='\t', overwrite=True)

        print '_______________________________________________________________'
    print 'Transformation finish in all epochs'
    print '###################################################################'


def flux_correction(field, CCD, epochs, p_ref='02', pix_corr=True):

    print 'AFLUX CORRECTION'
    epochs_to_remove = []

    for epoch in epochs[:, 0]:
        print 'Loading epoch %s' % (epoch)

        # loading catalogs, fits and match fil

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % (jorgepath, field, CCD, field, CCD, epoch, str(thresh), str(minarea))
        if not os.path.exists(cata_file):
            print '\t\tNo catalog file: %s' % (cata_file)
            continue
        cata = Table.read(cata_file, format='ascii')
        cata = cata[(cata['FLUX_AUTO'] > 0)]
        # sys.exit()

        for k, met in enumerate(method):
            print 'Tipe of photometry: %s' % (met)
            if epoch == p_ref:
                new_FLUX_AUTO = cata['FLUX_%s' % (met)]
                new_FLUXERR_AUTO = cata['FLUXERR_%s' % (met)]
                print '\t\tEpoch is reference, no relative transformation'
            else:
                # aflux correction ####
                print '\t\tApplying aflux'

                aflx_file = '%s/info/%s/%s/aflux_%s_%s_%s-%s.npy' % \
                            (jorgepath, field, CCD, field, CCD, epoch, p_ref)
                if not os.path.exists(aflx_file):
                    print '\t\t\tNo jm-match file: %s' % (aflx_file),
                    print 'passing to ff-match file'
                    mtch_file = '%s/SHARED/%s/%s/CALIBRATIONS/match_%s_%s_%s-02.npy'\
                     % (astropath, field, CCD, field, CCD, epoch)
                    if not os.path.exists(mtch_file):
                        print '\t\t\tNo jm either ff match files...'
                        epochs_to_remove.append(epoch)
                        continue
                    else:
                        aflux_coef = np.load(mtch_file)[:2]
                else:
                    aflux_coef = np.load(aflx_file)

                new_FLUX_AUTO = cata['FLUX_%s' % (met)] / aflux_coef[0]
                if pix_corr:
                    new_FLUXERR_AUTO = np.sqrt((
                        cata['FLUXERR_%s_COR' % (met)] / cata['FLUXERR_%s' %
                                                              (met)])**2 +
                                               (aflux_coef[1] /
                                                aflux_coef[0])**2)
                else:
                    new_FLUXERR_AUTO = np.sqrt((
                        cata['FLUXERR_%s'% (met)] / cata['FLUXERR_%s' %
                                                         (met)])**2 +
                                               (aflux_coef[1] /
                                                aflux_coef[0])**2)

                print '\t\tTransformation Done!'

            cata.add_column(Column(new_FLUX_AUTO, name='FLUX_%s_AFLUX' % (met),
                                   unit=u.degree), index=8)
            cata.add_column(Column(new_FLUXERR_AUTO, name='FLUXERR_%s_AFLUX' %
                                   (met), unit=u.degree), index=9)

        print '\tNew columns: ', cata.colnames
        cata.write(cata_file, format='ascii.commented_header',
                   delimiter='\t', overwrite=True)

        print '_______________________________________________________________'
    print 'Transformation finish in all epochs'
    print '###################################################################'

    return epochs_to_remove


def standard_calibration_DECam(
        field, CCD, epochs, FILTER, aflux=True, p_ref='02'):

    print 'STD CALIBRATION DECam'
    imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (
        astropath, field, CCD, field, CCD, p_ref)
    if not os.path.exists(imag_file):
        print '\tNo reference image file: %s' % (imag_file)
        return
    hdufits = fits.open(imag_file)
    AIRMASS = float(hdufits[0].header['AIRMASS'])

    for epoch in epochs[:, 0]:
        print 'Loading epoch %s' % (epoch)

        # loading catalogs, fits and match f
        imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (
            astropath, field, CCD, field, CCD, epoch)
        if not os.path.exists(imag_file):
            print '\t\tNo image file: %s' % (imag_file)
            continue

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % (jorgepath, field, CCD, field, CCD, epoch, str(thresh), str(minarea))
        if not os.path.exists(cata_file):
            print '\t\tNo catalog file: %s' % (cata_file)
            continue
        cata = Table.read(cata_file, format='ascii')
        # sys.exit()

        # Reading image and header

        hdufits = fits.open(imag_file)
        EXP_TIME = float(hdufits[0].header['EXPTIME'])
        CCDID = int(hdufits[0].header['CCDNUM'])
        CTE_file = np.loadtxt('%s/info/zeropoint/psmFitDES-mean-%s.csv'
                              % (jorgepath, FILTER), delimiter=',', skiprows=1,
                              usecols=[4, 5, 6, 7, 8, 9, 10, 11, 19],
                              dtype=str)
        Ag = float(CTE_file[CCDID - 1][2])
        err_Ag = float(CTE_file[CCDID - 1][3])
        Kg = float(CTE_file[CCDID - 1][6])
        err_Kg = float(CTE_file[CCDID - 1][7])

        # Transformation from ADU to magnitudes and er
        for k, met in enumerate(method):
            print 'Tipe of photometry: %s' % (met)
            if aflux:
                flx = cata['FLUX_%s_AFLUX' % (met)]
                e_flx = cata['FLUXERR_%s_AFLUX' % (met)]
            else:
                flx = cata['FLUX_%s' % (met)]
                e_flx = cata['FLUXERR_%s_COR' % (met)]

            print '\tApplying transformation from ADU to MAG'
            new_mag_auto_za = -2.5 * \
                np.log10(flx) + 2.5 * np.log10(EXP_TIME) - Ag - Kg * AIRMASS
            new_mag_auto_za_err = np.sqrt(((2.5 * e_flx) /
                                           (flx * np.log(10))) ** 2 +
                                          err_Ag ** 2 +
                                          (AIRMASS * err_Kg) ** 2)

            print '\tTransformation Done!'

            cata.add_column(Column(new_mag_auto_za, name='MAG_%s_ZA' % (met),
                                   unit=u.mag), index=12)
            cata.add_column(Column(new_mag_auto_za_err, name='MAGERR_%s_ZA' %
                                   (met), unit=u.mag), index=13)

        print '\tNew columns: ', cata.colnames
        cata.write(cata_file, format='ascii.commented_header',
                   delimiter='\t', overwrite=True)

        print '_______________________________________________________________'
    print 'Transformation finish in all epochs'
    print '###################################################################'

    return


def get_color_table(field, CCD, epochs_f1, epochs_f2, filter2='r', ref_f2='01'):

    warnings.filterwarnings("ignore")
    print 'GETING COLORS'

    ##########################################################################
    # Filter 1
    ##########################################################################

    epoch_c = []

    for epoch in epochs_f1:
        print '\tEpoch %s' % epoch[0]

        # catalogues

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
            (jorgepath, field, CCD, field, CCD,
             epoch[0], str(thresh), str(minarea))
        if not os.path.exists(cata_file):
            print '\t\tNo catalog file: %s' % (cata_file)
            continue
        cata = Table.read(cata_file, format='ascii')

        # epoch_c has all the catalogues, each element of epoch_c contain the
        # catalogue of a given epoch
        epoch_c.append(cata)

        # INFO of epochs

    if len(epoch_c) == 0:
        print '\t\t...No catalogues'
        print '_______________________________________________________________'
        return None

    indx_file = "%s/catalogues/%s/%s/temp_%s_%s_%s_master_index.txt" % (
        jorgepath, field, CCD, field, CCD, 'g')
    if not os.path.exists(indx_file):
        print '\t\t...No master index file: %s' % (indx_file)
        return None
    master_cat = np.loadtxt(indx_file, comments='#', dtype='i4')

    print '\tNumber of epochs: %i, effective: %i' % (len(epochs_f1),
                                                     len(master_cat[0]))

    ##########################################################################
    # Filter 2
    ##########################################################################

    epoch_c_f2 = []

    print 'Loading catalogues files'
    for epoch in epochs_f2:
        print '\tEpoch %s' % epoch[0]

        # catalogues ###############

        cata_file_f2 = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
            (jorgepath, field, CCD, field, CCD,
             epoch[0], str(thresh), str(minarea))
        if not os.path.exists(cata_file_f2):
            print '\t\tNo catalog file: %s' % (cata_file_f2)
            continue
        cata_f2 = Table.read(cata_file_f2, format='ascii')

        if epoch[0] == ref_f2:
            tree = cKDTree(np.transpose(np.array((cata_f2['X_IMAGE_REF'],
                                                  cata_f2['Y_IMAGE_REF']))))
            tree_len = len(cata_f2)

        # epoch_c has all the catalogues, each element of epoch_c contain the
        # catalogue of a given epoch
        epoch_c_f2.append(cata_f2)

    if len(epoch_c_f2) == 0:
        print '\t\t...No catalogues'
        print '_______________________________________________________________'
        return None

    indx_file_f2 = "%s/catalogues/%s/%s/temp_%s_%s_%s_master_index.txt" % (
        jorgepath, field, CCD, field, CCD, filter2)
    if not os.path.exists(indx_file_f2):
        print '\t\t...No master index file: %s' % (indx_file_f2)
        return None
    master_cat_f2 = np.loadtxt(indx_file_f2, comments='#', dtype='i4')

    print '\tNumber of epochs (%s): %i, effective: %i' % (filter2,
                                                          len(epochs_f2),
                                                          len(master_cat_f2[0]))

    ##########################################################################
    # objects > .8
    ##########################################################################

    median_mag_g, std_mag_g, median_err_g = [], [], []
    median_mag_r, std_mag_r, median_err_r = [], [], []
    pix_coord = []
    count_no_match = 0

    for obj in range(len(master_cat)):

        num_obs = np.where(master_cat[obj, :] > 0)
        if len(num_obs[0]) / float(len(epochs_f1)) >= occ:
            err_lc, mag_lc = [], []
            pix_x, pix_y = [], []
            for tim in range(len(master_cat[obj, :])):
                pos = master_cat[obj, tim]
                if pos > 0:
                    mag_lc.append(epoch_c[tim]['MAG_AUTO_ZA'][pos])
                    err_lc.append(epoch_c[tim]['MAGERR_AUTO_ZA'][pos])

                    pix_x.append(epoch_c[tim]['X_IMAGE_REF'][pos])
                    pix_y.append(epoch_c[tim]['Y_IMAGE_REF'][pos])

            median_mag_g.append(np.median(mag_lc))
            std_mag_g.append(np.std(mag_lc))
            median_err_g.append(np.median(err_lc))

            pix_coord.append(np.array([np.median(pix_x), np.median(pix_y)]))

            indx_cm = tree.query(pix_coord[-1], k=1, distance_upper_bound=10)[1]
            if indx_cm == tree_len:
                # print 'No match in filter r'
                count_no_match += 1
                median_mag_r.append(None)
                std_mag_r.append(None)
                median_err_r.append(None)
            else:
                indx_aux = np.where(master_cat_f2[:, 0] == indx_cm)
                obj_master_cat_f2 = master_cat_f2[indx_aux][0]
                err_lc, mag_lc = [], []
                for tim in range(len(obj_master_cat_f2)):
                    pos = master_cat_f2[indx_aux, tim][0]
                    if pos > 0:
                        mag_lc.append(epoch_c_f2[tim]['MAG_AUTO_ZA'][pos])
                        err_lc.append(epoch_c_f2[tim]['MAGERR_AUTO_ZA'][pos])

                median_mag_r.append(np.median(mag_lc))
                std_mag_r.append(np.std(mag_lc))
                median_err_r.append(np.median(err_lc))

    median_mag_g = np.asarray(median_mag_g)
    std_mag_g = np.asarray(std_mag_g)
    median_err_g = np.asarray(median_err_g)
    pix_coord = np.asarray(pix_coord)

    median_mag_r = np.asarray(median_mag_r)
    std_mag_r = np.asarray(std_mag_r)
    median_err_r = np.asarray(median_err_r)

    print '\tNumber of objects: %i' % (len(median_mag_g))
    print '\tNumber of objects without match in %s: %i' % (filter2,
                                                           count_no_match)
    print '___________________________________________________________________'

    keywords = ['X', 'Y', 'Median_g', 'STD_g', 'Median_err_g',
                'Median_%s' % filter2, 'STD_%s' % filter2,
                'Median_err_%s' % filter2]
    table = Table([pix_coord[:, 0], pix_coord[:, 1],
                   median_mag_g, std_mag_g, median_err_g,
                   median_mag_r, std_mag_r, median_err_r],
                  names=keywords, meta={'name': 'Catalogue %s' % (field)},
                  dtype=('float', 'float', 'float', 'float',
                         'float', 'float', 'float', 'float'))
    print 'Color table finish'
    print '###################################################################'
    return table


def color_calibration_DECam(field, CCD, epochs, color_table):

    print 'COLOR TERM CORRECTION DECam'

    for epoch in epochs[:, 0]:
        print 'Loading epoch %s' % (epoch)

        # loading catalogs, fits and match fi
        imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (
            astropath, field, CCD, field, CCD, epoch)
        if not os.path.exists(imag_file):
            print '\t\tNo image file: %s' % (imag_file)
            continue

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
            (jorgepath, field, CCD, field, CCD, epoch, str(thresh), str(minarea))
        if not os.path.exists(cata_file):
            print '\t\tNo catalog file: %s' % (cata_file)
            continue
        cata = Table.read(cata_file, format='ascii')
        print '\tLength of catalogue: %i' % (len(cata))
        # sys.exit()

        # Reading image and header #####

        print '\tApplying color correction'
        hdufits = fits.open(imag_file)

        FILTER = hdufits[0].header['FILTER'][0]
        CCDID = int(hdufits[0].header['CCDNUM'])
        CTE_file = np.loadtxt('%s/info/zeropoint/psmFitDES-mean-%s.csv'
                              % (jorgepath, FILTER), delimiter=',', skiprows=1,
                              usecols=[4, 5, 6, 7, 8, 9, 10, 11, 19],
                              dtype=str)
        bg = float(CTE_file[CCDID - 1][4])
        err_bg = float(CTE_file[CCDID - 1][4])
        g_r0 = float(CTE_file[CCDID - 1][8])

        # open color table ##########

        tree_XY_color = cKDTree(np.transpose([color_table['X'],
                                              color_table['Y']]))
        XY_cata = np.transpose([cata['X_IMAGE_REF'], cata['Y_IMAGE_REF']])
        superpos_ind = tree_XY_color.query(
            XY_cata, k=1, distance_upper_bound=4)[1]
        # dice los obj de single epoch encontrados en color
        index_filter = (superpos_ind < len(color_table))
        # indices de los correspondientes colores
        index = superpos_ind[index_filter]

        # print len(color_table['Median_g'][index])
        # print len(cata['FLUX_AUTO'])

        if FILTER == 'g':
            if len(color_table['Median_g'][index]) != len(cata['FLUX_AUTO']):
                color_mask_aux = []
                g_r = []
                for l, bol in enumerate(index_filter):
                    if bol:
                        position = superpos_ind[l]
                        if color_table['Median_r'][position] != 0.:
                            color_mask_aux.append(True)
                            g_r.append(color_table['Median_g'][position] -
                                       color_table['Median_r'][position])
                        else:
                            color_mask_aux.append(False)
                            g_r.append(None)
                    else:
                        color_mask_aux.append(False)
                        g_r.append(None)
                color_mask = np.array(color_mask_aux)
                g_r = np.array(g_r)
            else:
                color_mask = (color_table['Median_r'][index] != 0.)
                g_r = np.array(color_table['Median_g'][index] -
                               color_table['Median_r'][index])

        if FILTER == 'r':
            color_mask_aux = []
            g_r = []
            for l, bol in enumerate(index_filter):
                if bol:
                    position = superpos_ind[l]
                    if color_table['Median_r'][position] != 0.:
                        color_mask_aux.append(True)
                        g_r.append(color_table['Median_g'][position] -
                                   color_table['Median_r'][position])
                    else:
                        color_mask_aux.append(False)
                        g_r.append(None)
                else:
                    color_mask_aux.append(False)
                    g_r.append(None)
            color_mask = np.array(color_mask_aux)
            g_r = np.array(g_r)

        # Transformation from ADU to magnitudes and
        for k, met in enumerate(method):
            print 'Tipe of photometry: %s' % (met)
            new_mag_auto_zac = np.zeros(len(cata['MAG_%s_ZA' % (met)]))
            new_mag_auto_zac_err = np.zeros(len(cata['MAG_%s_ZA' % (met)]))
            for kk in range(len(cata['FLUX_%s' % (met)])):
                if color_mask[kk]:
                    new_mag_auto_zac[kk] = cata['MAG_%s_ZA' % (met)][kk] - \
                        bg * (g_r[kk] -g_r0)
                    new_mag_auto_zac_err[kk] = np.sqrt((cata['MAGERR_%s_ZA' %
                                                             (met)][kk])**2 +
                                                       (g_r[kk] * err_bg)**2)

                else:
                    new_mag_auto_zac[kk] = cata['MAG_%s_ZA' % (met)][kk]
                    new_mag_auto_zac_err[kk] = cata['MAGERR_%s_ZA' % (met)][kk]

            print '\tColor correction Done!'

            cata.add_column(Column(new_mag_auto_zac, name='MAG_%s_ZAC' % (met),
                                   unit=u.mag), index=14)
            cata.add_column(Column(new_mag_auto_zac_err, name='MAGERR_%s_ZAC' %
                                   (met), unit=u.mag), index=15)

        print '\tNew columns: ', cata.colnames
        cata.write(cata_file, format='ascii.commented_header',
                   delimiter='\t', overwrite=True)

        print '______________________________________________________________'
    print 'Transformation finish in all epochs'
    print '##################################################################'

    return


def zero_point_PS(field, CCD, epochs, FILTER, center=[0, 0],
                  aflux=False, Plot=False):

    print 'ABSOLUTE CORRECTION WITH PanSTARRS'

    PS1_path = '%s/PS1/%s_PS1_%s.vot' % (jorgepath, field, CCD)
    if not os.path.exists(PS1_path):
        print '\tNo PanSTARRS catalog for this Field/CCD...'
        print '\tQuering PS1 to retrieve data...'
        # quering PS1 with CCD center and .2 deg radius
        # (720 arcsec, then 1440 arcsec diagonal,
        # CCD is 1236 arcsec diagonal, enougth room for pointing error)
        PS1_data = panstarrs_query(ra_deg=center[0],
                                   dec_deg=center[1],
                                   rad_deg=0.2, mindet=2)
        PS1_data.write(PS1_path, format='votable')
    else:
        PS1_data = Table.read(PS1_path)

    PS1_data = PS1_data.to_pandas()
    print '\tShape of PS1             ', PS1_data.shape
    PS1_data_good_Q = PS1_data.query(
        '%sMeanKronMag > 16 and %sMeanKronMag < 21' %
        (FILTER, FILTER))
    print '\tShape of PS1 good quality', PS1_data_good_Q.shape

    PS1_coord = SkyCoord(ra=PS1_data_good_Q.raMean.values,
                         dec=PS1_data_good_Q.decMean.values,
                         frame='icrs', unit=u.degree)

    zp_PS1_epochs = []
    epo_no_cata = []

    for kk, epoch in enumerate(epochs[:, 0]):
        print 'Loading epoch %s' % (epoch)

        imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (
            astropath, field, CCD, field, CCD, epoch)
        if not os.path.exists(imag_file):
            print '\t\tNo image file: %s' % (imag_file)
            zp_PS1_epochs.append([None, None, None])
            continue

        hdufits = fits.open(imag_file)
        EXP_TIME = float(hdufits[0].header['EXPTIME'])

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
            (jorgepath, field, CCD, field, CCD, epoch, str(thresh),
             str(minarea))
        if not os.path.exists(cata_file):
            print '\t\tNo catalog file: %s' % (cata_file)
            zp_PS1_epochs.append([None, None, None])
            epo_no_cata.append(epoch)
            continue
        cata = Table.read(cata_file, format='ascii')

        HiTS_coord = SkyCoord(ra=cata['RA'], dec=cata['DEC'], frame='icrs',
                              unit=u.degree)
        idx, d2d, d3d = HiTS_coord.match_to_catalog_3d(PS1_coord)
        mask_less2 = (d2d.arcsec < dis_arcsec)
        print '\tNumber of matches within %.3f arcsec: %i' % (
            dis_arcsec, len(idx[mask_less2]))

        match_PS1 = PS1_data_good_Q.iloc[idx[mask_less2]]
        match_HiTS = cata[mask_less2]

        prev_zp = []
        for k, met in enumerate(method):
            print 'Tipe of photometry: %s' % (met)
            if aflux:
                flux_auto = match_HiTS['FLUX_%s_AFLUX' % (met)]
            else:
                flux_auto = match_HiTS['FLUX_%s' % (met)]

            m_ps1 = -2.5 * np.log10(flux_auto) + 2.5 * np.log10(EXP_TIME)

            if met == 'AUTO':
                col_filter_mag = '%sMeanKronMag' % (FILTER)
            else:
                col_filter_mag = '%sMeanApMag' % (FILTER)
            print '\tColumn name on PS with photometry: ', col_filter_mag

            phot_dif = match_PS1[col_filter_mag].values - m_ps1
            mask_d = mask_diff(match_PS1[col_filter_mag].values, m_ps1, n=3.)
            # ZP from difference after mask
            zp_ps1 = np.median(phot_dif[mask_d])
            # error of ZP from std of difference
            e_zp_ps1 = np.std(phot_dif[mask_d])

            # bootstrap over difference to improve ZP and e_ZP
            if len(phot_dif[mask_d]) != 0:
                nparam_density = stats.kde.gaussian_kde(phot_dif[mask_d])
                aux = np.linspace(zp_ps1 - .5, zp_ps1 + .5, 200)
                mode = aux[np.argmax(nparam_density(aux))]

                boostrap = np.zeros(100)
                boostrap_kde = np.zeros(100)
                for k in range(100):
                    boostrap[k] = np.median(np.random.choice(
                        phot_dif[mask_d], size=len(phot_dif[mask_d]),
                        replace=True))
                    nparam_density = stats.kde.gaussian_kde(np.random.choice(
                        phot_dif[mask_d], size=len(phot_dif[mask_d]),
                        replace=True))
                    boostrap_kde[k] = aux[np.argmax(nparam_density(aux))]
                zp_ps1_boost = np.median(boostrap)
                e_zp_ps1_boost = np.std(boostrap)
                zp_ps1_boost_kde = np.median(boostrap_kde)
                e_zp_ps1_boost_kde = np.std(boostrap_kde)
                zp_PS1_epochs.append([zp_ps1, e_zp_ps1, e_zp_ps1_boost])
                prev_zp = [zp_ps1, e_zp_ps1, e_zp_ps1_boost]
                prev_kde = [mode, e_zp_ps1_boost_kde]
            else:
                [zp_ps1, e_zp_ps1, e_zp_ps1_boost] = [
                    prev_zp[0], prev_zp[1], prev_zp[2]]
                [mode, e_zp_ps1_boost_kde] = [prev_kde[0], prev_kde[1]]

            print '\tPS       : ZP = %f +- %f' % (zp_ps1, e_zp_ps1)
            print '\tBootstrap: ZP = %f +- %f' % (zp_ps1_boost, e_zp_ps1_boost)
            print '\tKDE      : ZP = %f +- %f' % (mode, e_zp_ps1_boost_kde)

            zp_ps1 = mode

            if aflux:
                flux_auto_all = cata['FLUX_%s_AFLUX' % (met)]
                fluxerr_auto_all = cata['FLUXERR_%s_AFLUX' % (met)]
            else:
                flux_auto_all = cata['FLUX_%s' % (met)]
                fluxerr_auto_all = cata['FLUXERR_%s_COR' % (met)]

            mag_HiTS_ps1corr = -2.5 * \
                np.log10(flux_auto_all) + 2.5 * np.log10(EXP_TIME) + zp_ps1
            e_mag_HiTS_ps1corr = np.sqrt((2.5 * fluxerr_auto_all /
                                          (flux_auto_all * np.log(10)))**2 +
                                         e_zp_ps1_boost**2)

            cata.add_column(Column(mag_HiTS_ps1corr, name='MAG_%s_ZP' % (met),
                                   unit=u.mag), index=14)
            cata.add_column(Column(e_mag_HiTS_ps1corr, name='MAGERR_%s_ZP' %
                                   (met), unit=u.mag), index=15)
            if met == 'AUTO':
                ZP_PS = np.array([zp_ps1, e_zp_ps1, e_zp_ps1_boost],
                                 dtype={'names': ['ZP_PS', 'std_ZP_PS',
                                                  'std_ZP_PS_bootsrap'],
                                        'formats': ['float', 'float',
                                                    'float']})
                ZP_PS_name = '%s/info/%s/%s/ZP_%s_PS_%s_%s_%s.npy' % \
                    (jorgepath, field, CCD, met, field, CCD, epoch)
                np.save(ZP_PS_name, ZP_PS)

        print '\tNew columns: ', cata.colnames
        cata.write(cata_file, format='ascii.commented_header',
                   delimiter='\t', overwrite=True)

        if Plot:
            fig, ax = plt.subplots(1, 2, sharex=True, figsize=(10, 4))

            ax[0].axhline(zp_ps1, c='k', ls='-', lw=1)
            ax[0].axhline(zp_ps1 + 3. * np.std(phot_dif), c='k', ls='--', lw=1)
            ax[0].axhline(zp_ps1 - 3. * np.std(phot_dif), c='k', ls='--', lw=1)
            ax[0].scatter(match_PS1['col_filter_mag'].values,
                          phot_dif, marker='.', alpha=.8, lw=0, s=50, c='b')
            ax[0].scatter(match_PS1['col_filter_mag'].values[mask_d],
                          phot_dif[mask_d], marker='.', alpha=.8, lw=0,
                          s=50, c='g', label='N: %i' % (len(phot_dif[mask_d])))
            ax[0].legend(loc='best')
            ax[0].set_ylabel(r'$(PS1 - HiTS)_{%s}$' % FILTER)
            ax[0].set_xlabel(r'$PS1$')
            ax[0].set_title('ZP = %f +- %f/%f' % (zp_ps1, e_zp_ps1,
                                                  e_zp_ps1_boost), fontsize=15)
            ax[0].set_xlim(14, 22)

            ax[1].axhline(0, c='k', ls='-', lw=1)
            ax[1].errorbar(match_PS1['col_filter_mag'].values,
                           dif_HiTS_PS_dict[ccd],
                           yerr=e_dif_HiTS_PS_dict[ccd], fmt='.', alpha=.8,
                           lw=2, ms=7, c='b', label='%f' %
                           (np.median(mag_HiTS_ps1corr -
                                      match_PS1['col_filter_mag'].values)))
            ax[1].legend(loc='lower left')
            ax[1].set_ylim(-.5, .5)
            name = '%s_%s_%s_ZP_PS1' % (field, CCD, epo)
            plt.savefig(
                '%s/figures/%s/%s/%s.png' %
                (jorgepath, field, CCD, name), dpi=300)

        print '_______________________________________________________________'

    zp_PS1_epochs = np.array(zp_PS1_epochs)
    if Plot:
        plt.figure(figsize=(7, 5))
        plt.errorbar(epochs[:, 1], zp_PS1_epochs[:, 0],
                     yerr=zp_PS1_epochs[:, 2], fmt='.', lw=2, ms=7, c='k')
        plt.xlabel('MJD')
        plt.ylabel(r'$ZP_{%s}$' % FILTER)
        plt.grid(True)
        plt.savefig('%s/figures/%s/%s/%s_%s_%s_ZP_evo.png' %
                    (jorgepath, field, CCD, field, CCD, FILTER), dpi=300)

    print 'Transformation finish in all epochs'
    print '###################################################################'


if __name__ == '__main__':

    startTime = datetime.now()
    field = 'Blind15A_01'
    CCD = 'N1'

    g_ref_p = '02'
    g_ref_a = '02'

    r_ref_p = '01'
    r_ref_a = '02'

    u_ref_p = '02'
    u_ref_a = '02'

    i_ref_p = '02'
    i_ref_a = '02'

    # read command line option.
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'h:F:C:')
    except getopt.GetoptError as err:
        print help
        sys.exit()

    for o, a in optlist:
        if o in ('-h'):
            print help
            sys.exit()
        elif o in ('-F'):
            field = str(a)
        elif o in ('-C'):
            CCD = str(a)
        else:
            continue

    print 'Field: ', field
    print 'CCD: ', CCD

    if field[:8] == 'Blind15A':

        filter_g = 'g'
        filter_r = 'r'
        filter_i = 'i'
        filter_u = 'u'

        # filter g
        epochs_g_file = '%s/info/%s/%s_epochs_%s.txt' % (
            jorgepath, field, field, filter_g)
        if not os.path.exists(epochs_g_file):
            print 'No epochs file: %s' % (epochs_g_file)
            sys.exit()
        epochs_g_org = np.loadtxt(epochs_g_file, comments='#', dtype=str)
        epochs_g = np.loadtxt(epochs_g_file, comments='#', dtype=str)

        # filter r
        epochs_r_file = '%s/info/%s/%s_epochs_%s.txt' % (
            jorgepath, field, field, filter_r)
        if not os.path.exists(epochs_r_file):
            print 'No epochs file: %s' % (epochs_r_file)
            sys.exit()
        epochs_r = np.loadtxt(epochs_r_file, comments='#', dtype=str)

        # filter i
        epochs_i_file = '%s/info/%s/%s_epochs_%s.txt' % (
            jorgepath, field, field, filter_i)
        if not os.path.exists(epochs_i_file):
            print 'No epochs file: %s' % (epochs_i_file)
        epochs_i = np.loadtxt(epochs_i_file, comments='#', dtype=str)
        if epochs_i.shape == (2,):
            epochs_i = epochs_i.reshape(1, 2)
        if len(epochs_i) == 0:
            i_bool = False
        else:
            i_bool = True
            i_ref_p = epochs_i[0][0]

        # filter u
        epochs_u_file = '%s/info/%s/%s_epochs_%s.txt' % (
            jorgepath, field, field, filter_u)
        if not os.path.exists(epochs_u_file):
            print 'No epochs file: %s' % (epochs_u_file)
        epochs_u = np.loadtxt(epochs_u_file, comments='#', dtype=str)
        if epochs_u.shape == (2,):
            epochs_u = epochs_u.reshape(1, 2)
        if len(epochs_u) == 0:
            u_bool = False
        else:
            u_bool = True
            u_ref_p = epochs_u[0][0]

        # Convert positions and convert flux to reference epoch
        print 'Filter %s' % filter_g
        to_remove_g, radec_center = coordinates_correction(field, CCD,
                                                           epochs_g,
                                                           a_ref=g_ref_a)
        print 'Filter %s' % filter_r
        to_remove_r, _ = coordinates_correction(field, CCD, epochs_r,
                                                a_ref=r_ref_a)
        if i_bool:
            print 'Filter %s' % filter_i
            to_remove_i, _ = coordinates_correction(field, CCD, epochs_i,
                                                    a_ref=i_ref_a)
        if u_bool:
            print 'Filter %s' % filter_u
            to_remove_u, _ = coordinates_correction(field, CCD, epochs_u,
                                                    a_ref=u_ref_a)

        print 'Before:\n ', epochs_g
        print 'Remove:\n ', to_remove_g
        if len(to_remove_g) != 0:
            epochs_g = np.delete(epochs_g,
                                 np.where(epochs_g == to_remove_g)[0], axis=0)
        print 'After:\n ', epochs_g

        # Correct errors for correlation in pixels
        print 'Filter %s' % filter_g
        correct_pix_correlation(field, CCD, epochs_g)
        print 'Filter %s' % filter_r
        correct_pix_correlation(field, CCD, epochs_r)
        if i_bool:
            print 'Filter %s' % filter_i
            correct_pix_correlation(field, CCD, epochs_i)
        if u_bool:
            print 'Filter %s' % filter_u
            correct_pix_correlation(field, CCD, epochs_u)

        # aflux correction
        print 'Filter %s' % filter_g
        to_remove_g = flux_correction(field, CCD, epochs_g, p_ref=g_ref_p,
                                      pix_corr=True)
        print 'Filter %s' % filter_r
        to_remove_r = flux_correction(field, CCD, epochs_r, p_ref=r_ref_p,
                                      pix_corr=True)
        if i_bool:
            print 'Filter %s' % filter_i
            to_remove_i = flux_correction(field, CCD, epochs_i, p_ref=i_ref_p,
                                          pix_corr=True)
        if u_bool:
            print 'Filter %s' % filter_u
            to_remove_u = flux_correction(field, CCD, epochs_u, p_ref=u_ref_p,
                                          pix_corr=True)

        # apply ZeroPoint + airmass correction
        print 'Filter %s' % filter_g
        standard_calibration_DECam(field, CCD, epochs_g, filter_g,
                                   aflux=True, p_ref=g_ref_p)
        print 'Filter %s' % filter_r
        standard_calibration_DECam(field, CCD, epochs_r, filter_r,
                                   aflux=True, p_ref=r_ref_p)
        if i_bool:
            print 'Filter %s' % filter_i
            standard_calibration_DECam(field, CCD, epochs_i, filter_i,
                                       aflux=True, p_ref=i_ref_p)
        if u_bool:
            print 'Filter %s' % filter_u
            try:
                standard_calibration_DECam(field, CCD, epochs_u, filter_u,
                                           aflux=True, p_ref=u_ref_p)
            except KeyError:
                print 'KeyError: FLUX_AUTO_AFLUX'
                standard_calibration_DECam(field, CCD, epochs_u, filter_u,
                                           aflux=False, p_ref=u_ref_p)

        ##### Skiping color correction from DECam...
        # # Index matrix
        # print 'Creating index matrix...'
        # print 'Filter %s' % filter_g
        # run_crossmatch_lc(field, CCD, filter_g, kind='temp')
        # print 'Filter %s' % filter_r
        # run_crossmatch_lc(field, CCD, filter_r, kind='temp')
        # if i_bool and epochs_i.shape[0] > 1:
        #     print 'Filter %s' % filter_i
        #     run_crossmatch_lc(field, CCD, filter_i, kind='temp')
        # if u_bool and epochs_u.shape[0] > 1:
        #     print 'Filter %s' % filter_u
        #     run_crossmatch_lc(field, CCD, filter_u, kind='temp')
        # # Creaing table files
        # print 'Creaing table files with color photometry...'
        # color_table_r = get_color_table(field, CCD, epochs_g_org,
        #                                 epochs_r, filter2='r', ref_f2='01')
        # print color_table_r
        # color_table_u = get_color_table(field, CCD, epochs_g_org,
        #                                 epochs_u, filter2='u', ref_f2=u_ref_p)
        # print color_table_u

        # Color correction
        # print 'Filter %s' % filter_g
        # color_calibration_DECam(field, CCD, epochs_g, color_table_r)
        # print 'Filter %s' % filter_r
        # color_calibration_DECam(field, CCD, epochs_r, color_table_r)
        # color term correction for i-band needs z band
        # if False:
        #     print 'Filter %s' % filter_i
        #     color_calibration_DECam(field, CCD, epochs_i, color_table)

        # Zero Point calibration with PanSTARRS catalog
        print 'Filter %s' % filter_g
        zero_point_PS(field, CCD, epochs_g, filter_g, center=radec_center,
                      aflux=False, Plot=False)
        print 'Filter %s' % filter_r
        zero_point_PS(field, CCD, epochs_r, filter_r, aflux=False, Plot=False)
        if i_bool:
            print 'Filter %s' % filter_i
            zero_point_PS(field, CCD, epochs_i, filter_i, aflux=False,
                          Plot=False)

    elif field[:8] == 'Blind14A':

        filter_g = 'g'

        epochs_g_file = '%s/info/%s/%s_epochs_%s.txt' % (
            jorgepath, field, field, filter_g)
        if not os.path.exists(epochs_g_file):
            print 'No epochs file: %s' % (epochs_g_file)
            sys.exit()

        epochs_g = np.loadtxt(epochs_g_file, comments='#', dtype=str)

        # Convert positions and convert flux to reference epoch
        print 'Filter %s' % filter_g
        to_remove_g, radec_center = coordinates_correction(field, CCD,
                                                           epochs_g,
                                                           a_ref=g_ref_a)
        print 'Before:\n ', epochs_g
        print 'Remove:\n ', to_remove_g
        if len(to_remove_g) != 0:
            epochs_g = np.delete(epochs_g,
                                 np.where(epochs_g == to_remove_g)[0], axis=0)
        print 'After:\n ', epochs_g

        # Correct errors for correlation in pixels
        print 'Filter %s' % filter_g
        correct_pix_correlation(field, CCD, epochs_g)

        # aflux correction
        print 'Filter %s' % filter_g
        to_remove_g = flux_correction(field, CCD, epochs_g,
                                      p_ref=g_ref_p, pix_corr=True)

        # apply ZeroPoint + airmass correction
        print 'Filter %s' % filter_g
        standard_calibration_DECam(field, CCD, epochs_g, filter_g, aflux=True,
                                   p_ref=g_ref_p)

        # Zero Point calibration with PanSTARRS catalog
        print 'Filter %s' % filter_g
        zero_point_PS(field, CCD, epochs_g, filter_g, center=radec_center,
                      aflux=False, Plot=False)

    elif field[:8] == 'Blind13A':

        filter_u = 'u'

        epochs_u_file = '%s/info/%s/%s_epochs_%s.txt' % (
            jorgepath, field, field, filter_u)
        if not os.path.exists(epochs_u_file):
            print 'No epochs file: %s' % (epochs_u_file)
            sys.exit()
        epochs_u_org = np.loadtxt(epochs_u_file, comments='#', dtype=str)
        epochs_u = np.loadtxt(epochs_u_file, comments='#', dtype=str)

        # Convert positions and convert flux to reference epoch
        print 'Filter %s' % filter_u
        to_remove_u = coordinates_correction(field, CCD, epochs_u,
                                             a_ref=u_ref_a)
        print 'Before:\n ', epochs_u
        print 'Remove:\n ', to_remove_u
        epochs_u = np.delete(
            epochs_u, np.where(
                epochs_u == to_remove_u)[0], axis=0)
        print 'After:\n ', epochs_u

        # Correct errors for correlation in pixels
        print 'Filter %s' % filter_u
        correct_pix_correlation(field, CCD, epochs_u)

        # aflux correction
        print 'Filter %s' % filter_u
        to_remove_u = flux_correction(
            field, CCD, epochs_u, p_ref=u_ref_p, pix_corr=True)

        # apply ZeroPoint + airmass correction
        print 'Filter %s' % filter_u
        standard_calibration_DECam(field, CCD, epochs_u, filter_u,
                                   aflux=True, p_ref=u_ref_p)

    print 'Removing temporal files...'
    os.system('rm -vr %s/catalogues/%s/%s/temp_*' % (jorgepath, field, CCD))

    print 'It took', (datetime.now() - startTime), 'seconds'
    print '___________________________________________________________________'
    print 'Done!'
