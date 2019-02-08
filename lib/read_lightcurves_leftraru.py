import sys
import os
import glob
import tarfile
from datetime import datetime
import getopt
import numpy as np
from astropy.table import Table, vstack, unique
from scipy.spatial import cKDTree

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


def three_filters(field, CCD, occ=1., compress=False):

    ##########################################################################
    #                           Filter 1                                     #
    ##########################################################################

    epochs_file = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath,
                                                   field, field, 'g')
    if not os.path.exists(epochs_file):
        print '..No epochs file: %s' % (epochs_file)
        sys.exit()
    epochs = np.loadtxt(epochs_file, comments='#', dtype=str)

    epoch_c = []
    to_remove_g = []
    print '\tLoading catalogues files for g-band'
    for i, epoch in enumerate(epochs):
        print '\t\tEpoch %s' % epoch[0]

    ##########################################################################
    #                          catalogues                                    #
    ##########################################################################

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_\
                     thresh%s_minarea%s_backsize64_final.dat" % (jorgepath,
                                                                 field, CCD,
                                                                 field, CCD,
                                                                 epoch[0],
                                                                 str(thresh),
                                                                 str(minarea))
        if not os.path.exists(cata_file):
            print 'No catalog file: %s' % (cata_file)
            to_remove_g.append(i)
            continue
        cata = Table.read(cata_file, format='ascii')

        # epoch_c has all the catalogues, each element of epoch_c
        # contain the catalogue of a given epoch
        epoch_c.append(cata)

    if len(epoch_c)/float(len(epochs)) < occ:
        print '...Not enaught catalogues'
        print '_______________________________________________________________'
        return None

    indx_file = "%s/lightcurves/%s/%s/%s_%s_%s_master_index.txt" % (jorgepath,
                                                                    field,
                                                                    CCD,
                                                                    field,
                                                                    CCD, 'g')
    if not os.path.exists(indx_file):
        print '...No master index file: %s' % (indx_file)
        return None
    master_cat = np.loadtxt(indx_file, comments='#', dtype='i4')

    print '\tNumber of epochs: %i, effective: %i' % (len(epochs),
                                                     len(master_cat[0]))
    epochs = np.delete(epochs, to_remove_g, axis=0)

    ##########################################################################
    #                         Filter 2                                       #
    ##########################################################################

    epochs_file_f2 = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field,
                                                      field, 'r')
    if not os.path.exists(epochs_file_f2):
        print '..No epochs file: %s' % (epochs_file_f2)
        sys.exit()
    epochs_f2 = np.loadtxt(epochs_file_f2, comments='#', dtype=str)

    epoch_c_f2, to_remove_r = [], []

    print '\tLoading catalogues files for r-band'
    for k, epoch in enumerate(epochs_f2):
        print '\t\tEpoch %s' % epoch[0]

        # catalogues
        cata_file_f2 = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster\
                        _thresh%s_minarea%s_backsize64_final.dat" % \
                       (jorgepath, field, CCD, field, CCD, epoch[0],
                        str(thresh), str(minarea))
        if not os.path.exists(cata_file_f2):
            print 'No catalog file: %s' % (cata_file_f2)
            to_remove_r.append(k)
            continue
        cata_f2 = Table.read(cata_file_f2, format='ascii')

        if epoch[0] == '01':
            tree_r = cKDTree(np.array([cata_f2['X_IMAGE'],
                                       cata_f2['Y_IMAGE']]).T)
            tree_len_r = len(cata_f2)

        # epoch_c has all the catalogues, each element of epoch_c
        # contain the catalogue of a given epoch
        epoch_c_f2.append(cata_f2)

    if len(epoch_c_f2) == 0:
        print '...No catalogues'
        print '______________________________________________________________'
        return None

    indx_file_f2 = "%s/lightcurves/%s/%s/%s_%s_%s_master_index.txt" % \
                   (jorgepath, field, CCD, field, CCD, 'r')
    if not os.path.exists(indx_file_f2):
        print '...No master index file: %s' % (indx_file_f2)
        return None
    master_cat_f2 = np.loadtxt(indx_file_f2, comments='#', dtype='i4')

    print '\tNumber of epochs: %i, effective: %i' % (len(epochs_f2),
                                                     len(master_cat_f2[0]))
    epochs_f2 = np.delete(epochs_f2, to_remove_r, axis=0)

    ##########################################################################
    #                           Filter 3                                     #
    ##########################################################################

    epochs_file_f3 = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field,
                                                      field, 'i')
    if not os.path.exists(epochs_file_f3):
        print '...No epochs file: %s' % (epochs_file_f3)
        f3_bool = False
        # sys.exit()
    epochs_f3 = np.loadtxt(epochs_file_f3, comments='#', dtype=str)

    if len(epochs_f3) != 0 and epochs_f3.shape != (2,):
        f3_bool = True
        epoch_c_f3, to_remove_i = [], []

        print '\tLoading catalogues files for i-band'
        for k, epoch in enumerate(epochs_f3):
            print '\t\tEpoch %s' % epoch[0]

            # catalogues

            cata_file_f3 = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster\
                            _thresh%s_minarea%s_backsize64_final.dat" % \
                           (jorgepath, field, CCD, field, CCD, epoch[0],
                            str(thresh), str(minarea))
            if not os.path.exists(cata_file_f3):
                print 'No catalog file: %s' % (cata_file_f3)
                to_remove_i.append(k)
                continue
            cata_f3 = Table.read(cata_file_f3, format='ascii')

            if epoch[0] == epochs_f3[0][0]:
                tree_i = cKDTree(np.array([cata_f3['X_IMAGE'],
                                           cata_f3['Y_IMAGE']]).T)
                tree_len_i = len(cata_f3)

            # epoch_c has all the catalogues, each element of epoch_c
            # contain the catalogue of a given epoch
            epoch_c_f3.append(cata_f3)

        if len(epoch_c_f3) == 0:
            f3_bool = False
            print '...No catalogues'
            print '__________________________________________________________'

            epochs_f3 = np.delete(epochs_f3, to_remove_i, axis=0)

        indx_file_f3 = "%s/lightcurves/%s/%s/%s_%s_%s_master_index.txt" % \
                       (jorgepath, field, CCD, field, CCD, 'i')
        if not os.path.exists(indx_file_f3):
            f3_bool = False
            print '...No master index file: %s' % (indx_file_f3)
        else:
            master_cat_f3 = np.loadtxt(indx_file_f3, comments='#', dtype='i4')

        print '\tNumber of epochs: %i, effective: %i' % (len(epochs_f3),
                                                         len(master_cat_f3[0]))
    else:
        print 'No epochs for filter i'
        f3_bool = False

    ##########################################################################
    ##########################################################################

    lc_mag_g, lc_err_g = [], []
    lc_mag_ap1_g, lc_err_ap1_g = [], []
    lc_mag_ap2_g, lc_err_ap2_g = [], []
    lc_mag_ap3_g, lc_err_ap3_g = [], []
    lc_mag_ap4_g, lc_err_ap4_g = [], []
    lc_mag_ap5_g, lc_err_ap5_g = [], []
    lc_mag_r, lc_err_r = [], []
    lc_mag_ap1_r, lc_err_ap1_r = [], []
    lc_mag_ap2_r, lc_err_ap2_r = [], []
    lc_mag_ap3_r, lc_err_ap3_r = [], []
    lc_mag_ap4_r, lc_err_ap4_r = [], []
    lc_mag_ap5_r, lc_err_ap5_r = [], []
    lc_mag_i, lc_err_i = [], []
    lc_mag_ap1_i, lc_err_ap1_i = [], []
    lc_mag_ap2_i, lc_err_ap2_i = [], []
    lc_mag_ap3_i, lc_err_ap3_i = [], []
    lc_mag_ap4_i, lc_err_ap4_i = [], []
    lc_mag_ap5_i, lc_err_ap5_i = [], []
    x_pix, y_pix = [], []
    count_no_match_r = 0
    count_no_match_i = 0
    obj_id = []
    show_g = []
    show_r = []
    show_i = []
    lc_mjd_g, lc_epo_g = [], []
    lc_mjd_r, lc_epo_r = [], []
    lc_mjd_i, lc_epo_i = [], []

    for obj in range(len(master_cat)):
        print '\r\t\t Object %i of %i' % (obj, len(master_cat)),

        num_obs = np.where(master_cat[obj, :] > 0)
        if len(num_obs[0])/float(len(epochs)) >= occ:
            err_lc, mag_lc = [], []
            err_lc_ap1, mag_lc_ap1 = [], []
            err_lc_ap2, mag_lc_ap2 = [], []
            err_lc_ap3, mag_lc_ap3 = [], []
            err_lc_ap4, mag_lc_ap4 = [], []
            err_lc_ap5, mag_lc_ap5 = [], []
            pix_x, pix_y = [], []
            epo_lc, mjd_lc = [], []

            for tim in range(len(master_cat[obj, :])):
                pos = master_cat[obj, tim]
                if pos >= 0:
                    mag_lc.append(epoch_c[tim]['MAG_AUTO_ZP'][pos])
                    err_lc.append(epoch_c[tim]['MAGERR_AUTO_ZP'][pos])

                    mag_lc_ap1.append(epoch_c[tim]['MAG_APER_1_ZP'][pos])
                    err_lc_ap1.append(epoch_c[tim]['MAGERR_APER_1_ZP'][pos])

                    mag_lc_ap2.append(epoch_c[tim]['MAG_APER_2_ZP'][pos])
                    err_lc_ap2.append(epoch_c[tim]['MAGERR_APER_2_ZP'][pos])

                    mag_lc_ap3.append(epoch_c[tim]['MAG_APER_3_ZP'][pos])
                    err_lc_ap3.append(epoch_c[tim]['MAGERR_APER_3_ZP'][pos])

                    mag_lc_ap4.append(epoch_c[tim]['MAG_APER_4_ZP'][pos])
                    err_lc_ap4.append(epoch_c[tim]['MAGERR_APER_4_ZP'][pos])

                    mag_lc_ap5.append(epoch_c[tim]['MAG_APER_5_ZP'][pos])
                    err_lc_ap5.append(epoch_c[tim]['MAGERR_APER_5_ZP'][pos])

                    epo_lc.append(epochs[tim, 0])
                    mjd_lc.append(float(epochs[tim, 1]))

                    pix_x.append(epoch_c[tim]['X_IMAGE'][pos])
                    pix_y.append(epoch_c[tim]['Y_IMAGE'][pos])

            arg_sort_g = np.argsort(mjd_lc)
            lc_mag_g.append(np.array(mag_lc)[arg_sort_g])
            lc_err_g.append(np.array(err_lc)[arg_sort_g])

            lc_mag_ap1_g.append(np.array(mag_lc_ap1)[arg_sort_g])
            lc_err_ap1_g.append(np.array(err_lc_ap1)[arg_sort_g])

            lc_mag_ap2_g.append(np.array(mag_lc_ap2)[arg_sort_g])
            lc_err_ap2_g.append(np.array(err_lc_ap2)[arg_sort_g])

            lc_mag_ap3_g.append(np.array(mag_lc_ap3)[arg_sort_g])
            lc_err_ap3_g.append(np.array(err_lc_ap3)[arg_sort_g])

            lc_mag_ap4_g.append(np.array(mag_lc_ap4)[arg_sort_g])
            lc_err_ap4_g.append(np.array(err_lc_ap4)[arg_sort_g])

            lc_mag_ap5_g.append(np.array(mag_lc_ap5)[arg_sort_g])
            lc_err_ap5_g.append(np.array(err_lc_ap5)[arg_sort_g])

            lc_epo_g.append(np.array(epo_lc)[arg_sort_g])
            lc_mjd_g.append(np.array(mjd_lc)[arg_sort_g])

            show_g.append(len(mag_lc))

            x_pix.append(np.median(pix_x))
            y_pix.append(np.median(pix_y))

            # Add detections in r-band
            indx_cm = tree_r.query([x_pix[-1], y_pix[-1]], k=1,
                                   distance_upper_bound=2)[1]
            if indx_cm == tree_len_r:
                # print 'No match in filter r'
                count_no_match_r += 1
                lc_mag_r.append(None)
                lc_err_r.append(None)

                lc_mag_ap1_r.append(None)
                lc_err_ap1_r.append(None)

                lc_mag_ap2_r.append(None)
                lc_err_ap2_r.append(None)

                lc_mag_ap3_r.append(None)
                lc_err_ap3_r.append(None)

                lc_mag_ap4_r.append(None)
                lc_err_ap4_r.append(None)

                lc_mag_ap5_r.append(None)
                lc_err_ap5_r.append(None)

                lc_epo_r.append(None)
                lc_mjd_r.append(None)
                show_r.append(0)
            else:
                indx_aux = indx_cm
                obj_master_cat_f2 = master_cat_f2[indx_aux]
                rerr_lc, rmag_lc = [], []
                rerr_lc_ap1, rmag_lc_ap1 = [], []
                rerr_lc_ap2, rmag_lc_ap2 = [], []
                rerr_lc_ap3, rmag_lc_ap3 = [], []
                rerr_lc_ap4, rmag_lc_ap4 = [], []
                rerr_lc_ap5, rmag_lc_ap5 = [], []
                repo_lc, rmjd_lc = [], []
                for tim in range(len(obj_master_cat_f2)):
                    pos = master_cat_f2[indx_aux, tim]
                    if pos >= 0:
                        rmag_lc.append(epoch_c_f2[tim]['MAG_AUTO_ZP'][pos])
                        rerr_lc.append(epoch_c_f2[tim]['MAGERR_AUTO_ZP'][pos])

                        rmag_lc_ap1.append(epoch_c_f2[tim]['MAG_APER_1_ZP']
                                           [pos])
                        rerr_lc_ap1.append(epoch_c_f2[tim]['MAGERR_APER_1_ZP']
                                           [pos])

                        rmag_lc_ap2.append(epoch_c_f2[tim]['MAG_APER_2_ZP']
                                           [pos])
                        rerr_lc_ap2.append(epoch_c_f2[tim]['MAGERR_APER_2_ZP']
                                           [pos])

                        rmag_lc_ap3.append(epoch_c_f2[tim]['MAG_APER_3_ZP']
                                           [pos])
                        rerr_lc_ap3.append(epoch_c_f2[tim]['MAGERR_APER_3_ZP']
                                           [pos])

                        rmag_lc_ap4.append(epoch_c_f2[tim]['MAG_APER_4_ZP']
                                           [pos])
                        rerr_lc_ap4.append(epoch_c_f2[tim]['MAGERR_APER_4_ZP']
                                           [pos])

                        rmag_lc_ap5.append(epoch_c_f2[tim]['MAG_APER_5_ZP']
                                           [pos])
                        rerr_lc_ap5.append(epoch_c_f2[tim]['MAGERR_APER_5_ZP']
                                           [pos])

                        repo_lc.append(epochs_f2[tim, 0])
                        rmjd_lc.append(float(epochs_f2[tim, 1]))

                arg_sort_r = np.argsort(rmjd_lc)
                lc_mag_r.append(np.array(rmag_lc)[arg_sort_r])
                lc_err_r.append(np.array(rerr_lc)[arg_sort_r])

                lc_mag_ap1_r.append(np.array(rmag_lc_ap1)[arg_sort_r])
                lc_err_ap1_r.append(np.array(rerr_lc_ap1)[arg_sort_r])

                lc_mag_ap2_r.append(np.array(rmag_lc_ap2)[arg_sort_r])
                lc_err_ap2_r.append(np.array(rerr_lc_ap2)[arg_sort_r])

                lc_mag_ap3_r.append(np.array(rmag_lc_ap3)[arg_sort_r])
                lc_err_ap3_r.append(np.array(rerr_lc_ap3)[arg_sort_r])

                lc_mag_ap4_r.append(np.array(rmag_lc_ap4)[arg_sort_r])
                lc_err_ap4_r.append(np.array(rerr_lc_ap4)[arg_sort_r])

                lc_mag_ap5_r.append(np.array(rmag_lc_ap5)[arg_sort_r])
                lc_err_ap5_r.append(np.array(rerr_lc_ap5)[arg_sort_r])

                show_r.append(len(rmag_lc))

                lc_epo_r.append(np.array(repo_lc)[arg_sort_r])
                lc_mjd_r.append(np.array(rmjd_lc)[arg_sort_r])

            # Add detections in i-band
            if f3_bool:
                indx_cm = tree_i.query([x_pix[-1], y_pix[-1]],
                                       k=1, distance_upper_bound=2)[1]
                if indx_cm == tree_len_i:
                    # fill with default value if epoch exist but no detection
                    count_no_match_i += 1
                    lc_mag_i.append(None)
                    lc_err_i.append(None)

                    lc_mag_ap1_i.append(None)
                    lc_err_ap1_i.append(None)

                    lc_mag_ap2_i.append(None)
                    lc_err_ap2_i.append(None)

                    lc_mag_ap3_i.append(None)
                    lc_err_ap3_i.append(None)

                    lc_mag_ap4_i.append(None)
                    lc_err_ap4_i.append(None)

                    lc_mag_ap5_i.append(None)
                    lc_err_ap5_i.append(None)

                    lc_epo_i.append(None)
                    lc_mjd_i.append(None)
                    show_i.append(0)

                else:
                    indx_aux = np.where(master_cat_f3[:, 0] == indx_cm)[0]
                    obj_master_cat_f3 = master_cat_f3[indx_aux][0]
                    ierr_lc, imag_lc = [], []
                    ierr_lc_ap1, imag_lc_ap1 = [], []
                    ierr_lc_ap2, imag_lc_ap2 = [], []
                    ierr_lc_ap3, imag_lc_ap3 = [], []
                    ierr_lc_ap4, imag_lc_ap4 = [], []
                    ierr_lc_ap5, imag_lc_ap5 = [], []
                    iepo_lc, imjd_lc = [], []
                    for tim in range(len(obj_master_cat_f3)):
                        pos = master_cat_f3[indx_aux, tim][0]
                        if pos >= 0:
                            imag_lc.append(epoch_c_f3[tim]['MAG_AUTO_ZP']
                                           [pos])
                            ierr_lc.append(epoch_c_f3[tim]['MAGERR_AUTO_ZP']
                                           [pos])

                            imag_lc_ap1.append(epoch_c_f3[tim]
                                               ['MAG_APER_1_ZP'][pos])
                            ierr_lc_ap1.append(epoch_c_f3[tim]
                                               ['MAGERR_APER_1_ZP'][pos])

                            imag_lc_ap2.append(epoch_c_f3[tim]
                                               ['MAG_APER_2_ZP'][pos])
                            ierr_lc_ap2.append(epoch_c_f3[tim]
                                               ['MAGERR_APER_2_ZP'][pos])

                            imag_lc_ap3.append(epoch_c_f3[tim]
                                               ['MAG_APER_3_ZP'][pos])
                            ierr_lc_ap3.append(epoch_c_f3[tim]
                                               ['MAGERR_APER_3_ZP'][pos])

                            imag_lc_ap4.append(epoch_c_f3[tim]
                                               ['MAG_APER_4_ZP'][pos])
                            ierr_lc_ap4.append(epoch_c_f3[tim]
                                               ['MAGERR_APER_4_ZP'][pos])

                            imag_lc_ap5.append(epoch_c_f3[tim]
                                               ['MAG_APER_5_ZP'][pos])
                            ierr_lc_ap5.append(epoch_c_f3[tim]
                                               ['MAGERR_APER_5_ZP'][pos])

                            iepo_lc.append(epochs_f3[tim, 0])
                            imjd_lc.append(float(epochs_f3[tim, 1]))

                    arg_sort_i = np.argsort(imjd_lc)
                    lc_mag_i.append(np.array(imag_lc)[arg_sort_i])
                    lc_err_i.append(np.array(ierr_lc)[arg_sort_i])

                    lc_mag_ap1_i.append(np.array(imag_lc_ap1)[arg_sort_i])
                    lc_err_ap1_i.append(np.array(ierr_lc_ap1)[arg_sort_i])

                    lc_mag_ap2_i.append(np.array(imag_lc_ap2)[arg_sort_i])
                    lc_err_ap2_i.append(np.array(ierr_lc_ap2)[arg_sort_i])

                    lc_mag_ap3_i.append(np.array(imag_lc_ap3)[arg_sort_i])
                    lc_err_ap3_i.append(np.array(ierr_lc_ap3)[arg_sort_i])

                    lc_mag_ap4_i.append(np.array(imag_lc_ap4)[arg_sort_i])
                    lc_err_ap4_i.append(np.array(ierr_lc_ap4)[arg_sort_i])

                    lc_mag_ap5_i.append(np.array(imag_lc_ap5)[arg_sort_i])
                    lc_err_ap5_i.append(np.array(ierr_lc_ap5)[arg_sort_i])

                    lc_epo_i.append(np.array(iepo_lc)[arg_sort_i])
                    lc_mjd_i.append(np.array(imjd_lc)[arg_sort_i])

                    show_i.append(len(imag_lc))

            else:
                # fill with default value if no image exist
                lc_mag_i.append(None)
                lc_err_i.append(None)

                lc_mag_ap1_i.append(None)
                lc_err_ap1_i.append(None)

                lc_mag_ap2_i.append(None)
                lc_err_ap2_i.append(None)

                lc_mag_ap3_i.append(None)
                lc_err_ap3_i.append(None)

                lc_mag_ap4_i.append(None)
                lc_err_ap4_i.append(None)

                lc_mag_ap5_i.append(None)
                lc_err_ap5_i.append(None)

                lc_epo_i.append(None)
                lc_mjd_i.append(None)
                show_i.append(0)

    print ''
    print 'Total of objects with %i%% or more occurrences: %i' % \
          (occ * 100, len(show_g))
    print 'Total of objects with no crossmatch in filter %s: %i' % \
          ('r', count_no_match_r)
    print 'Total of objects with no crossmatch in filter %s: %i' % \
          ('i', count_no_match_i)

    if len(show_g) == 0:
        print 'No sources detected with more than occurrences asked...'
        return

    print 'Writing LC...'

    files_name_g, files_name_r, files_name_i = [], [], []
    for k in range(len(lc_epo_g)):
        print '\r', k+1,
        files_name_g.append('%s/lightcurves/%s/%s/%s_%s_%04i_%04i_%s.dat' %
                            (jorgepath, field, CCD, field, CCD,
                             x_pix[k], y_pix[k], 'g'))
        # print 'g ',show_g[k]
        keywords = ['EPOCH', 'MJD', 'MAG_KRON', 'MAGERR_KRON', 'MAG_AP1',
                    'MAGERR_AP1', 'MAG_AP2', 'MAGERR_AP2', 'MAG_AP3',
                    'MAGERR_AP3', 'MAG_AP4', 'MAGERR_AP4', 'MAG_AP5',
                    'MAGERR_AP5']
        table_g = Table([lc_epo_g[k], lc_mjd_g[k], lc_mag_g[k],
                         lc_err_g[k], lc_mag_ap1_g[k], lc_err_ap1_g[k],
                         lc_mag_ap2_g[k], lc_err_ap2_g[k], lc_mag_ap3_g[k],
                         lc_err_ap3_g[k], lc_mag_ap4_g[k], lc_err_ap4_g[k],
                         lc_mag_ap5_g[k], lc_err_ap5_g[k]],
                        names=keywords,
                        meta={'NAME': 'Light-Curves filter %s' % ('g')},
                        dtype=('S2', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                               'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))
        table_g.write(files_name_g[-1], format='ascii.commented_header',
                      delimiter='\t')

        if show_r[k] != 0:
            files_name_r.append('%s/lightcurves/%s/%s/%s_%s_%04i_%04i_%s.dat'
                                % (jorgepath, field, CCD, field, CCD,
                                   x_pix[k], y_pix[k], 'r'))
            # print 'r ',show_r[k]
            keywords_r = ['EPOCH', 'MJD', 'MAG_KRON', 'MAGERR_KRON',
                          'MAG_AP1', 'MAGERR_AP1', 'MAG_AP2', 'MAGERR_AP2',
                          'MAG_AP3', 'MAGERR_AP3', 'MAG_AP4', 'MAGERR_AP4',
                          'MAG_AP5', 'MAGERR_AP5']
            table_r = Table([lc_epo_r[k], lc_mjd_r[k], lc_mag_r[k],
                             lc_err_r[k], lc_mag_ap1_r[k], lc_err_ap1_r[k],
                             lc_mag_ap2_r[k], lc_err_ap2_r[k],
                             lc_mag_ap3_r[k], lc_err_ap3_r[k],
                             lc_mag_ap4_r[k], lc_err_ap4_r[k],
                             lc_mag_ap5_r[k], lc_err_ap5_r[k]],
                            names=keywords_r,
                            meta={'NAME': 'Light-Curves filter %s' % ('r')},
                            dtype=('S2', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                   'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))
            table_r.write(files_name_r[-1], format='ascii.commented_header',
                          delimiter='\t')

        if show_i[k] != 0:
            files_name_i.append('%s/lightcurves/%s/%s/%s_%s_%04i_%04i_%s.dat'
                                % (jorgepath, field, CCD, field, CCD,
                                   x_pix[k], y_pix[k], 'i'))
            # print 'r ',show_r[k]
            keywords_i = ['EPOCH', 'MJD', 'MAG_KRON', 'MAGERR_KRON',
                          'MAG_AP1', 'MAGERR_AP1', 'MAG_AP2', 'MAGERR_AP2',
                          'MAG_AP3', 'MAGERR_AP3', 'MAG_AP4', 'MAGERR_AP4',
                          'MAG_AP5', 'MAGERR_AP5']
            table_i = Table([lc_epo_i[k], lc_mjd_i[k], lc_mag_i[k],
                             lc_err_i[k], lc_mag_ap1_i[k], lc_err_ap1_i[k],
                             lc_mag_ap2_i[k], lc_err_ap2_i[k],
                             lc_mag_ap3_i[k], lc_err_ap3_i[k],
                             lc_mag_ap4_i[k], lc_err_ap4_i[k],
                             lc_mag_ap5_i[k], lc_err_ap5_i[k]],
                            names=keywords_i,
                            meta={'NAME': 'Light-Curves filter %s' % ('i')},
                            dtype=('S2', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                   'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))
            table_i.write(files_name_i[-1], format='ascii.commented_header',
                          delimiter='\t')

    if compress:
        print '\nCompressing LCs in file:'
        print '\t%s/lightcurves/%s/%s/%s_%s_LC_%i.tar.gz' % (jorgepath, field,
                                                             CCD, field, CCD,
                                                             occ * 100)
        tar = tarfile.open('%s/lightcurves/%s/%s/%s_%s_LC_%i.tar.gz' %
                           (jorgepath, field, CCD, field, CCD, occ * 100),
                           "w:gz")
        for lc in files_name_g:
            nam = os.path.split(lc)[1]
            tar.add(lc, arcname=nam)
        for lc in files_name_r:
            nam = os.path.split(lc)[1]
            tar.add(lc, arcname=nam)
        for lc in files_name_i:
            nam = os.path.split(lc)[1]
            tar.add(lc, arcname=nam)
        tar.close
    return


def one_filter(field, CCD, occ=15, compress=False, band='g'):

    ##########################################################################
    #                           Filter 1                                     #
    ##########################################################################

    epochs_file = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath,
                                                   field, field, band)
    if not os.path.exists(epochs_file):
        print '..No epochs file: %s' % (epochs_file)
        sys.exit()
    epochs = np.loadtxt(epochs_file, comments='#', dtype=str)
    if len(epochs) == 0:
        print 'No observations in this filter...'
        return
    if epochs.shape == (2,):
        epochs = epochs.reshape(1, 2)

    epoch_c = []
    to_remove = []
    print '\tLoading catalogues files for g-band'
    for i, epoch in enumerate(epochs):
        print '\t\tEpoch %s' % epoch[0]

        # catalogues

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
                    (jorgepath, field, CCD, field, CCD, epoch[0],
                     str(thresh), str(minarea))
        if not os.path.exists(cata_file):
            print 'No catalog file: %s' % (cata_file)
            to_remove.append(i)
            continue

        cata = Table.read(cata_file, format='ascii')
        # epoch_c has all the catalogues, each element of
        # epoch_c contain the catalogue of a given epoch
        epoch_c.append(cata)

    if len(epoch_c) < occ:
        print '...Not enaught catalogues'
        print '______________________________________________________________'
        return

    indx_file = "%s/lightcurves/%s/%s/%s_%s_%s_master_index.txt" % \
                (jorgepath, field, CCD, field, CCD, band)
    if not os.path.exists(indx_file):
        print '...No master index file: %s' % (indx_file)
        return None
    master_cat = np.loadtxt(indx_file, comments='#', dtype='i4')

    if master_cat.shape == (len(master_cat),):
        master_cat = master_cat.reshape(len(master_cat), 1)
    print '\tNumber of epochs: %i, effective: %i' % (len(epochs),
                                                     len(master_cat[0]))
    epochs = np.delete(epochs, to_remove, axis=0)

    if band == 'u':
        corr = 'ZA'
    else:
        corr = 'ZP'

    ##########################################################################
    ##########################################################################

    lc_mag, lc_err = [], []
    lc_mag_ap1, lc_err_ap1 = [], []
    lc_mag_ap2, lc_err_ap2 = [], []
    lc_mag_ap3, lc_err_ap3 = [], []
    lc_mag_ap4, lc_err_ap4 = [], []
    lc_mag_ap5, lc_err_ap5 = [], []
    x_pix, y_pix = [], []
    obj_id = []
    show = []
    lc_mjd, lc_epo = [], []

    for obj in range(len(master_cat)):
        print '\r\t\t Object %i of %i' % (obj, len(master_cat)),

        num_obs = np.where(master_cat[obj, :] > 0)
        if len(num_obs[0]) >= occ:
            show.append(len(num_obs[0]))
            err_lc, mag_lc = [], []
            err_lc_ap1, mag_lc_ap1 = [], []
            err_lc_ap2, mag_lc_ap2 = [], []
            err_lc_ap3, mag_lc_ap3 = [], []
            err_lc_ap4, mag_lc_ap4 = [], []
            err_lc_ap5, mag_lc_ap5 = [], []
            pix_x, pix_y = [], []
            epo_lc, mjd_lc = [], []

            for tim in range(len(master_cat[obj, :])):
                pos = master_cat[obj, tim]
                if pos >= 0:
                    mag_lc.append(epoch_c[tim]['MAG_AUTO_%s' % corr][pos])
                    err_lc.append(epoch_c[tim]['MAGERR_AUTO_%s' % corr][pos])

                    mag_lc_ap1.append(epoch_c[tim]['MAG_APER_%s' % corr]
                                      [pos])
                    err_lc_ap1.append(epoch_c[tim]['MAGERR_APER_%s' % corr]
                                      [pos])

                    mag_lc_ap2.append(epoch_c[tim]['MAG_APER_1_%s' % corr]
                                      [pos])
                    err_lc_ap2.append(epoch_c[tim]['MAGERR_APER_1_%s' % corr]
                                      [pos])

                    mag_lc_ap3.append(epoch_c[tim]['MAG_APER_2_%s' % corr]
                                      [pos])
                    err_lc_ap3.append(epoch_c[tim]['MAGERR_APER_2_%s' % corr]
                                      [pos])

                    mag_lc_ap4.append(epoch_c[tim]['MAG_APER_3_%s' % corr]
                                      [pos])
                    err_lc_ap4.append(epoch_c[tim]['MAGERR_APER_3_%s' % corr]
                                      [pos])

                    mag_lc_ap5.append(epoch_c[tim]['MAG_APER_4_%s' % corr]
                                      [pos])
                    err_lc_ap5.append(epoch_c[tim]['MAGERR_APER_4_%s' % corr]
                                      [pos])

                    epo_lc.append(epochs[tim, 0])
                    mjd_lc.append(float(epochs[tim, 1]))

                    pix_x.append(epoch_c[tim]['X_IMAGE_REF'][pos])
                    pix_y.append(epoch_c[tim]['Y_IMAGE_REF'][pos])

            arg_sort = np.argsort(mjd_lc)
            lc_mag.append(np.array(mag_lc)[arg_sort])
            lc_err.append(np.array(err_lc)[arg_sort])

            lc_mag_ap1.append(np.array(mag_lc_ap1)[arg_sort])
            lc_err_ap1.append(np.array(err_lc_ap1)[arg_sort])

            lc_mag_ap2.append(np.array(mag_lc_ap2)[arg_sort])
            lc_err_ap2.append(np.array(err_lc_ap2)[arg_sort])

            lc_mag_ap3.append(np.array(mag_lc_ap3)[arg_sort])
            lc_err_ap3.append(np.array(err_lc_ap3)[arg_sort])

            lc_mag_ap4.append(np.array(mag_lc_ap4)[arg_sort])
            lc_err_ap4.append(np.array(err_lc_ap4)[arg_sort])

            lc_mag_ap5.append(np.array(mag_lc_ap5)[arg_sort])
            lc_err_ap5.append(np.array(err_lc_ap5)[arg_sort])

            lc_epo.append(np.array(epo_lc)[arg_sort])
            lc_mjd.append(np.array(mjd_lc)[arg_sort])

            x_pix.append(np.median(pix_x))
            y_pix.append(np.median(pix_y))

    print 'Total of objects with %i%% or more occurrences: %i' % \
          (occ, len(show))

    if len(show) == 0:
        print 'No sources detected with more than the occurrences asked...'
        return

    print 'Writing LC...'

    files_name = []
    for k in range(len(lc_mjd)):
        lc_name = '%s/lightcurves/%s/%s/%s_%s_%04i_%04i_%s.dat' % \
            (jorgepath, field, CCD, field, CCD, x_pix[k], y_pix[k], band)
        files_name.append(lc_name)

        # print 'g ',show[k]
        # keywords = ['EPOCH', 'MJD', 'MAG_KRON', 'MAGERR_KRON', 'MAG_AP1',
        #             'MAGERR_AP1', 'MAG_AP2', 'MAGERR_AP2', 'MAG_AP3',
        #             'MAGERR_AP3', 'MAG_AP4', 'MAGERR_AP4', 'MAG_AP5',
        #             'MAGERR_AP5']
        # table = Table([lc_epo[k], lc_mjd[k], lc_mag[k], lc_err[k],
        #                lc_mag_ap1[k], lc_err_ap1[k], lc_mag_ap2[k],
        #                lc_err_ap2[k], lc_mag_ap3[k], lc_err_ap3[k],
        #                lc_mag_ap4[k], lc_err_ap4[k], lc_mag_ap5[k],
        #                lc_err_ap5[k]],
        #               names=keywords,
        #               meta={'NAME': 'Light-Curves filter %s' % (band)},
        #               dtype=('S2', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
        #                      'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))
        keywords = ['EPOCH', 'MJD', 'MAG_KRON', 'MAGERR_KRON']
        table = Table([lc_epo[k], lc_mjd[k], lc_mag[k], lc_err[k]],
                      names=keywords,
                      meta={'NAME': 'Light-Curves filter %s' % (band)},
                      dtype=('i4', 'f8', 'f8', 'f8'))

        if os.path.exists(lc_name):
            print 'Extending lightcurve...'
            print lc_name
            aux_table = Table.read(lc_name, format='ascii.commented_header',
                                   delimiter='\t')
            table = vstack([aux_table, table])
            table.sort('MJD')

        table = unique(table, keys='MJD')

        table.write(lc_name, format='ascii.commented_header',
                    delimiter='\t', overwrite=True)

    if compress:
        print 'Removing previous tar file...'
        tar_gz = '%s/lightcurves/%s/%s/%s_%s_LC_occ%i.tar.gz' % \
                 (jorgepath, field, CCD, field, CCD, occ)
        try:
            os.remove(tar_gz)
        except OSError:
            pass

        print 'Compressing LCs in new file:'
        print '\t%s/lightcurves/%s/%s/%s_%s_LC_occ%i.tar.gz' % \
              (jorgepath, field, CCD, field, CCD, occ)
        tar = tarfile.open('%s/lightcurves/%s/%s/%s_%s_LC_occ%i.tar.gz' %
                           (jorgepath, field, CCD, field, CCD, occ),
                           "w:gz")
        for lc in files_name:
            nam = os.path.split(lc)[1]
            tar.add(lc, arcname=nam)
        tar.close

        print 'Removing LC files...'
        filelist = glob.glob('%s/lightcurves/%s/%s/%s_%s_*.dat' %
                             (jorgepath, field, CCD, field, CCD))
        for f in filelist:
            os.remove(f)
    return


if __name__ == '__main__':
    startTime = datetime.now()
    field = ''
    CCD = 'N1'
    occ = 15
    compress = True
    FILTER = 'all'

    if len(sys.argv) == 1:
        print help
        sys.exit()

    # read command line option.
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'F:C:o:b:c:')
    except getopt.GetoptError, err:
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
            FILTER = str(a)
        elif o in ('-c'):
            if a == 'False':
                compress = False
            else:
                compress = True
        else:
            continue
    print 'Field: ', field
    print 'CCD: ', CCD
    print 'occ: ', occ

    print 'Removing LC files...'
    filelist = glob.glob('%s/lightcurves/%s/%s/%s_%s_*.dat' %
                         (jorgepath, field, CCD, field, CCD))
    for f in filelist:
        os.remove(f)

    if FILTER == 'all':
        three_filters(field, CCD, occ=occ, compress=compress)
    else:
        print 'Filter: %s' % (FILTER)
        one_filter(field, CCD, occ=occ, compress=compress, band=FILTER)

    print 'It took', (datetime.now()-startTime), 'seconds'
    print '_________________________________________________________________'
