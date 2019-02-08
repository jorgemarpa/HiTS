# codigo que corre sextractor para una lista de imagenes,
# pero calcula flujos y magnitudes.

import numpy as np
from astropy.io import fits
from astropy.table import Table
import os
import glob
import sys
import re

###################################################################
# cargamos la lista que contiene los nombres de los archivos

field = sys.argv[1]
CCD = sys.argv[2]

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'

file_image = np.sort(glob.glob("%s/DATA/%s/%s/%s_%s_*_image_crblaster.fits*" %
                               (astropath, field, CCD, field, CCD)),
                     kind='mergesort')
file_wtmap = np.sort(glob.glob("%s/fits/%s/%s/%s_%s_*_wtmap_crblaster.fits*" %
                               (jorgepath, field, CCD, field, CCD)),
                     kind='mergesort')

###################################################################
# definimos parametros para sextractor

back_size = 64
auto_params = [2.5, 3.5]
THRE = [1.0]
aTHRESH = 1.5
min_area = [3]
aperture = [1, 2, 3, 4, 5]

param = '%s/sextractor/ctio_decam.param' % (jorgepath)
param_full = '%s/sextractor/ctio_decam.param' % (jorgepath)
nnw = '%s/sextractor/default.nnw' % (jorgepath)
sex = '%s/sextractor/ctio_decam.sex' % (jorgepath)
sex_scamp = '%s/sextractor/for_scamp.sex' % (jorgepath)
param_scamp = '%s/sextractor/for_scamp.param' % (jorgepath)

###################################################################
# create directories
if not os.path.exists("%s/catalogues/%s" % (jorgepath, field)):
    print "Creating field folder"
    os.makedirs("%s/catalogues/%s" % (jorgepath, field))
if not os.path.exists("%s/catalogues/%s/%s" % (jorgepath, field, CCD)):
    print "Creating CCD folder"
    os.makedirs("%s/catalogues/%s/%s" % (jorgepath, field, CCD))

if not os.path.exists("%s/fits/%s" % (jorgepath, field)):
    print "Creating field folder"
    os.makedirs("%s/fits/%s" % (jorgepath, field))
if not os.path.exists("%s/fits/%s/%s" % (jorgepath, field, CCD)):
    print "Creating CCD folder"
    os.makedirs("%s/fits/%s/%s" % (jorgepath, field, CCD))
if not os.path.exists("%s/fits/%s/%s/CHECKIMAGE" % (jorgepath, field, CCD)):
    print "Creating CHECKIMAGE folder"
    os.makedirs("%s/fits/%s/%s/CHECKIMAGE" % (jorgepath, field, CCD))

os.system(" rm -v %s/fits/%s/%s/CHECKIMAGE/*fits.fz" % (jorgepath, field, CCD))


###################################################################
# corremos sextractor en cada epoca

all_names = []

for minarea in min_area:
    for THRESH in THRE:
        for time in range(len(file_image)):

            hdulist = fits.open(file_image[time])

            GAIN = hdulist[0].header['GAINA']
            RON = hdulist[0].header['RDNOISEA']
            SAT = hdulist[0].header['SATURATA']
            SCALE = hdulist[0].header['PIXSCAL1']
            FWHM_p = hdulist[0].header['FWHM']
            SEEING = FWHM_p * SCALE
            EXP_TIME = float(hdulist[0].header['EXPTIME'])
            AIRMASS = float(hdulist[0].header['AIRMASS'])
            CCDID = int(hdulist[0].header['CCDNUM'])
            FILTER = hdulist[0].header['FILTER'][0]

            CTE_file = np.loadtxt('%s/info/zeropoint/psmFitDES-mean-%s.csv' %
                                  (jorgepath, FILTER), skiprows=1,
                                  usecols=(6, 10), delimiter=',')
            Ag = CTE_file[CCDID - 1][0]
            Kg = CTE_file[CCDID - 1][1]

            ZP = 25.0
            ZP1 = 2.5 * np.log10(EXP_TIME) - Ag - Kg * AIRMASS

            if FWHM_p <= 1.75:
                filter_name = '%s/sextractor/gauss_1.5_3x3.conv' % (jorgepath)
            elif FWHM_p > 1.75 and FWHM_p <= 2.25:
                filter_name = '%s/sextractor/gauss_2.0_5x5.conv' % (jorgepath)
            elif FWHM_p > 2.25 and FWHM_p <= 2.75:
                filter_name = '%s/sextractor/gauss_2.5_5x5.conv' % (jorgepath)
            elif FWHM_p > 2.75 and FWHM_p <= 3.25:
                filter_name = '%s/sextractor/gauss_3.0_7x7.conv' % (jorgepath)
            elif FWHM_p > 3.25 and FWHM_p <= 4.5:
                filter_name = '%s/sextractor/gauss_4.0_7x7.conv' % (jorgepath)
            elif FWHM_p > 4.5:
                filter_name = '%s/sextractor/gauss_5.0_9x9.conv' % (jorgepath)

            method = re.findall(r'\/Blind\d\dA\_\d\d\_\w\d+\_(.*?)\.fits',
                                file_image[time])[0]
            if time < len(file_wtmap):
                method_1 = re.findall(r'\/Blind\d\dA\_\d\d\_\w\d+\_(.*?)\.fits',
                                      file_wtmap[time])[0]
            else:
                method_1 = ''

            print 'Epoch %s' % method
            print method_1

            NAME = '%s/catalogues/%s/%s/%s_%s_%s_thresh%s_minarea%i_backsize%i_cat.dat' % (
                jorgepath, field, CCD, field, CCD, method, str(THRESH),
                minarea, back_size)
            all_names.append(NAME)

            if method[:2] == method_1[:2]:
                print 1

                cmd = 'sex %s -c %s -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME %s -PARAMETERS_NAME %s -FILTER_NAME %s -WEIGHT_TYPE MAP_VAR -WEIGHT_IMAGE %s -PHOT_AUTOPARAMS %s,%s -PHOT_APERTURES %s,%s,%s,%s,%s -DETECT_MINAREA %s -DETECT_THRESH %s -ANALYSIS_THRESH %s -BACK_SIZE %s -SATUR_LEVEL %s -GAIN %s -MAG_ZEROPOINT %s -SEEING_FWHM %s -PIXEL_SCALE %s -STARNNW_NAME %s' % \
                    (file_image[time], sex, NAME, param, filter_name,
                     file_wtmap[time],
                     str(auto_params[0]), str(auto_params[1]),
                     str((FWHM_p / 2) * 2.), str((FWHM_p / 2) * 4.),
                     str((FWHM_p / 2) * 6.), str((FWHM_p / 2) * 8.),
                     str((FWHM_p / 2) * 10.), str(minarea), str(THRESH),
                     str(aTHRESH), str(back_size), str(SAT), str(GAIN),
                     str(ZP1), str(SEEING), str(SCALE), nnw)
            else:
                print 1.1
                cmd = 'sex %s -c %s -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME %s -PARAMETERS_NAME %s -FILTER_NAME %s -WEIGHT_GAIN Y -WEIGHT_TYPE NONE -PHOT_AUTOPARAMS %s,%s -PHOT_APERTURES %s,%s,%s,%s,%s -DETECT_MINAREA %s -DETECT_THRESH %s -ANALYSIS_THRESH %s -BACK_SIZE %s -SATUR_LEVEL %s -GAIN %s -MAG_ZEROPOINT %s -SEEING_FWHM %s -PIXEL_SCALE %s -STARNNW_NAME %s' % \
                    (file_image[time], sex, NAME, param, filter_name,
                     str(auto_params[0]), str(auto_params[1]),
                     str((FWHM_p / 2) * 2.), str((FWHM_p / 2) * 4.),
                     str((FWHM_p / 2) * 6.), str((FWHM_p / 2) * 8.),
                     str((FWHM_p / 2) * 10.), str(minarea),
                     str(THRESH), str(aTHRESH), str(back_size), str(SAT),
                     str(GAIN), str(ZP1), str(SEEING),
                     str(SCALE), nnw)

            os.system(cmd)

            # leo el catalogo creado, filtro objetos para quedarme con tipo
            # estrella, calculo nuevo FWHM y corro nuevamnete sextractor con el
            # nuevo kernel
            cata = Table.read(NAME, format='ascii')
            # print cata.columns
            cata = cata[(cata['FLUX_AUTO'] > 0) & (cata['FLUX_AUTO'] < 1e5) &
                        (cata['FWHM_IMAGE'] < 15) &
                        (1 - 1 / cata['ELONGATION'] < 0.4) &
                        (cata['CLASS_STAR'] > 0.7) & (cata['FLAGS'] < 10) &
                        (cata['X_IMAGE'] > 200) & (cata['X_IMAGE'] < 1848) &
                        (cata['Y_IMAGE'] > 200) & (cata['Y_IMAGE'] < 3896)]
            FWHM_1 = np.median(cata['FWHM_IMAGE'])

            SEEING_1 = FWHM_1 * SCALE

            if FWHM_1 == 0 or np.isnan(
                    FWHM_1) or FWHM_1 < FWHM_p - 2.5 or FWHM_1 > FWHM_p + 2.5:
                FWHM_1 = FWHM_p
                SEEING_1 = FWHM_1 * SCALE
            elif FWHM_1 <= 1.75:
                filter_name = '%s/sextractor/gauss_1.5_3x3.conv' % (jorgepath)
            elif FWHM_1 > 1.75 and FWHM_1 <= 2.25:
                filter_name = '%s/sextractor/gauss_2.0_5x5.conv' % (jorgepath)
            elif FWHM_1 > 2.25 and FWHM_1 <= 2.75:
                filter_name = '%s/sextractor/gauss_2.5_5x5.conv' % (jorgepath)
            elif FWHM_1 > 2.75 and FWHM_1 <= 3.25:
                filter_name = '%s/sextractor/gauss_3.0_7x7.conv' % (jorgepath)
            elif FWHM_1 > 3.25 and FWHM_1 <= 4.5:
                filter_name = '%s/sextractor/gauss_4.0_7x7.conv' % (jorgepath)
            elif FWHM_1 > 4.5:
                filter_name = '%s/sextractor/gauss_5.0_9x9.conv' % (jorgepath)

            print 'FWHM %.4f -> %.4f' % (FWHM_p, FWHM_1)

            name_seg = '%s/fits/%s/%s/CHECKIMAGE/%s_%s_%s_segmentation_thresh%s_minarea%i_backsize%i.fits' % \
                (jorgepath, field, CCD, field, CCD, method, str(THRESH),
                 minarea, back_size)
            name_back = '%s/fits/%s/%s/CHECKIMAGE/%s_%s_%s_background_thresh%s_minarea%i_backsize%i.fits' % \
                (jorgepath, field, CCD, field, CCD, method, str(THRESH),
                 minarea, back_size)
            name_obj = '%s/fits/%s/%s/CHECKIMAGE/%s_%s_%s_object_thresh%s_minarea%i_backsize%i.fits' % \
                (jorgepath, field, CCD, field, CCD, method, str(THRESH),
                 minarea, back_size)

            if method[:2] == method_1[:2]:
                print 2
                cmd = 'sex %s -c %s -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME %s -PARAMETERS_NAME %s -FILTER_NAME %s -WEIGHT_TYPE MAP_VAR -WEIGHT_IMAGE %s -PHOT_AUTOPARAMS %s,%s -PHOT_APERTURES %s,%s,%s,%s,%s -DETECT_MINAREA %s -DETECT_THRESH %s -ANALYSIS_THRESH %s -BACK_SIZE %s -SATUR_LEVEL %s -GAIN %s -MAG_ZEROPOINT %s -SEEING_FWHM %s -PIXEL_SCALE %s -STARNNW_NAME %s -CHECKIMAGE_TYPE SEGMENTATION,BACKGROUND -CHECKIMAGE_NAME %s,%s' % \
                    (file_image[time], sex, NAME, param_full, filter_name,
                     file_wtmap[time], str(auto_params[0]),
                     str(auto_params[1]), str((FWHM_p / 2) * 2.),
                     str((FWHM_p / 2) * 4.), str((FWHM_p / 2) * 6.),
                     str((FWHM_p / 2) * 8.), str((FWHM_p / 2) * 10.),
                     str(minarea), str(THRESH), str(aTHRESH), str(back_size),
                     str(SAT), str(GAIN), str(ZP1), str(SEEING), str(SCALE),
                     nnw, name_seg, name_back)
            else:
                print 2.1
                cmd = 'sex %s -c %s -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME %s -PARAMETERS_NAME %s -FILTER_NAME %s -WEIGHT_GAIN Y -WEIGHT_TYPE NONE -PHOT_AUTOPARAMS %s,%s -PHOT_APERTURES %s,%s,%s,%s,%s -DETECT_MINAREA %s -DETECT_THRESH %s -ANALYSIS_THRESH %s -BACK_SIZE %s -SATUR_LEVEL %s -GAIN %s -MAG_ZEROPOINT %s -SEEING_FWHM %s -PIXEL_SCALE %s -STARNNW_NAME %s -CHECKIMAGE_TYPE SEGMENTATION,BACKGROUND,APERTURES -CHECKIMAGE_NAME %s,%s,%s' % \
                    (file_image[time], sex, NAME, param_full, filter_name,
                     str(auto_params[0]), str(auto_params[1]),
                     str((FWHM_p / 2) * 2.), str((FWHM_p / 2) * 4.),
                     str((FWHM_p / 2) * 6.), str((FWHM_p / 2) * 8.),
                     str((FWHM_p / 2) * 10.), str(minarea),
                     str(THRESH), str(aTHRESH), str(back_size), str(SAT),
                     str(GAIN), str(ZP1), str(SEEING), str(SCALE),
                     nnw, name_seg, name_back, name_obj)

            os.system(cmd)

            os.system('fpack %s' % (name_seg))
            os.remove(name_seg)
            os.system('fpack %s' % (name_back))
            os.remove(name_back)

            # final_head = 'NUMBER\tX_IMAGE\tY_IMAGE\tX_WORLD\tY_WORLD\tFLUX_AUTO\tFLUXERR_AUTO\tMAG_AUTO\tMAGERR_AUTO\tFLUX_RADIUS\tFWHM_IMAGE\tCLASS_STAR\tELONGATION\tFLAGS\tA_IMAGE\tB_IMAGE\tTHETA_IMAGE\tKRON_RADIUS\tFLUX_APER_1\tFLUX_APER_2\tFLUX_APER_3\tFLUX_APER_4\tFLUX_APER_5   \tFLUXERR_APER_1\tFLUXERR_APER_2\tFLUXERR_APER_3\tFLUXERR_APER_4\tFLUXERR_APER_5'

            # print final_head
            # data_temp = np.loadtxt(NAME, comments='#')
            # np.savetxt(NAME, data_temp, header=final_head, delimiter='\t')

print 'Done!'
