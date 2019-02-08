# codigo que corre sextractor para una lista de imagenes, pero calcula flujos
# no magnitudes.

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

stack_image = np.sort(glob.glob("%s/fits/%s/%s/%s_%s_image_stack_*.fits" %
                                (jorgepath, field, CCD, field, CCD)),
                      kind='mergesort')
stack_wtmap = np.sort(glob.glob("%s/fits/%s/%s/%s_%s_wtmap_stack_*.fits" %
                                (jorgepath, field, CCD, field, CCD)),
                      kind='mergesort')
print stack_image

###################################################################
# definimos parametros para sextractor

back_size = 64
auto_params = [2.5, 3.5]
THRE = [1.0, 1.5]
aTHRESH = 2.4
min_area = [4, 5]

param = '%s/sextractor/ctio_decam.param' % (jorgepath)
nnw = '%s/sextractor/default.nnw' % (jorgepath)
sex = '%s/sextractor/ctio_decam.sex' % (jorgepath)

###################################################################
# create directories
if not os.path.exists("%s/catalogues/%s" % (jorgepath, field)):
    print "Creating field folder"
    os.makedirs("%s/catalogues/%s" % (jorgepath, field))
if not os.path.exists("%s/catalogues/%s/%s" % (jorgepath, field, CCD)):
    print "Creating CCD folder"
    os.makedirs("%s/catalogues/%s/%s" % (jorgepath, field, CCD))


###################################################################
# corremos sextractor en cada epoca
for minarea in min_area:
	for THRESH in THRE:
		for time in range(len(stack_image)):
			method = re.findall(r'\/Blind\d\dA\_\d\d\_\w\d+\_(.*?)\.fits', stack_image[time])[0]

			hdulist = fits.open(stack_image[time])

			#print stack_image[time]

			GAIN = hdulist[0].header['GAINA']
			SAT = hdulist[0].header['SATURATA']
			SCALE = hdulist[0].header['PIXSCAL1']
			FWHM_p = hdulist[0].header['FWHM']
			SEEING = FWHM_p*SCALE
			EXP_TIME = float(hdulist[0].header['EXPTIME'])
			AIRMASS = float(hdulist[0].header['AIRMASS'])
			CCDID = int(hdulist[0].header['CCDNUM'])
			FILTER = hdulist[0].header['FILTER'][0]

			method = method.replace('_%s' % FILTER, '')
			print 'Epoch %s' % method

			CTE_file = np.loadtxt('%s/sextractor/zeropoint/%s.dat' % (jorgepath, FILTER), comments='#', usecols = (2,6))
			Ag = CTE_file[CCDID-1][0]
			Kg = CTE_file[CCDID-1][1]

			ZP = 25.0
			ZP1 = 2.5*np.log10(EXP_TIME) - Ag - Kg*AIRMASS

			if FWHM_p <= 1.75: filter_name = '%s/sextractor/gauss_1.5_3x3.conv' % (jorgepath)
			elif FWHM_p > 1.75 and FWHM_p <= 2.25: filter_name = '%s/sextractor/gauss_2.0_5x5.conv' % (jorgepath)
			elif FWHM_p > 2.25 and FWHM_p <= 2.75: filter_name = '%s/sextractor/gauss_2.5_5x5.conv' % (jorgepath)
			elif FWHM_p > 2.75 and FWHM_p <= 3.25: filter_name = '%s/sextractor/gauss_3.0_7x7.conv' % (jorgepath)
			elif FWHM_p > 3.25 and FWHM_p <= 4.5: filter_name = '%s/sextractor/gauss_4.0_7x7.conv' % (jorgepath)
			elif FWHM_p > 4.5: filter_name = '%s/sextractor/gauss_5.0_9x9.conv' % (jorgepath)


			NAME = '%s/catalogues/%s/%s/%s_%s_%s_thresh%s_minarea%i_backsize%i_cat.dat' % (jorgepath, field, CCD, field, CCD, method, str(THRESH), minarea, back_size)

			if len(stack_wtmap) != 0:
				print 1
				cmd = 'sex '+stack_image[time]+' -c '+sex+' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME '+NAME+' -PARAMETERS_NAME '+param+' -FILTER_NAME '+filter_name+' -WEIGHT_TYPE MAP_VAR -WEIGHT_IMAGE '+stack_wtmap[time]+' -PHOT_AUTOPARAMS '+str(auto_params[0])+','+str(auto_params[1])+' -DETECT_MINAREA '+str(minarea)+' -DETECT_THRESH '+str(THRESH)+' -ANALYSIS_THRESH '+str(aTHRESH)+' -BACK_SIZE '+str(back_size)+' -SATUR_LEVEL '+str(SAT)+' -GAIN '+str(GAIN)+' -MAG_ZEROPOINT '+str(ZP1)+' -SEEING_FWHM '+str(SEEING)+' -PIXEL_SCALE '+str(SCALE)+' -STARNNW_NAME '+nnw+''
			else:
				print 1.1
				cmd = 'sex '+stack_image[time]+' -c '+sex+' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME '+NAME+' -PARAMETERS_NAME '+param+' -FILTER_NAME '+filter_name+' -PHOT_AUTOPARAMS '+str(auto_params[0])+','+str(auto_params[1])+' -DETECT_MINAREA '+str(minarea)+' -DETECT_THRESH '+str(THRESH)+' -ANALYSIS_THRESH '+str(aTHRESH)+' -BACK_SIZE '+str(back_size)+' -SATUR_LEVEL '+str(SAT)+' -GAIN '+str(GAIN)+' -MAG_ZEROPOINT '+str(ZP1)+' -SEEING_FWHM '+str(SEEING)+' -PIXEL_SCALE '+str(SCALE)+' -STARNNW_NAME '+nnw+''

			os.system(cmd)

			## leo el catalogo creado, filtro objetos para quedarme con tipo estrella, calculo nuevo FWHM y corro nuevamnete sextractor con el nuevo kernel
			# cata = np.loadtxt(NAME, comments='#')
			# print cata.shape
			# cata = cata[(cata[:,5] > 0) & (cata[:,7] < 30) & (cata[:,10] < 20) & (1-1/cata[:,12] < 0.5) & (cata[:,11] > 0.6) & (cata[:,13] < 10) & (cata[:,1]>200) & (cata[:,1]<1848) & (cata[:,2]>200) & (cata[:,2]<3896)]
			cata = Table.read(NAME, format = 'ascii')
			cata = cata[(cata['FLUX_AUTO'] > 0) & (cata['FLUX_AUTO'] < 1e5) & (cata['FWHM_IMAGE'] < 15) & \
			 			(1-1/cata['ELONGATION'] < 0.4) & (cata['CLASS_STAR'] > 0.7) & (cata['FLAGS'] < 10) &\
						(cata['X_IMAGE']>200) & (cata['X_IMAGE']<1848) & \
						(cata['Y_IMAGE']>200) & (cata['Y_IMAGE']<3896)]
			FWHM_1 = np.median(cata['FWHM_IMAGE'])
			print 'FWHM %.4f -> %.4f' % (FWHM_p,FWHM_1)
			SEEING_1 = FWHM_1*SCALE

			if FWHM_1 <= 1.75: filter_name = '%s/sextractor/gauss_1.5_3x3.conv' % (jorgepath)
			elif FWHM_1 > 1.75 and FWHM_1 <= 2.25: filter_name = '%s/sextractor/gauss_2.0_5x5.conv' % (jorgepath)
			elif FWHM_1 > 2.25 and FWHM_1 <= 2.75: filter_name = '%s/sextractor/gauss_2.5_5x5.conv' % (jorgepath)
			elif FWHM_1 > 2.75 and FWHM_1 <= 3.25: filter_name = '%s/sextractor/gauss_3.0_7x7.conv' % (jorgepath)
			elif FWHM_1 > 3.25 and FWHM_1 <= 4.5: filter_name = '%s/sextractor/gauss_4.0_7x7.conv' % (jorgepath)
			elif FWHM_1 > 4.5: filter_name = '%s/sextractor/gauss_5.0_9x9.conv' % (jorgepath)
			elif np.isnan(FWHM_1): filter_name = '%s/sextractor/gauss_5.0_9x9.conv' % (jorgepath)

			if len(stack_wtmap) != 0:
				print 2
				cmd = 'sex '+stack_image[time]+' -c '+sex+' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME '+NAME+' -PARAMETERS_NAME '+param+' -FILTER_NAME '+filter_name+' -WEIGHT_TYPE MAP_VAR -WEIGHT_IMAGE '+stack_wtmap[time]+' -PHOT_AUTOPARAMS '+str(auto_params[0])+','+str(auto_params[1])+' -PHOT_APERTURES '+str(FWHM_1*1.)+','+str(FWHM_1*2.)+','+str(FWHM_1*3.)+','+str(FWHM_1*4.)+','+str(FWHM_1*5.)+' -DETECT_MINAREA '+str(minarea)+' -DETECT_THRESH '+str(THRESH)+' -ANALYSIS_THRESH '+str(aTHRESH)+' -BACK_SIZE '+str(back_size)+' -SATUR_LEVEL '+str(SAT)+' -GAIN '+str(GAIN)+' -MAG_ZEROPOINT '+str(ZP1)+' -SEEING_FWHM '+str(SEEING_1)+' -PIXEL_SCALE '+str(SCALE)+' -STARNNW_NAME '+nnw+''
			else:
				print 2.1
				cmd = 'sex '+stack_image[time]+' -c '+sex+' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME '+NAME+' -PARAMETERS_NAME '+param+' -FILTER_NAME '+filter_name+' -PHOT_AUTOPARAMS '+str(auto_params[0])+','+str(auto_params[1])+' -PHOT_APERTURES '+str(FWHM_1*1.)+','+str(FWHM_1*2.)+','+str(FWHM_1*3.)+','+str(FWHM_1*4.)+','+str(FWHM_1*5.)+' -DETECT_MINAREA '+str(minarea)+' -DETECT_THRESH '+str(THRESH)+' -ANALYSIS_THRESH '+str(aTHRESH)+' -BACK_SIZE '+str(back_size)+' -SATUR_LEVEL '+str(SAT)+' -GAIN '+str(GAIN)+' -MAG_ZEROPOINT '+str(ZP1)+' -SEEING_FWHM '+str(SEEING_1)+' -PIXEL_SCALE '+str(SCALE)+' -STARNNW_NAME '+nnw+''

			os.system(cmd)

print 'Done!'
