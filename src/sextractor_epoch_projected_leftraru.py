##codigo que corre sextractor para una lista de imagenes, pero calcula flujos y magnitudes.

import numpy as np
from astropy.io import fits
import os
import glob
import sys
import re

###################################################################
##cargamos la lista que contiene los nombres de los archivos

field = sys.argv[1]
CCD = sys.argv[2]

astropath = '/home/apps/astro'
jorgepath = '/home/jmartinez/HiTS'

file_image = np.sort(glob.glob("%s/DATA/%s/%s/%s_%s_*_image_crblaster_grid02_lanczos2.fits*" % (astropath, field, CCD, field, CCD)), kind='mergesort')
file_wtmap = np.sort(glob.glob("%s/DATA/%s/%s/%s_%s_*_wtmap_grid02_lanczos2.fits*" % (astropath, field, CCD, field, CCD)), kind='mergesort')
ref_image = glob.glob('%s/DATA/%s/%s/%s_%s_02_image_crblaster.fits' % (astropath, field, CCD, field, CCD))
ref_wtmap = glob.glob('%s/DATA/%s/%s/%s_%s_02_wtmap_crblaster.fits' % (astropath, field, CCD, field, CCD))
file_image = np.hstack((file_image, ref_image))
file_wtmap = np.hstack((file_wtmap, ref_wtmap))

###################################################################
## definimos parametros para sextractor

back_size = 64
auto_params = [2.5,3.5]
THRE = [1.0]
aTHRESH = 2.4
min_area = [1]

param = '%s/sextractor/ctio_decam.param' % (jorgepath)
nnw = '%s/sextractor/default.nnw' % (jorgepath)
sex = '%s/sextractor/ctio_decam.sex' % (jorgepath)

###################################################################
## create directories
if not os.path.exists("%s/catalogues/%s" % (jorgepath, field)):
	print "Creating field folder"
	os.makedirs("%s/catalogues/%s" % (jorgepath, field))
if not os.path.exists("%s/catalogues/%s/%s" % (jorgepath, field, CCD)):
	print "Creating CCD folder"
	os.makedirs("%s/catalogues/%s/%s" % (jorgepath, field, CCD))


###################################################################
##corremos sextractor en cada epoca

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
			SEEING = FWHM_p*SCALE
			EXP_TIME = float(hdulist[0].header['EXPTIME'])
			AIRMASS = float(hdulist[0].header['AIRMASS'])
			CCDID = int(hdulist[0].header['CCDNUM'])
			FILTER = hdulist[0].header['FILTER'][0]

			print FWHM_p

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

			method = re.findall(r'\/Blind\d\dA\_\d\d\_\w\d+\_(.*?)\.fits', file_image[time])[0]
			print len(file_image), len(file_wtmap)
			if time < len(file_wtmap):
				method_1 = re.findall(r'\/Blind\d\dA\_\d\d\_\w\d+\_(.*?)\.fits', file_wtmap[time])[0]
			else:
				method_1 = ''
			print method
			print method_1

			NAME = '%s/catalogues/%s/%s/%s_%s_%s_thresh%s_minarea%i_backsize%i_cat.dat' % (jorgepath, field, CCD, field, CCD, method, str(THRESH), minarea, back_size)
			all_names.append(NAME)

			if method[:2] == method_1[:2]:
				cmd = 'sex '+file_image[time]+' -c '+sex+' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME '+NAME+' -PARAMETERS_NAME '+param+' -FILTER_NAME '+filter_name+' -WEIGHT_TYPE MAP_VAR -WEIGHT_IMAGE '+file_wtmap[time]+' -PHOT_AUTOPARAMS '+str(auto_params[0])+','+str(auto_params[1])+' -DETECT_MINAREA '+str(minarea)+' -DETECT_THRESH '+str(THRESH)+' -ANALYSIS_THRESH '+str(aTHRESH)+' -BACK_SIZE '+str(back_size)+' -SATUR_LEVEL '+str(SAT)+' -GAIN '+str(GAIN)+' -MAG_ZEROPOINT '+str(ZP1)+' -SEEING_FWHM '+str(SEEING)+' -PIXEL_SCALE '+str(SCALE)+' -STARNNW_NAME '+nnw+''
			else:
				cmd = 'sex '+file_image[time]+' -c '+sex+' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME '+NAME+' -PARAMETERS_NAME '+param+' -FILTER_NAME '+filter_name+' -WEIGHT_GAIN Y -WEIGHT_TYPE NONE -PHOT_AUTOPARAMS '+str(auto_params[0])+','+str(auto_params[1])+' -DETECT_MINAREA '+str(minarea)+' -DETECT_THRESH '+str(THRESH)+' -ANALYSIS_THRESH '+str(aTHRESH)+' -BACK_SIZE '+str(back_size)+' -SATUR_LEVEL '+str(SAT)+' -GAIN '+str(GAIN)+' -MAG_ZEROPOINT '+str(ZP1)+' -SEEING_FWHM '+str(SEEING)+' -PIXEL_SCALE '+str(SCALE)+' -STARNNW_NAME '+nnw+''

			os.system(cmd)

			## leo el catalogo creado, filtro objetos para quedarme con tipo estrella, calculo nuevo FWHM y corro nuevamnete sextractor con el nuevo kernel
			cata = np.loadtxt(NAME, comments='#')
			cata = cata[(cata[:,7] < 30) & (cata[:,10] < 20) & (cata[:,12] < 1./0.8) & (cata[:,11] > 0.6) & (cata[:,13] < 10) & (cata[:,1]>100) & (cata[:,1]<1948) & (cata[:,2]>100) & (cata[:,2]<3996)]
			FWHM_1 = np.median(cata[:,10])
			print FWHM_1

			if FWHM_1 <= 1.75: filter_name = '%s/sextractor/gauss_1.5_3x3.conv' % (jorgepath)
			elif FWHM_1 > 1.75 and FWHM_1 <= 2.25: filter_name = '%s/sextractor/gauss_2.0_5x5.conv' % (jorgepath)
			elif FWHM_1 > 2.25 and FWHM_1 <= 2.75: filter_name = '%s/sextractor/gauss_2.5_5x5.conv' % (jorgepath)
			elif FWHM_1 > 2.75 and FWHM_1 <= 3.25: filter_name = '%s/sextractor/gauss_3.0_7x7.conv' % (jorgepath)
			elif FWHM_1 > 3.25 and FWHM_1 <= 4.5: filter_name = '%s/sextractor/gauss_4.0_7x7.conv' % (jorgepath)
			elif FWHM_1 > 4.5: filter_name = '%s/sextractor/gauss_5.0_9x9.conv' % (jorgepath)

			if method[:2] == method_1[:2]:
				cmd = 'sex '+file_image[time]+' -c '+sex+' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME '+NAME+' -PARAMETERS_NAME '+param+' -FILTER_NAME '+filter_name+' -WEIGHT_TYPE MAP_VAR -WEIGHT_IMAGE '+file_wtmap[time]+' -PHOT_AUTOPARAMS '+str(auto_params[0])+','+str(auto_params[1])+' -DETECT_MINAREA '+str(minarea)+' -DETECT_THRESH '+str(THRESH)+' -ANALYSIS_THRESH '+str(aTHRESH)+' -BACK_SIZE '+str(back_size)+' -SATUR_LEVEL '+str(SAT)+' -GAIN '+str(GAIN)+' -MAG_ZEROPOINT '+str(ZP1)+' -SEEING_FWHM '+str(SEEING)+' -PIXEL_SCALE '+str(SCALE)+' -STARNNW_NAME '+nnw+''
			else:
				cmd = 'sex '+file_image[time]+' -c '+sex+' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME '+NAME+' -PARAMETERS_NAME '+param+' -FILTER_NAME '+filter_name+' -WEIGHT_GAIN Y -WEIGHT_TYPE NONE -PHOT_AUTOPARAMS '+str(auto_params[0])+','+str(auto_params[1])+' -DETECT_MINAREA '+str(minarea)+' -DETECT_THRESH '+str(THRESH)+' -ANALYSIS_THRESH '+str(aTHRESH)+' -BACK_SIZE '+str(back_size)+' -SATUR_LEVEL '+str(SAT)+' -GAIN '+str(GAIN)+' -MAG_ZEROPOINT '+str(ZP1)+' -SEEING_FWHM '+str(SEEING)+' -PIXEL_SCALE '+str(SCALE)+' -STARNNW_NAME '+nnw+''

			os.system(cmd)


print 'Done!'
