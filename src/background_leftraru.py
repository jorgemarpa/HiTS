##codigo que corre sextractor para obtener backgorund image

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
band = sys.argv[3]

astropath = '/home/apps/astro/data'
jorgepath = '/home/jmartinez/HiTS'

###################################################################
## definimos parametros para sextractor

back_size = 64
auto_params = [2.5,3.5]
THRE = 1.5

param = '%s/sextractor/default.param' % (jorgepath)
nnw = '%s/sextractor/default.nnw' % (jorgepath)
sex = '%s/sextractor/default.sex' % (jorgepath)

epochs_file = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, band)
if not os.path.exists(epochs_file):
	print 'No epochs file: %s' % (epochs_file)
	sys.exit()
epochs = np.loadtxt(epochs_file, comments = '#', dtype = str)

for epo in epochs[:,0]:

	if epo == '02':
		ref_image = '%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits' % (astropath, field, CCD, field, CCD, epo)
		back_name = '%s/fits/%s/%s/CHECKIMAGE/%s_%s_%s_image_crblaster_internal_background%s.fits' % (jorgepath, field, CCD, field, CCD, epo, str(back_size))
	else:
		ref_image = '%s/DATA/%s/%s/%s_%s_%s_image_crblaster_grid02_lanczos2.fits' % (astropath, field, CCD, field, CCD, epo)
		back_name = '%s/fits/%s/%s/CHECKIMAGE/%s_%s_%s_image_crblaster_grid02_lanczos2_internal_background%s.fits' % (jorgepath, field, CCD, field, CCD, epo, str(back_size))

	###################################################################
	if not os.path.exists(ref_image):
		print 'No projected image...'
		continue
	## create directories
	if not os.path.exists("%s/catalogues/%s" % (jorgepath, field)):
		print "Creating field folder"
		os.makedirs("%s/catalogues/%s" % (jorgepath, field))
	if not os.path.exists("%s/catalogues/%s/%s" % (jorgepath, field, CCD)):
		print "Creating CCD folder"
		os.makedirs("%s/catalogues/%s/%s" % (jorgepath, field, CCD))


	###################################################################

	if not os.path.exists(back_name):

		print 'No background image for epoch %s, runing SE...' % (epo)
		hdulist = fits.open(ref_image)

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


		if FWHM_p <= 1.75: filter_name = '%s/sextractor/gauss_1.5_3x3.conv' % (jorgepath)
		elif FWHM_p > 1.75 and FWHM_p <= 2.25: filter_name = '%s/sextractor/gauss_2.0_5x5.conv' % (jorgepath)
		elif FWHM_p > 2.25 and FWHM_p <= 2.75: filter_name = '%s/sextractor/gauss_2.5_5x5.conv' % (jorgepath)
		elif FWHM_p > 2.75 and FWHM_p <= 3.25: filter_name = '%s/sextractor/gauss_3.0_7x7.conv' % (jorgepath)
		elif FWHM_p > 3.25 and FWHM_p <= 4.5: filter_name = '%s/sextractor/gauss_4.0_7x7.conv' % (jorgepath)
		elif FWHM_p > 4.5: filter_name = '%s/sextractor/gauss_5.0_9x9.conv' % (jorgepath)

		method = re.findall(r'\/Blind\d\dA\_\d\d\_\w\d+\_(.*?)\.fits', ref_image)[0]

		NAME = '%s/catalogues/%s/%s/%s_%s_%s_thresh%s_backsize%i_cat.dat' % (jorgepath, field, CCD, field, CCD, method, str(THRE), back_size)

		cmd = 'sex '+ref_image+' -c '+sex+' -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME '+NAME+' -PARAMETERS_NAME '+param+' -FILTER_NAME '+filter_name+' -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME '+back_name+' -PHOT_AUTOPARAMS '+str(auto_params[0])+','+str(auto_params[1])+' -DETECT_THRESH '+str(THRE)+' -ANALYSIS_THRESH '+str(THRE)+' -BACK_SIZE '+str(back_size)+' -SATUR_LEVEL '+str(SAT)+' -GAIN '+str(GAIN)+' -SEEING_FWHM '+str(SEEING)+' -PIXEL_SCALE '+str(SCALE)+' -STARNNW_NAME '+nnw+''

		os.system(cmd)
		os.system('rm -f %s' % (NAME))

else:
	print 'Background image already calculated...'

print 'Done!'
