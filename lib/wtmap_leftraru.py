import numpy as np
from astropy.io import fits
import os
import glob
import sys
import re

field = sys.argv[1]
CCD = sys.argv[2]

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'

def wtmap_model(image, gain):
	return image/(np.ones(image.shape)*gain)

file_image = np.sort(glob.glob("%s/DATA/%s/%s/%s_%s_*_image_crblaster.fits*" % (astropath, field, CCD, field, CCD)), kind='mergesort')

for k in range(len(file_image)):
	hdu = fits.open(file_image[k])
	data = hdu[0].data
	gain = hdu[0].header['GAINA']

	wtmap_name = re.findall(r'\/Blind\d\dA\_\d\d\_\w\d+\_(.*?)\.fits', file_image[k])[0]
	wtmap_name = wtmap_name.replace('image', 'wtmap')
	wtmap_name = '%s/fits/%s/%s/%s_%s_%s.fits' % (jorgepath, field, CCD, field, CCD, wtmap_name)

	if not os.path.exists(wtmap_name):
		print 'Creating weight map'
		wtmap = wtmap_model(data, gain)
		fits.writeto(wtmap_name, wtmap, clobber = True)
	else:
		print 'weight map exists'