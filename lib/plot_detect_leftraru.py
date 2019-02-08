## crea archivos reg con las detecciones de SE ue pasan un filtro dado.

import sys
import os
import glob
import warnings
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from misc_func_leftraru import *

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
BINS = np.logspace(1, 7, 75)
bound = 200
thresh = 1.0
minarea = 1
PLOTS = False

## 0 = number
## 1,2 = X,Y
## 3,4 = RA,DEC
## 5,6 = FLUX-ERR_AUTO
## 7, 8 = MAG-ERR_AUTO
## 9 = FLUX_RADIUS
## 10 = FWHM
## 11 = CLASS_STAR
## 12 = ELONGATION
## 13 = FLAG
## 14 = A_IMAGE
## 15 = B_IMAGE
## 16 = THETA_IMAGE
## 17 = KRON_RADIUS

def run_code(field, CCD, epoch):
	warnings.filterwarnings("ignore")

	if not os.path.exists("%s/%s" % (webpath, field)):
		print "Creating field folder"
		os.makedirs("%s/%s" % (webpath, field))
	if not os.path.exists("%s/%s/%s" % (webpath, field, CCD)):
		print "Creating CCD folder"
		os.makedirs("%s/%s/%s" % (webpath, field, CCD))

	## files for single epoch catalogues and image
	file_epoch = glob.glob("%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_*thresh%s_minarea%s_backsize64_zp.dat" % (jorgepath, field, CCD, field, CCD, epoch, str(thresh), str(minarea)))
	imag_epoch = glob.glob("%s/DATA/%s/%s/%s_%s_%s_image.fits" % (astropath, field, CCD, field, CCD, epoch))

	## import single epoch catalogue and image
	imag_hdu = fits.open(imag_epoch[0])
	imag = imag_hdu[0].data
	cata = np.loadtxt(file_epoch[0], comments='#')
	#cata = cata[(cata[:,5] > 0) & (cata[:,7] < 30) & (cata[:,1]>0+bound) & (cata[:,1]<2048-bound) & (cata[:,2]>0+bound) & (cata[:,2]<4096-bound)]

	name = '%s_%s_%s' % (field, CCD, epoch)

	reg_file = '%s/%s/%s/%s_%s_%s_ellipse.reg' % (webpath, field, CCD, field, CCD, epoch)

	## masking by 200 < FLUX < 1000 for matched objects
	regions = np.array([cata[:,1], cata[:,2], cata[:,14]*cata[:,17], cata[:,15]*cata[:,17], cata[:,16]], dtype = 'float')
	ellipse = np.empty(len(regions.T), dtype='S7')
	ellipse[:] = 'ellipse'
	regions = np.vstack((ellipse, regions)).T
	print regions.shape

	## saving reg file
	with file(reg_file, 'w') as outfile:
		outfile.write('# Region file format: DS9 version 4.1\n')
		outfile.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
		np.savetxt(outfile, regions, fmt = '%s', delimiter = ' ')


	if PLOTS:
		fig = plt.figure(0)
		ax = fig.add_subplot(111, aspect='auto')
		fig.suptitle(name+'_detections', fontsize = 10)
		log_imag = np.log10(imag + 200)
		vmin = np.percentile(log_imag, 10)
		vmax = np.percentile(log_imag, 90)
		ax.imshow(log_imag, interpolation = 'nearest', cmap = 'gray', vmin = vmin, vmax = vmax)
		for k in range(len(regions)):

			print k, regions[k]
			ells = Ellipse(xy = regions[k,1:3], width = regions[k,3], height = regions[k,4], angle = regions[k,5], color = 'g', fill = False)
			ax.add_artist(ells)
			#ax.set_fill(False)
			#ax.set_color('g')

		plt.savefig('%s/%s/%s/%s_sources.png' % (webpath, field, CCD, name), dpi = 300, orientation = 'landscape')


if __name__ == '__main__':
	field = sys.argv[1]
	CCD = sys.argv[2]
	epoch = sys.argv[3]
	run_code(field, CCD, epoch)
