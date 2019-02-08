#!/usr/bin/python

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.matching import match_coordinates_sky
from misc_func_leftraru import *

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
sharepath = '/home/apps/astro/home/jmartinez'

thresh = 1.0      # threshold of catalog
minarea = 1      # min area for detection
deg2rad = 0.0174532925
rad2deg = 57.2957795
tolerance = 0.0003

## match files: aflux | e_aflux | rms | order | sol_astrometry
## matchRADEC: afluxADUB | e_afluxADUB | rmsdeg | CRVAL1 | CRVAL2 | CRPIX1 | CRPIX2 | CD11 | CD12 | CD21 | CD22 | nPV1 | nPV2 | order | sol_astrometry_RADEC | PV(flatten)

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

## in fits header: shape = (X, Y) = (2048, 4096)
## importing image with pyfits: shape = (Y, X) = (row, col) = (4096, 2048)
## SE catalogue: shape = (X, Y) = (2048, 4096)

def run_lc(field, CCD, FILTER, row_pix, col_pix):

	epochs = np.loadtxt('%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, FILTER), dtype={'names': ('EPOCH', 'MJD'), 'formats': ('S2', 'f4')}, comments='#')

	name = '%s_%s_%s_%i_%i' % (field, CCD, FILTER, row_pix, col_pix)

	time = []
	time_series = []
	time_series_epoch = []
	time_series_no = []
	images = []
	paso = 0

	print 'Loading catalogues'
	for epoch in epochs:
		print 'Epoch %s' % epoch[0]

		######################################################################### INFO epoch file ########################################################################

		INFO_file = '%s/%s/%s_%s_%s.npy' % (sharepath, field, field, epoch[0], FILTER)
		if not os.path.exists(INFO_file):
			print 'No file: %s' % (INFO_file)
			paso += 1
			continue
		INFO = np.load(INFO_file)

		############################################################################ catalogues ##########################################################################

		cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_trans.dat" % (jorgepath, field, CCD, field, CCD, epoch[0], str(thresh), str(minarea))
		if not os.path.exists(cata_file):
			print 'No catalog file: %s' % (cata_file)
			continue
		cata = np.loadtxt(cata_file, comments='#')
		cata_XY = np.transpose(np.array((cata[:,1], cata[:,2])))
		tree_XY = cKDTree(cata_XY)

		XY_obj = np.transpose(np.array((col_pix, row_pix)))
		indx = tree_XY.query(XY_obj, k = 1, distance_upper_bound = 2)[1]
		if indx == len(cata):
			print 'No match in epoch %s' % epoch[0]
			time_series_no.append(INFO)
		else:
			time_series.append(cata[indx])
			time_series_epoch.append(epoch)
			row_pix, col_pix = cata[indx,2], cata[indx,1]
			print row_pix, col_pix

		############################################################################### fits ############################################################################

		imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (astropath, field, CCD, field, CCD, epoch[0])
		if not os.path.exists(imag_file):
			print 'No image file: %s' % (imag_file)
			continue
		hdufits = fits.open(imag_file)
		data = hdufits[0].data
		if epoch[0] == '02':
			print epoch[0]
			images.append(data[row_pix-25:row_pix+25,col_pix-25:col_pix+25])
		else:
			mtch_file = "%s/SHARED/%s/%s/CALIBRATIONS/match_%s_%s_%s-02.npy" % (astropath, field, CCD, field, CCD, epoch[0])
			if not os.path.exists(mtch_file):
				print 'No match file: %s' % (mtch_file)
				continue
			match_coef = np.load(mtch_file)
			col_pix2, row_pix2 = applytransformation(match_coef[3], col_pix, row_pix, match_coef[4:])
			images.append(data[row_pix2-25:row_pix2+25,col_pix2-25:col_pix2+25])
		print

	print '_________________________________________________________'
	time_series = np.asarray(time_series)
	time_series_epoch = np.asarray(time_series_epoch)
	time_series_no = np.asarray(time_series_no)

	############################################################################## ploting LC ###########################################################################

	if len (time_series_no) == len(epochs) - paso or len(time_series) < 8:
		print 'Brake: only No matches or too less'
		return

	print 'Ploting...'

	if len(time_series) > 0 or len(time_series_no) > 0:
		fig, ax = plt.subplots(2, figsize = (12,9))
		fig.suptitle(name, fontsize = 15)
		ax[0].grid()
		ax[1].grid()
		if len(time_series) > 0:
			ax[0].errorbar(time_series_epoch['MJD'], time_series[:,7], yerr = time_series[:,8], fmt = 'o', color = 'b', label = 'DETECT_MAG')
			ax[1].errorbar(time_series_epoch['EPOCH'], time_series[:,7], yerr = time_series[:,8], fmt = 'o', color = 'b', label = 'DETECT_MAG')
		if len(time_series_no) > 0:
			ax[0].scatter(time_series_no['MJD'], time_series_no['LIMIT_MAG'], marker = 'v', color = 'r', label = 'LIMIT_MAG')
			ax[1].scatter(time_series_no['EPOCH'], time_series_no['LIMIT_MAG'], marker = 'v', color = 'r', label = 'LIMIT_MAG')
		ax[0].legend(loc = 'upper right', fontsize='xx-small')
		ax[1].legend(loc = 'upper right', fontsize='xx-small')
		ax[0].set_xlabel('MJD')
		ax[0].set_ylabel('MAG_g')
		ax[1].set_xlabel('Time Index')
		ax[1].set_ylabel('MAG_g')
		ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		#ax.set_xlim(np.min(epochs[0], epochs[25]))
		ax[0].invert_yaxis()
		ax[1].invert_yaxis()
		plt.savefig('%s/RR-Ly/%s_lightcurve.png' % (webpath, name), dpi = 300)

	if len(images) == len(time_series):
		fig, ax = plt.subplots(1, len(images), sharey = True,  figsize = (1. * len(images), 1.))
		fig.suptitle(name, fontsize = 12, y = 1.4)
		for k in range(len(ax)):
			image_log = images[k]
			vmin = np.percentile(image_log, 10)
			vmax = np.percentile(image_log, 90)
			ax[k].imshow(image_log, interpolation = 'nearest', cmap = 'gray', vmin = vmin, vmax = vmax)
			ax[k].set_xlabel('%s' % epochs['EPOCH'][k])
			ax2 = ax[k].twiny()
			ax2.set_xlabel('%s' % epochs['MJD'][k], fontsize = 8)
		fig.subplots_adjust(hspace = 0, wspace=0)
		plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=False)
		plt.setp([a.get_yticklabels() for a in fig.axes[:]], visible=False)
		plt.savefig('%s/RR-Ly/%s_sequence.png' % (webpath, name), dpi = 300, bbox_inches = 'tight', pad_inches = 0.3)



if __name__ == '__main__':
	#field = sys.argv[1]
	FILTER = 'g'
	#print field
	#ra = float(sys.argv[3])
	#dec = float(sys.argv[4])

	rr_list = np.loadtxt('%s/info/toJorel.dat' % (jorgepath), comments = '#', usecols = [1,3,4], dtype = {'names': ('FIELD15', 'RA', 'DEC'), 'formats': ('S2', 'f8', 'f8')})
	print rr_list.shape
	ang_dist = []

	CHIPS = np.loadtxt('%s/info/ccds.txt' % (jorgepath), comments = '#', dtype = str)

	for k in range(5,len(rr_list)):
		field = 'Blind15A_%s' % (str(rr_list['FIELD15'][k]))
		print field
		ra = rr_list['RA'][k]
		dec = rr_list['DEC'][k]
		print 'RA = %.10f, DEC= %.10f' % (ra,dec)
		row_pix = None
		col_pix = None
		dist = []
		for CCD in CHIPS:

			print 'Looking in CCD %s' % CCD
			cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_trans.dat" % (jorgepath, field, CCD, field, CCD, '02', str(thresh), str(minarea))
			if not os.path.exists(cata_file):
				print 'No file: %s' % files_epoch
				continue
			cata = np.loadtxt(cata_file, comments='#')
			tree = SkyCoord(ra = cata[:,3], dec = cata[:,4], unit=(u.degree,u.degree))
			radec_obj = SkyCoord(ra = ra, dec = dec, unit=(u.degree,u.degree))
			idx, ang, d3d = match_coordinates_sky(radec_obj, tree, nthneighbor = 1)
			ang0 = np.array(ang)
			dist.append(ang0)
			print 'Nearest neighbor at %f arcsec' % (ang0*3600)

			# ang0 is in degrees 0.0003 deg ~ 1.0 arcsec
			if ang0 < tolerance:
				row_pix, col_pix = cata[idx,2], cata[idx,1]
				print 'RADEC in catalogue: %f, %f' % (cata[idx,3], cata[idx,4])
				print 'Angular distance: %f acrsec' % (ang0*3600)
				print 'YES!!'
				print '########################################'
				break
			else:
				print 'No match!'
				print '----------------------------------------'
				continue

		ang_dist.append(np.min(dist))
		if row_pix != None and col_pix != None:
			print '(row, col) ',row_pix, col_pix
			run_lc(field, CCD, FILTER, row_pix, col_pix)
			print 'Done!'
		else:
			print '\nNo matched object within %f arcsec' % (tolerance*3600)
		print '\n::::::::::::::::::::::::::::::::::::::::::::::::::::::'

	print 'Set of minimum angular distances between RR-Ly and 2015 best candidate'
	print ang_dist
	print 'Done!'
