#!/usr/bin/python

## cript to calculate and plots statistics in lightcurves for a given field all CCD

import sys
import os
import glob
from datetime import datetime
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import re
import warnings
import getopt

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
sharepath = '/home/apps/astro/home/jmartinez'

thresh = 1.0      # threshold of catalog
minarea = 1      # min area for detection
FILTER = 'g'

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


def import_lc(field, CCD, pdt, corr, pdt_version):

	warnings.filterwarnings("ignore")

	epochs_file = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, FILTER)
	if not os.path.exists(epochs_file):
		print 'No epochs file: %s' % (epochs_file)
		sys.exit()
	epochs = np.loadtxt(epochs_file, comments = '#', dtype = str)

	INFO = []

	print 'Loading INFO files'
	for epoch in epochs:
		print 'Epoch %s' % epoch[0]

		############################################################################# INFO of epochs #################################################################################

		INFO_file = '%s/%s/%s_%s_%s.npy' % (sharepath, field, field, epoch[0], FILTER)
		if not os.path.exists(INFO_file):
			print 'No file: %s' % (INFO_file)
			continue
		INFO.append(np.load(INFO_file))

	INFO = np.asarray(INFO)

	####################################################################################### Load LC ##################################################################################

	mean_mag, std_mag, mean_err, dev_mag = [], [], [], []
	if not pdt:
		list_lc = np.sort(glob.glob("%s/lightcurves/%s/%s/%s_%s_*_%s.dat" % (jorgepath, field, CCD, field, CCD, corr)), kind='mergesort')
	else:
		list_lc = np.sort(glob.glob("%s/lightcurves/%s/%s/pdtrend/%s/%s_%s_*_%s.dat" % (jorgepath, field, CCD, pdt_version, field, CCD, corr)), kind='mergesort')

	if len(list_lc) == 0:
		print 'No LC in this CCD'
		return ['nan'], ['nan'], ['nan'], ['nan'], ['nan']

	med_mag, std_mag, med_err, dev_mag = [], [], [], []
	for lc_file in list_lc:
		LC = np.loadtxt(lc_file, comments = '#')
		med_mag.append(np.median(LC[:,0]))
		std_mag.append(np.std(LC[:,0]))
		med_err.append(np.median(LC[:,1]))
		dev_mag.append(LC[:,0] - np.median(LC[:,0]))

	med_mag = np.asarray(med_mag)
	std_mag = np.asarray(std_mag)
	med_err = np.asarray(med_err)
	dev_mag = np.asarray(dev_mag)

	return med_mag, std_mag, med_err, [INFO['AIRMASS'], INFO['SEEING']], dev_mag



if __name__ == '__main__':
	startTime = datetime.now()

	field = ''
	pdt = False
	corr = 'af'
	pdt_version = 'MAG'

	if len(sys.argv) == 1:
		print help
		sys.exit()

	#read command line option.
	try:
		optlist, args = getopt.getopt(sys.argv[1:], 'F:p:a:v:')
	except getopt.GetoptError, err:
		print help
		sys.exit()

	for o, a in optlist:
		if o in ('-F'):
			field = str(a)
		elif o in ('-a'):
			corr = str(a)
		elif o in ('-v'):
			pdt_version = str(a)
		elif o in ('-p'):
			if a == 'False':
				pdt = False
			else:
				pdt = True
		else:
			continue


	CHIPS = np.loadtxt('%s/info/ccds.txt' % (jorgepath), comments = '#', dtype = str)

	med_mag_all, std_mag_all, med_err_all, dev_mag_all = [], [], [], []
	pdt_version = '%s_%s' % (corr.upper(), pdt_version)

	for ccd in CHIPS[:]:
		print 'CCD: %s' % ccd
		#if ccd == 'N11': break
		med_mag, std_mag, med_err, INFO, dev_mag= import_lc(field, ccd, pdt, corr, pdt_version)
		if len(med_mag) == 1 and med_mag[0] == 'nan': continue
		med_mag_all.append(med_mag)
		std_mag_all.append(std_mag)
		med_err_all.append(med_err)
		dev_mag_all.append(dev_mag)
		print '_______________________________________________________________________'
	print '_______________________________________________________________________'
	print '_______________________________________________________________________'

	dev_mag_all = np.vstack(dev_mag_all[:])
	print dev_mag_all.shape

	med_mag_all = [val for sublist in med_mag_all for val in sublist]
	med_mag_all = np.asarray(med_mag_all)
	std_mag_all = [val for sublist in std_mag_all for val in sublist]
	std_mag_all = np.asarray(std_mag_all)
	med_err_all = [val for sublist in med_err_all for val in sublist]
	med_err_all = np.asarray(med_err_all)

	print 'Number of objects with all epochs: %i' % (med_mag_all.shape[0])

	if len(med_mag_all) == 0:
		print 'No LC in this field to calculate statistics'
		sys.exit()

	#airmas_bin = np.histogram(INFO[0],20)
	#seeing_bin = np.histogram(INFO[1],20)

	if True:
		if not os.path.exists("%s/%s" % (webpath, field)):
			print "Creating field folder"
			os.makedirs("%s/%s" % (webpath, field))

		if not pdt:
			name = '%s_%s_PHOT_AUTO' % (field, pdt_version)
		else:
			name = '%s_%s_PDT_PHOT_AUTO' % (field, pdt_version)
		fig, ax = plt.subplots(2,3, figsize = (18,8))
		fig.suptitle(name, fontsize = 15)

		ax[0,0].hist(med_mag_all, bins = 200, log = True, color = 'g', alpha = 0.5, histtype = 'stepfilled', normed = False, label = 'Objects with all epochs')
		ax[0,0].legend(loc = 'upper right', fontsize='xx-small')
		ax[0,0].set_xlabel('Median MAG')
		ax[0,0].set_ylabel('N')

		ax[1,0].hist(std_mag_all, bins = 200, log = True, color = 'g', alpha = 0.5, histtype = 'stepfilled', normed = False, label = 'Objects with all epochs')
		ax[1,0].legend(loc = 'upper right', fontsize='xx-small')
		ax[1,0].set_xlabel('STD of LC')
		ax[1,0].set_ylabel('N')

		ax[0,1].scatter(med_mag_all, std_mag_all, marker = '.', color = 'b', label = 'Objects with all epochs')
		ax[0,1].legend(loc = 'upper right', fontsize='xx-small')
		ax[0,1].set_xlabel('Median MAG')
		ax[0,1].set_ylabel('STD of LC')
		ax[0,1].set_ylim(0, .5)
		ax[0,1].grid()

		ax[1,1].scatter(med_err_all, std_mag_all, marker = '.', color = 'b', label = 'Objects with all epochs')
		xxx = np.linspace(np.min(med_err_all), np.max(med_err_all), 100)
		ax[1,1].plot(xxx, xxx, '-r', label = 'Slope 1')
		ax[1,1].legend(loc = 'upper right', fontsize='xx-small')
		ax[1,1].set_xlabel('Median of error')
		ax[1,1].set_ylabel('STD of LC')
		ax[1,1].set_ylim(0, .5)
		ax[1,1].grid()

		for dev in dev_mag_all:
			ax[0,2].scatter(INFO[0], dev, marker = '.', alpha = 0.3, color = 'b')
			#ax[0,2].scatter(INFO[0], np.median(dev), marker = '*', alpha = 0.7, color = 'r')
			ax[1,2].scatter(INFO[1], dev, marker = '.', alpha = 0.3, color = 'b')
			#ax[1,2].scatter(INFO[1], np.median(dev), marker = '*', alpha = 0.7, color = 'r')

		rms = np.sqrt(np.median(dev_mag_all, axis = 0).dot(np.median(dev_mag_all, axis = 0))/np.median(dev_mag_all, axis = 0).size)
		ax[0,2].scatter(INFO[0], np.median(dev_mag_all, axis = 0), marker = '*', alpha = 0.7, color = 'r', label = 'rms %f' % (rms))
		ax[1,2].scatter(INFO[1], np.median(dev_mag_all, axis = 0), marker = '*', alpha = 0.7, color = 'r', label = 'rms %f' % (rms))

		ax[0,2].legend(loc = 'upper right', fontsize='x-small')
		ax[0,2].set_xlabel('AIRMASS')
		ax[0,2].set_ylabel('Deviation from Median MAG')
		ax[0,2].set_ylim(-3*np.std(dev_mag_all), 3*np.std(dev_mag_all))
		ax[0,2].grid()

		ax[1,2].legend(loc = 'upper right', fontsize='x-small')
		ax[1,2].set_xlabel('SEEING')
		ax[1,2].set_ylabel('Deviation from Median MAG')
		ax[1,2].set_ylim(-3*np.std(dev_mag_all), 3*np.std(dev_mag_all))
		ax[1,2].grid()

		plt.savefig('%s/%s/%s_occurrences.png' % (webpath, field, name), dpi = 300)

	print 'It took', (datetime.now()-startTime), 'seconds'
	print '_______________________________________________________________________'
