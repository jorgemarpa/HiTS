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
import seaborn as sns
from astropy.table import Table, vstack

sns.set(style="white", color_codes=True, context="poster")

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
sharepath = '/home/apps/astro/home/jmartinez'

thresh = 1.0      # threshold of catalog
minarea = 1      # min area for detection
FILTER = 'g'
occ = 0.7

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


def run_code(field, CCD):

	warnings.filterwarnings("ignore")

	epochs_file = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, FILTER)
	if not os.path.exists(epochs_file):
		print 'No epochs file: %s' % (epochs_file)
		sys.exit()
	epochs = np.loadtxt(epochs_file, comments = '#', dtype = str)

	INFO = []
	epoch_c = []

	print 'Loading catalogues and INFO files'
	for epoch in epochs:
		print 'Epoch %s' % epoch[0]

		################################################################################## catalogues #######################################################################################

		cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_zp.dat" % (jorgepath, field, CCD, field, CCD, epoch[0], str(thresh), str(minarea))
		if not os.path.exists(cata_file):
			print 'No catalog file: %s' % (cata_file)
			continue
		cata = Table.read(cata_file, format = 'ascii')

		## epoch_c has all the catalogues, each element of epoch_c contain the catalogue of a given epoch
		epoch_c.append(cata)

		############################################################################## INFO of epochs #######################################################################################

		INFO_file = '%s/%s/%s_%s_%s.npy' % (sharepath, field, field, epoch[0], FILTER)
		if not os.path.exists(INFO_file):
			print 'No file: %s' % (INFO_file)
			continue
		INFO.append(np.load(INFO_file))

	INFO = np.asarray(INFO)

	if len(epoch_c) == 0:
		print 'No catalogues'
		print '_______________________________________________________________________'
		return ['nan'], ['nan'], ['nan'], ['nan'], ['nan'], ['nan'], ['nan']

	print '_______________________________________________________________________'

	indx_file = "%s/lightcurves/%s/%s/%s_%s_g_master_index.txt" % (jorgepath, field, CCD, field, CCD)
	if not os.path.exists(indx_file):
		print 'No master index file: %s' % (indx_file)
		return ['nan'], ['nan'], ['nan'], ['nan'], ['nan'], ['nan'], ['nan']
	master_cat = np.loadtxt(indx_file, comments = '#')

	########################################################################################## Plots ####################################################################################

	show, mean_mag, std_mag, mean_err, perc, dev_mag = [], [], [], [], [], []

	for obj in range(len(master_cat)):

		num_obs = np.where(master_cat[obj,:] > 0)
		show.append(len(num_obs[0]))
		perc.append(float(len(num_obs[0]))/len(epochs))
		mag_g = []
		err_g = []
		for tim in range(len(master_cat[obj,:])):
			pos = master_cat[obj,tim]
			if pos > 0:
				try:
					mag_g.append(epoch_c[tim]['MAG_AUTO_ZACI'][pos])
				except KeyError:
					mag_g.append(epoch_c[tim]['MAG_AUTO_ZAC'][pos])
				try:
					err_g.append(epoch_c[tim]['MAGERR_AUTO_ZACI_COR'][pos])
				except KeyError:
					err_g.append(epoch_c[tim]['MAGERR_AUTO_ZACI'][pos])
		mean_mag.append(np.median(mag_g))
		std_mag.append(np.std(mag_g))
		mean_err.append(np.median(err_g))
		if perc[-1] >= 1.:
			dev_mag.append(mag_g - np.median(mag_g))

	dev_mag = np.array(dev_mag)

	return show, mean_mag, std_mag, mean_err, perc, [INFO['AIRMASS'], INFO['SEEING']], dev_mag



if __name__ == '__main__':
	startTime = datetime.now()
	field = sys.argv[1]
	#ccd = sys.argv[2]

	CHIPS = np.loadtxt('%s/info/ccds.txt' % (jorgepath), comments = '#', dtype = str)

	show_all, mean_mag_all, std_mag_all, perc_all, dev_mag_all, mean_err_all = [], [], [], [], [], []

	for ccd in CHIPS:
		print 'CCD: %s' % ccd
		if ccd == 'S14': continue
		show, mean_mag, std_mag, mean_err, perc, INFO, dev_mag = run_code(field, ccd)
		if len(show) == 1 and show[0] == 'nan': continue
		show_all.append(show)
		mean_mag_all.append(mean_mag)
		std_mag_all.append(std_mag)
		mean_err_all.append(mean_err)
		dev_mag_all.append(dev_mag)
		perc_all.append(perc)

	print '_______________________________________________________________________'

	#dev_mag_8_all = np.asarray(dev_mag_8_all)

	show_all = [val for sublist in show_all for val in sublist]
	show_all = np.asarray(show_all)
	mean_mag_all = [val for sublist in mean_mag_all for val in sublist]
	mean_mag_all = np.asarray(mean_mag_all)
	std_mag_all = [val for sublist in std_mag_all for val in sublist]
	std_mag_all = np.asarray(std_mag_all)
	mean_err_all = [val for sublist in mean_err_all for val in sublist]
	mean_err_all = np.asarray(mean_err_all)
	perc_all = [val for sublist in perc_all for val in sublist]
	perc_all = np.asarray(perc_all)

	mask_occ = (perc_all > occ)
	dev_mag_occ = np.vstack(dev_mag_all[:])

	print dev_mag_occ.shape

	print 'Total of objects detected %i' % (len(show_all))
	print 'Number of objects with %i%% or more occurrences: %i' % (occ*100, show_all[mask_occ].shape[0])

	if len(show_all) == 0:
		print 'No LC to calculate statistics'
		sys.exit()

	if True:
		if not os.path.exists("%s/%s" % (webpath, field)):
			print "Creating field folder"
			os.makedirs("%s/%s" % (webpath, field))

		name = '%s_all_PHOT_AUTO' % (field)
		fig, ax = plt.subplots(3,1, figsize = (6,15))
		# ax[0,0].hist(show_all, bins = 20, log = True, color = 'g', alpha = 0.5, histtype = 'stepfilled', normed = False)
		# ax[0,0].legend(loc = 'upper right', fontsize='xx-small')
		# ax[0,0].set_xlabel('Occurrences')
		# ax[0,0].set_ylabel('N')

		# ax[1,0].hist(mean_mag_8_all, bins = 200, log = True, color = 'g', alpha = 0.5, histtype = 'stepfilled', normed = False, label = 'Objects with %i percent of Occurrences' % (int(occ*100)))
		# ax[1,0].legend(loc = 'upper right', fontsize='xx-small')
		# ax[1,0].set_xlabel('Median MAG')
		# ax[1,0].set_ylabel('N')
        #
		# ax[2,0].hist(std_mag_8_all, bins = 200, log = True, color = 'g', alpha = 0.5, histtype = 'stepfilled', normed = False, label = 'Objects with %i percent of Occurrences' % (int(occ*100)))
		# ax[2,0].legend(loc = 'upper right', fontsize='xx-small')
		# ax[2,0].set_xlabel('STD of LC')
		# ax[2,0].set_ylabel('N')

		# ax[0,1].scatter(show_all, mean_mag_all, marker = '.', color = 'b')
		# ax[0,1].legend(loc = 'upper right', fontsize='xx-small')
		# ax[0,1].set_xlabel('Occurrences')
		# ax[0,1].set_ylabel('Median MAG')
		# ax[0,1].set_xlim(0, np.max(show_all)+1)
		# ax[0,1].invert_yaxis()
		# ax[0,1].grid()

		ax[0].hist(show_all, bins = np.max(show_all), log = True, color = 'k', histtype = 'step',
		 normed = False, linewidth = 2)
		ax[0].legend(loc = 'upper right', fontsize='xx-small')
		ax[0].set_xlabel('Occurrences')
		ax[0].set_ylabel('N')

		ax[1].scatter(mean_mag_all[mask_occ], std_mag_all[mask_occ], marker = '.', color = 'k')
		ax[1].legend(loc = 'upper right', fontsize='xx-small')
		ax[1].set_xlabel(r'median($m_{g}$)')
		ax[1].set_ylabel(r'std(lc)')
		ax[1].set_ylim(-0.01, .5)
		ax[1].set_xlim(15, 24)

		# ax[1,0].scatter(mean_err_all[mask_occ], std_mag_all[mask_occ], marker = '.', color = 'b', label = 'Objects with %i percent of Occurrences' % (int(occ*100)))
		# xxx = np.linspace(np.min(mean_err_all[mask_occ]), np.max(mean_err_all[mask_occ]), 100)
		# ax[1,0].plot(xxx, xxx, '-r', label = 'Slope 1')
		# ax[1,0].legend(loc = 'upper right', fontsize='xx-small')
		# ax[1,0].set_xlabel('Median of error')
		# ax[1,0].set_ylabel('STD of LC')
		# ax[1,0].set_ylim(0, .5)
		# ax[1,0].grid()


		for dev in dev_mag_occ:
			ax[2].scatter(INFO[0], dev, marker = '.', alpha = 0.3, color = 'k')
			# ax[1,1].scatter(INFO[1], dev, marker = '.', alpha = 0.3, color = 'b')


		ax[2].plot(INFO[0], np.median(dev_mag_occ, axis = 0), 'r*', alpha = 0.8, markersize=9, label='median')
		ax[2].legend(loc = 'upper right', fontsize='xx-small')
		ax[2].set_xlabel('AIRMASS')
		ax[2].set_ylabel(r'dev from median $m_{g}$')
		ax[2].set_ylim(-5 * np.std(dev_mag_occ), 5 * np.std(dev_mag_occ))


		# ax[1,1].legend(loc = 'upper right', fontsize='xx-small')
		# ax[1,1].scatter(INFO[1], np.median(dev_mag_occ, axis=0), marker = '*', alpha = 0.8, color = 'r')
		# ax[1,1].set_xlabel('SEEING')
		# ax[1,1].set_ylabel('Deviation from Median MAG')
		# ax[1,1].set_ylim(-2,4)
		# ax[1,1].grid()

		plt.savefig('%s/%s/%s_occurrences.pdf' % (webpath, field, name), dpi = 600, format='pdf', bbox_inches='tight')

	print 'It took', (datetime.now()-startTime), 'seconds'
	print '_______________________________________________________________________'
