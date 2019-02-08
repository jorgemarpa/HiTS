import sys
import os
import glob
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.io import fits
import seaborn as sns
from matplotlib.ticker import MaxNLocator
sns.set(style="white", color_codes=True, context="poster")

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
sharepath = '/home/apps/astro/home/jmartinez'

thresh = 1.0      # threshold of catalog
minarea = 1      # min area for detection


def limit_plot(field, filter):
	epoch_file = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, filter)
	epochs = np.loadtxt(epoch_file, dtype={'names': ('EPOCH', 'MJD'), 'formats': ('S2', 'f4')}, comments='#')
	INFO_all = []
	for k in range(len(epochs)):
		INFO_file = '%s/info/%s/%s_%s_%s.npy' % (jorgepath, field, field, epochs['EPOCH'][k], filter)
		if not os.path.exists(INFO_file):
			print 'No file: %s' % (INFO_file)
			continue
		INFO = np.load(INFO_file)
		INFO_all.append(INFO)
		print INFO['EPOCH'], '->', INFO['MJD']
	INFO_all = np.asarray(INFO_all)
	print 'Load Done!'
	print INFO_all['LIMIT_MAG_EA5']


	print 'Ploting Figures'

	#rc('mathtext', default='regular')
	fig, ax = plt.subplots(2,1, figsize=(9,6), sharex=True)
	ax[0].plot(INFO_all['EPOCH'], INFO_all['SEEING'], '.-k', label = 'SEEING (arcsec)')
	ax[0].plot(INFO_all['EPOCH'], INFO_all['AIRMASS'], '*-b', label = 'AIRMASS')
	#ax[0].plot(INFO_all['AIRMASS'], INFO_all['COMPLET_MAG'], '.k', label='Completeness')
	#ax[0].plot(INFO_all['AIRMASS'], INFO_all['LIMIT_MAG_EA5'], '*k', label='Limiting')
	ax[0].legend(loc = 'upper right', fontsize='x-small')
	ax[0].set_xlim(0,36)

	ax[1].plot(INFO_all['EPOCH'], INFO_all['COMPLET_MAG'], '*-b', label = 'Completeness (80%)')
	ax[1].plot(INFO_all['EPOCH'], INFO_all['LIMIT_MAG_EA5'], '.-k', label = 'Limiting')
	#ax[1].plot(INFO_all['EPOCH'], INFO_all['LIMIT_MAG_EA10'], '.-c', label = 'LIMIT_MAG_SNR10')
	ax[1].legend(loc = 'upper right', fontsize='x-small')
	ax[1].set_xlabel('observation number')
	ax[1].set_ylabel(r'$m_{g}$')
	ax[1].set_ylim(19,26)
	ax[1].set_xlim(0,36)
	nbins = len(ax[1].get_xticklabels())
	ax[1].yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))

	#ax[1].plot(INFO_all['SEEING'], INFO_all['COMPLET_MAG'], '.k', label='Completeness')
	#ax[1].plot(INFO_all['SEEING'], INFO_all['LIMIT_MAG_EA5'], '*k', label='Limiting')
	#ax[1].legend(loc = 'upper right', fontsize='x-small')
	#ax[1].set_xlabel('AIRMASS')
	#ax[1].set_ylabel('Mag')

	fig.tight_layout()
	fig.subplots_adjust(hspace=0)
	plt.savefig('%s/%s/%s_%s_night_evo.pdf' % (webpath, field, field, filter), dpi = 600, format='pdf',   bbox_inches='tight')
	plt.close()


def airmass_evo(field, filter):
	epoch_file = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, filter)
	epochs = np.loadtxt(epoch_file, dtype={'names': ('EPOCH', 'MJD'), 'formats': ('S2', 'f4')}, comments='#')
	airmass, FWHM = [], []
	# for epo in epochs['EPOCH']:
	# 	imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (astropath, field, 'N4', field, 'N4', epo)
	# 	if not os.path.exists(imag_file):
	# 		print 'No image file: %s' % (imag_file)
	# 		continue
	# 	hdufits = fits.open(imag_file)
	# 	airmass.append(float(hdufits[0].header['AIRMASS']))
	#
	# 	cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_%s.dat" % (jorgepath, field, 'N4', field, 'N4', epo, str(thresh), str(minarea), 'zp')
	# 	if not os.path.exists(cata_file):
	# 		print 'No catalog file: %s' % (cata_file)
	# 		sys.exit()
	# 	cata = np.loadtxt(cata_file, comments='#')
	# 	cata = cata[(cata[:,10] < 20) & (1-1/cata[:,12] <= 0.5) & (cata[:,11] >= 0.) & (cata[:,13] < 4) & (cata[:,9] < 2.9)]
	#
	# 	FWHM.append(np.median(cata[:10])*0.27)
	#
	# fig = plt.figure()
	# fig.suptitle(field+'_'+filter, fontsize = 15)
	# ax = fig.add_subplot(111)
	# ax.grid()
	# ax.plot(epochs['EPOCH'], FWHM, '.-b', alpha = 0.7, label = 'SEEING (arcsec)')
	# ax.plot(epochs['EPOCH'], airmass, '.-g', alpha = 0.7, label = 'AIRMASS')
	# ax.legend(loc = 'upper right', fontsize='x-small')
	# ax.set_xlabel('EPOCH')
	# ax.set_xlim(np.min(epochs['EPOCH'].astype(np.int))-1, np.max(epochs['EPOCH'].astype(np.int))+1)
	#
	# plt.savefig('%s/%s/%s_%s_airmass_evo.png' % (webpath, field, field, filter), dpi = 300)
	# plt.close()

	INFO_all = []
	for epo in epochs['EPOCH']:
		INFO_file = '%s/%s/%s_%s_%s.npy' % (sharepath, field, field, epo, filter)
		if not os.path.exists(INFO_file):
			print 'No file: %s' % (INFO_file)
			continue
		INFO = np.load(INFO_file)
		INFO_all.append(INFO)
	INFO_all = np.asarray(INFO_all)
	print 'Load Done!'

	fig = plt.figure()
	fig.suptitle(field+'_'+filter, fontsize = 15)
	ax = fig.add_subplot(111)
	ax.grid()
	ax.plot(INFO_all['EPOCH'], INFO_all['SEEING'], '.-b', alpha = 0.7, label = 'SEEING (arcsec)')
	ax.plot(INFO_all['EPOCH'], INFO_all['AIRMASS'], '.-g', alpha = 0.7, label = 'AIRMASS')
	ax.legend(loc = 'upper right', fontsize='x-small')
	ax.set_xlabel('EPOCH')
	ax.set_xlim(np.min(epochs['EPOCH'].astype(np.int))-1, np.max(epochs['EPOCH'].astype(np.int))+1)

	ax2 = ax.twinx()
	ax2.plot(INFO_all['EPOCH'], INFO_all['BACK_LEVEL'], '.-r', label = 'BACK_LEVEL')
	ax2.legend(loc = 'lower right', fontsize='x-small')
	ax2.set_ylabel('BACKGROUND ADU')

	print 'SEEING'
	print 'MEDIAN:', np.median(INFO_all['SEEING'])
	print 'MAX:', np.max(INFO_all['SEEING'])
	print 'MIN:', np.min(INFO_all['SEEING'])
	print '______________________________________________________'
	print 'AIRMASS'
	print 'MEDIAN:', np.median(INFO_all['AIRMASS'])
	print 'MAX:', np.max(INFO_all['AIRMASS'])
	print 'MIN:', np.min(INFO_all['AIRMASS'])
	print '______________________________________________________'
	print 'BACK_LEVEL'
	print 'MEDIAN:', np.median(INFO_all['BACK_LEVEL'])
	print 'MAX:', np.max(INFO_all['BACK_LEVEL'])
	print 'MIN:', np.min(INFO_all['BACK_LEVEL'])
	print '______________________________________________________'

	for epo, see, air, back in zip(INFO_all['EPOCH'], INFO_all['SEEING'], INFO_all['AIRMASS'], INFO_all['BACK_LEVEL']):
		print epo, see, air, back

	plt.savefig('%s/%s/%s_%s_airmass_evo.png' % (webpath, field, field, filter), dpi = 300)
	plt.close()




if __name__ == '__main__':
	field = sys.argv[1]
	filter = sys.argv[2]
	plot = sys.argv[3]

	if plot == 'airmass':
		airmass_evo(field, filter)
	if plot == 'limits':
		if field == 'all_15':
			fields = np.loadtxt('%s/info/fields_Blind15A.txt' % (jorgepath), dtype=str)
			for fiel in fields:
				print fiel
				try:
					limit_plot(fiel, filter)
				except:
					continue
		else:
			limit_plot(field, filter)
