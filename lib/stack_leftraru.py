#!/usr/bin/python

import sys
import os
import glob
from datetime import datetime
import numpy as np
import scipy as sc
from numpy.lib.stride_tricks import as_strided
from astropy.io import fits

## Main path of files
astropath = '/home/apps/astro/data'
jorgepath = '/home/jmartinez/HiTS'

def my_stack(field, CCD, combine, band):

	startTime = datetime.now()

	epochs_file = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, band)
	if not os.path.exists(epochs_file):
		print 'No epochs file: %s' % (epochs_file)
		sys.exit()
	epochs = np.loadtxt(epochs_file, comments = '#', dtype = str)


	## loading all the images, exposure time, checking FWHM, creating variance map, subtracting sky
	exp_time, FWHM, airmass, FILTER, gain = [], [], [], [], []

	for epo in epochs[:,0]:
		print '\tLoading epoch %s' % (epo)

		ima_file = '%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits' % (astropath, field, CCD, field, CCD, epo)
		if not os.path.exists(ima_file):
			print '\t\tImage for epoch %s do not exist...' % (epo)
			continue
		hdu_ima = fits.open(ima_file)
		exp_time.append(float(hdu_ima[0].header['EXPTIME']))
		FILTER.append(hdu_ima[0].header['FILTER'][0])
		airmass.append(float(hdu_ima[0].header['AIRMASS']))
		FWHM.append(float(hdu_ima[0].header['FWHM']))
		gain.append(hdu_ima[0].header['GAINA'])
		if epo == '02':
			ref_head = hdu_ima[0].header

	FILTER = np.asarray(FILTER)
	exp_time = np.asarray(exp_time)
	airmass = np.asarray(airmass)
	gain = np.asarray(gain)
	FWHM = np.asarray(FWHM)

	print 'Ranking best epochs...'
	index_sort_FWHM = np.argsort(FWHM)

	ima_back, final_idx = [], []
	count = 0
	for idx in index_sort_FWHM:
		if FILTER[idx] != band: continue
		print '\tEpoch %s with FWHM %f' % (epochs[idx,0], FWHM[idx])

		if epochs[idx,0] == '02':
			proy_ima_file = '%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits' % (astropath, field, CCD, field, CCD, epochs[idx,0])
			proy_bac_file = glob.glob("%s/fits/%s/%s/CHECKIMAGE/%s_%s_%s_image_crblaster_internal_background64.fits" % (jorgepath, field, CCD, field, CCD, epochs[idx,0]))
		else:
			proy_ima_file = '%s/DATA/%s/%s/%s_%s_%s_image_crblaster_grid02_lanczos2.fits' % (astropath, field, CCD, field, CCD, epochs[idx,0])
			proy_bac_file = glob.glob("%s/fits/%s/%s/CHECKIMAGE/%s_%s_%s_image_crblaster_grid02_lanczos2_*_background64.fits" % (jorgepath, field, CCD, field, CCD, epochs[idx,0]))

		if not os.path.exists(proy_ima_file):
			print '\tImage for epoch %s do not exist...' % (epochs[idx,0])
			print '___________________________________'
			continue
		print proy_bac_file[0]
		proy_hdu_ima = fits.open(proy_ima_file)
		proy_imag = proy_hdu_ima[0].data
		proy_hdu_bac = fits.open(proy_bac_file[0])
		proy_back = proy_hdu_bac[0].data

		print '\tSubstracting sky'
		ima_back.append(proy_imag - proy_back)
		count += 1
		final_idx.append(idx)
		print '___________________________________'
		if count == 10: break

	FWHM = FWHM[final_idx]
	EXPTIME = exp_time[final_idx]
	GAINA = gain[final_idx]
	AIRMASS = airmass[final_idx]

	AIRMASS = np.mean(airmass)

	# sys.exit()

	print '____________________________________'
	print '%i images to combine' % (len(ima_back))
	print 'Shape of images '+str(ima_back[0].shape)

	if combine == 'MEDIAN':
		print 'Combining images'
		final_image = np.median(ima_back, axis = 0)
		EXPTIME = np.median(exp_time)
		GAINA = 2*np.mean(gain)*len(ima_file)/3

	elif combine == 'MEAN':
		print 'Combining images'
		final_image = np.mean(ima_back, axis = 0)
		EXPTIME = np.mean(exp_time)
		GAINA = np.mean(gain)*len(ima_file)

	elif combine == 'SUM':
		print 'Combining images'
		final_image = np.sum(ima_back, axis = 0)
		# if len(wtm_file) == len(ima_file): final_wtmap = add_wtmap(wtm)
		EXPTIME = np.sum(exp_time)
		GAINA = np.mean(gain)

	else: print 'Operation type not recognized\n. Aborting'

	###################################################################
	## create directories
	if not os.path.exists("%s/fits/%s" % (jorgepath, field)):
		print "Creating field folder"
		os.makedirs("%s/fits/%s" % (jorgepath, field))
	if not os.path.exists("%s/fits/%s/%s" % (jorgepath, field, CCD)):
		print "Creating CCD folder"
		os.makedirs("%s/fits/%s/%s" % (jorgepath, field, CCD))


	outfile_image = '%s/fits/%s/%s/%s_%s_image_stack_%s_%s.fits' % (jorgepath, field, CCD, field, CCD, combine, band)
	outfile_wtmap = '%s/fits/%s/%s/%s_%s_wtmap_stack_%s.fits' % (jorgepath, field, CCD, field, CCD, combine)
	ref_head['AIRMASS'] = AIRMASS
	ref_head['EXPTIME'] = EXPTIME
	ref_head['FILTER'] = band
	ref_head['GAINA'] = GAINA
	fits.writeto(outfile_image, final_image, ref_head, clobber = True)
	# if len(wtm_file) == len(ima_file):
	# 	fits.writeto(outfile_wtmap, final_wtmap, clobber = True)
	# 	print 'Weight Map Done!'
	# else :
	# 	print 'No wtmap of stack'

	print 'Stack Done!'

	print 'It took', (datetime.now()-startTime)


def swarp_stack(field, CCD, combine, band):

	startTime = datetime.now()

	epochs_file = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, band)
	if not os.path.exists(epochs_file):
		print 'No epochs file: %s' % (epochs_file)
		sys.exit()
	epochs = np.loadtxt(epochs_file, comments = '#', dtype = str)

	## loading all the images, exposure time, checking FWHM, creating variance map, subtracting sky
	to_stack = []
	exp_time, FWHM, airmass, FILTER, gain = [], [], [], [], []


	for epo in epochs[:,0]:
		print '\tLoading epoch %s' % (epo)

		ima_file = '%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits' % (astropath, field, CCD, field, CCD, epo)
		if not os.path.exists(ima_file):
			print '\t\tImage for epoch %s do not exist...' % (epo)
			continue
		hdu_ima = fits.open(ima_file)
		exp_time.append(float(hdu_ima[0].header['EXPTIME']))
		FILTER.append(hdu_ima[0].header['FILTER'][0])
		airmass.append(float(hdu_ima[0].header['AIRMASS']))
		FWHM.append(float(hdu_ima[0].header['FWHM']))
		gain.append(hdu_ima[0].header['GAINA'])
		to_stack.append(ima_file)
		if epo == epochs[0,0]:
			ref_head = hdu_ima[0].header

	FILTER = np.asarray(FILTER)
	exp_time = np.asarray(exp_time)
	airmass = np.asarray(airmass)
	gain = np.asarray(gain)
	FWHM = np.asarray(FWHM)

	EXPTIME = np.sum(exp_time)
	GAINA = np.mean(gain)
	AIRMASS = np.mean(airmass)
	ref_head['AIRMASS'] = AIRMASS
	ref_head['EXPTIME'] = EXPTIME
	ref_head['FILTER'] = band
	ref_head['GAINA'] = GAINA

	s = ' '
	to_stack_string = s.join(to_stack)
	print to_stack_string
	outfile_image = '%s/fits/%s/%s/%s_%s_image_stack_%s_%s.fits' % (jorgepath, field, CCD, field, CCD, combine, band)
	print outfile_image

	cmd = 'swarp %s -IMAGEOUT_NAME %s -COMBINE_TYPE %s' % (to_stack_string, outfile_image, combine)
	print cmd
	os.system(cmd)

	hdulist = fits.open(outfile_image)
	stack_data = hdulist[0].data
	fits.writeto(outfile_image, stack_data, ref_head, clobber=True)

	print 'Stack Done!'

	print 'It took', (datetime.now()-startTime)


def add_wtmap(wtm_list):
	ones = np.ones(wtm_list[0].shape)
	suma = np.zeros(wtm_list[0].shape)
	for k in range(len(wtm_list)):
		suma += ones/wtm_list[k]
	return ones/suma

if __name__ == '__main__':
	field = sys.argv[1]
	ccd = sys.argv[2]
	combine = sys.argv[3]
	band = sys.argv[4]
	print field[:7]

	# if field[:8] == 'Blind15A':
	# 	band = 'g'
	#
	# elif field[:8] == 'Blind14A':
	# 	band = 'g'
	#
	# elif field[:8] == 'Blind13A':
	# 	band = 'u'

	outfile_image = '%s/fits/%s/%s/%s_%s_image_stack_%s.fits' % (jorgepath, field, ccd, field, ccd, combine)
	# if not os.path.exists(outfile_image):
	if band in ['g','u']:
		my_stack(field, ccd, combine, band)
	else:
		swarp_stack(field, ccd, combine, band)
