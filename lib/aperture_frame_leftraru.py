## crea archivos reg con las detecciones de SE ue pasan un filtro dado.

import sys
import os
import glob
import re
import warnings
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn import linear_model
from astropy.stats import median_absolute_deviation
from matplotlib.patches import Circle
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

	## import stack catalogue and image
	stack_file = glob.glob("%s/catalogues/%s/%s/%s_%s_image_stack_SUM_thresh%s_minarea%i_backsize64_cat.dat" % (jorgepath, field, CCD, field, CCD, str(thresh), 5))
	stack_imaf = glob.glob("%s/fits/%s/%s/%s_%s_image_stack_SUM.fits" % (jorgepath, field, CCD, field, CCD))
	stack_cata = np.loadtxt(stack_file[0], comments='#')
	stack_hdu = fits.open(stack_imaf[0])
	stack_imag = stack_hdu[0].data
	stack_cata = stack_cata[(stack_cata[:,5] > 0) & (stack_cata[:,7] < 30) & (stack_cata[:,1]>0+bound) & (stack_cata[:,1]<2048-bound) & (stack_cata[:,2]>0+bound) & (stack_cata[:,2]<4096-bound)]
	stack_XY = np.transpose(np.array((stack_cata[:,1],stack_cata[:,2])))
	tree_XY = cKDTree(stack_XY)
	print 'Objects in stack catalogue \t\t\t\t%i' % len(stack_cata)

	## files for single epoch catalogues and image
	file_epoch = glob.glob("%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_*thresh%s_minarea%s_backsize64_cat.dat" % (jorgepath, field, CCD, field, CCD, epoch, str(thresh), str(minarea)))
	if epoch == '02':
		imag_epoch = glob.glob("%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (astropath, field, CCD, field, CCD, epoch))
		imag_epoch_bc = glob.glob("%s/DATA/%s/%s/%s_%s_%s_image.fits" % (astropath, field, CCD, field, CCD, epoch))
	else:
		imag_epoch = glob.glob("%s/DATA/%s/%s/%s_%s_%s_image_crblaster_grid02_lanczos2.fits*" % (astropath, field, CCD, field, CCD, epoch))

	## import single epoch catalogue and image
	cata = np.loadtxt(file_epoch[0], comments='#')
	imag_hdu = fits.open(imag_epoch[0])
	imag_hdu_bc = fits.open(imag_epoch_bc[0])
	imag = imag_hdu[0].data
	imag_bc = imag_hdu_bc[0].data
	cata = cata[(cata[:,5] > 0) & (cata[:,7] < 30) & (cata[:,1]>0+bound) & (cata[:,1]<2048-bound) & (cata[:,2]>0+bound) & (cata[:,2]<4096-bound)]
	XY_cata = np.transpose(np.array((cata[:,1], cata[:,2])))

	## import next epoch to compare time evo
	next_epoch = str(int(epoch) + 1).zfill(2)
	imag_epoch_next = glob.glob("%s/DATA/%s/%s/%s_%s_%s_image_crblaster_grid02_lanczos2.fits*" % (astropath, field, CCD, field, CCD, next_epoch))
	imag_hdu_next = fits.open(imag_epoch_next[0])
	imag_next = imag_hdu_next[0].data

	## finding sames objects in single epoch and stack catalogue
	superpos_ind = tree_XY.query(XY_cata, k = 1, distance_upper_bound= 1)[1]
	index_filter = (superpos_ind < len(stack_cata)) # dice los obj de single epoch encontrados en stack
	index = superpos_ind[index_filter] # dide los obj del stack encontrados en single epoch
	unmatch = np.logical_not(index_filter) # dice los obj de single epoch que no tienen match
	repeat_s = stack_cata[index] # objetos encontados en stack
	repeat_e = cata[index_filter]
	umatch_e = cata[unmatch]

	print 'Objects in single-epoch catalogue \t\t\t%i' % len(cata)
	print 'Objects with match  \t\t\t\t\t%i' % (len(repeat_e))


	name = '%s_%s_%s' % (field, CCD, epoch)

	ADU_slope, ADU_inter, ADUx, ADU_y, ADU_outlier = line_reg_RANSAC(np.log10(repeat_s[:,5]), np.log10(repeat_e[:,5]), main_path='%s/%s/%s' % (webpath, field, CCD) , pl_name = name+'_ADU', plots = PLOTS, log = False)


	header = '# Region file format: DS9 version 4.1' #\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'
	reg_file = '%s/%s/%s/%s_%s_%s_match.reg' % (webpath, field, CCD, field, CCD, epoch)
	reg_file_stack = '%s/%s/%s/%s_%s_%s-stack_match.reg' % (webpath, field, CCD, field, CCD, epoch)
	reg_file_um = '%s/%s/%s/%s_%s_%s_unmatch.reg' % (webpath, field, CCD, field, CCD, epoch)
	reg_file_ol = '%s/%s/%s/%s_%s_%s_outliers_RANSAC.reg' % (webpath, field, CCD, field, CCD, epoch)
	reg_file_ol_s = '%s/%s/%s/%s_%s_%s-stack_outliers_RANSAC.reg' % (webpath, field, CCD, field, CCD, epoch)
	reg_file_star_s = '%s/%s/%s/%s_%s_%s-stack_star.reg' % (webpath, field, CCD, field, CCD, epoch)
	reg_file_galx_s = '%s/%s/%s/%s_%s_%s-stack_galx.reg' % (webpath, field, CCD, field, CCD, epoch)
	reg_file_star_e = '%s/%s/%s/%s_%s_%s_star.reg' % (webpath, field, CCD, field, CCD, epoch)
	reg_file_galx_e = '%s/%s/%s/%s_%s_%s_galx.reg' % (webpath, field, CCD, field, CCD, epoch)
	reg_file_star_er = '%s/%s/%s/%s_%s_%s-match_star.reg' % (webpath, field, CCD, field, CCD, epoch)
	reg_file_galx_er = '%s/%s/%s/%s_%s_%s-match_galx.reg' % (webpath, field, CCD, field, CCD, epoch)

	## masking by 200 < FLUX < 1000 for matched objects
	mask_FLUX = (repeat_e[:,5] > 200) & (repeat_e[:,5] < 1000)
	mask_FLUX_stack = (np.power(repeat_s[:,5],ADU_slope) * np.power(10,ADU_inter) > 200) & (np.power(repeat_s[:,5],ADU_slope) * np.power(10,ADU_inter) < 1000)
	regions = np.array([repeat_e[mask_FLUX,1],repeat_e[mask_FLUX,2], repeat_e[mask_FLUX,10]*2], dtype = 'float')
	regions_stack = np.array([repeat_s[mask_FLUX_stack,1],repeat_s[mask_FLUX_stack,2], repeat_s[mask_FLUX_stack,10]*2], dtype = 'float')
	circle = np.empty(len(regions.T), dtype='S6')
	circle_stack = np.empty(len(regions_stack.T), dtype='S6')
	circle[:] = 'circle'
	circle_stack[:] = 'circle'
	regions = np.vstack((circle, regions)).T
	regions_stack = np.vstack((circle_stack, regions_stack)).T

	## masking for unmatched objects
	mask_FLUX_um = (umatch_e[:,5] > 200) & (umatch_e[:,5] <1000)
	regions_um = np.array([umatch_e[mask_FLUX_um,1],umatch_e[mask_FLUX_um,2], umatch_e[mask_FLUX_um,10]*2], dtype = 'float')
	circle_um = np.empty(len(regions_um.T), dtype='S6')
	circle_um[:] = 'circle'
	regions_um = np.vstack((circle_um, regions_um)).T

	## masking for outliers
	regions_ol = np.array([repeat_e[ADU_outlier,1],repeat_e[ADU_outlier,2], repeat_e[ADU_outlier,10]*2, repeat_e[ADU_outlier,5]], dtype = 'float')
	regions_ol_s = np.array([repeat_s[ADU_outlier,1],repeat_s[ADU_outlier,2], repeat_s[ADU_outlier,10]*2, repeat_s[ADU_outlier,5]], dtype = 'float')
	circle_ol = np.empty(len(regions_ol.T), dtype='S6')
	circle_ol_s = np.empty(len(regions_ol_s.T), dtype='S6')
	circle_ol[:] = 'circle'
	circle_ol_s[:] = 'circle'
	regions_ol = np.vstack((circle_ol, regions_ol)).T
	regions_ol_s = np.vstack((circle_ol_s, regions_ol_s)).T

	## masking galaxy and star like sources
	mask_shape_stack = (stack_cata[:,10] < 20) & (1-1/stack_cata[:,12] <= 0.5) & (stack_cata[:,11] >= 0.) & (stack_cata[:,13] < 10) & (stack_cata[:,9] < 3) #np.median(stack_cata[:,9]))
	mask_shape_epoch = (cata[:,10] < 20) & (1-1/cata[:,12] <= 0.5) & (cata[:,11] >= 0.) & (cata[:,13] < 10) & (cata[:,9] < 3) #np.median(cata[:,9]))
	mask_shape_repeat = (repeat_s[:,10] < 20) & (1-1/repeat_s[:,12] <= 0.5) & (repeat_s[:,11] >= 0.) & (repeat_s[:,13] < 10) & (repeat_s[:,9] < 3) #np.median(repeat_s[:,9]))
	regions_star_s = np.array([stack_cata[mask_shape_stack,1],stack_cata[mask_shape_stack,2], stack_cata[mask_shape_stack,10]*2], dtype = 'float')
	regions_galx_s = np.array([stack_cata[~mask_shape_stack,1],stack_cata[~mask_shape_stack,2], stack_cata[~mask_shape_stack,10]*2], dtype = 'float')
	regions_star_er = np.array([repeat_e[mask_shape_repeat,1],repeat_e[mask_shape_repeat,2], repeat_e[mask_shape_repeat,10]*2.2], dtype = 'float')
	regions_galx_er = np.array([repeat_e[~mask_shape_repeat,1],repeat_e[~mask_shape_repeat,2], repeat_e[~mask_shape_repeat,10]*2.2], dtype = 'float')
	regions_star_e = np.array([cata[mask_shape_epoch,1],cata[mask_shape_epoch,2], cata[mask_shape_epoch,10]*2], dtype = 'float')
	regions_galx_e = np.array([cata[~mask_shape_epoch,1],cata[~mask_shape_epoch,2], cata[~mask_shape_epoch,10]*2], dtype = 'float')
	circle_star_s = np.empty(len(regions_star_s.T), dtype='S6')
	circle_galx_s = np.empty(len(regions_galx_s.T), dtype='S6')
	circle_star_er = np.empty(len(regions_star_er.T), dtype='S6')
	circle_galx_er = np.empty(len(regions_galx_er.T), dtype='S6')
	circle_star_e = np.empty(len(regions_star_e.T), dtype='S6')
	circle_galx_e = np.empty(len(regions_galx_e.T), dtype='S6')
	circle_star_s[:] = 'circle'
	circle_galx_s[:] = 'circle'
	circle_star_er[:] = 'circle'
	circle_galx_er[:] = 'circle'
	circle_star_e[:] = 'circle'
	circle_galx_e[:] = 'circle'
	regions_star_s = np.vstack((circle_star_s, regions_star_s)).T
	regions_galx_s = np.vstack((circle_galx_s, regions_galx_s)).T
	regions_star_er = np.vstack((circle_star_er, regions_star_er)).T
	regions_galx_er = np.vstack((circle_galx_er, regions_galx_er)).T
	regions_star_e = np.vstack((circle_star_e, regions_star_e)).T
	regions_galx_e = np.vstack((circle_galx_e, regions_galx_e)).T

	## saving reg file
	np.savetxt(reg_file, regions, fmt = '%s', delimiter = ' ', header = header)
	np.savetxt(reg_file_stack, regions_stack, fmt = '%s', delimiter = ' ', header = header)
	np.savetxt(reg_file_um, regions_um, fmt = '%s', delimiter = ' ', header = header)
	np.savetxt(reg_file_ol, regions_ol, fmt = '%s', delimiter = ' ', header = header)
	np.savetxt(reg_file_ol_s, regions_ol_s, fmt = '%s', delimiter = ' ', header = header)
	with file(reg_file_star_s, 'w') as outfile:
		outfile.write('# Region file format: DS9 version 4.1\n')
		outfile.write('global color=green \n')
		np.savetxt(outfile, regions_star_s, fmt = '%s', delimiter = ' ')

	with file(reg_file_galx_s, 'w') as outfile:
		outfile.write('# Region file format: DS9 version 4.1\n')
		outfile.write('global color=blue \n')
		np.savetxt(outfile, regions_galx_s, fmt = '%s', delimiter = ' ')

	with file(reg_file_star_er, 'w') as outfile:
		outfile.write('# Region file format: DS9 version 4.1\n')
		outfile.write('global color=green \n')
		np.savetxt(outfile, regions_star_er, fmt = '%s', delimiter = ' ')

	with file(reg_file_galx_er, 'w') as outfile:
		outfile.write('# Region file format: DS9 version 4.1\n')
		outfile.write('global color=blue \n')
		np.savetxt(outfile, regions_galx_er, fmt = '%s', delimiter = ' ')

	with file(reg_file_star_e, 'w') as outfile:
		outfile.write('# Region file format: DS9 version 4.1\n')
		outfile.write('global color=green \n')
		np.savetxt(outfile, regions_star_e, fmt = '%s', delimiter = ' ')

	with file(reg_file_galx_e, 'w') as outfile:
		outfile.write('# Region file format: DS9 version 4.1\n')
		outfile.write('global color=blue \n')
		np.savetxt(outfile, regions_galx_e, fmt = '%s', delimiter = ' ')


	print 'RANSAC outliers \t\t\t\t\t%i' % (len(regions_ol))
	print '____________________________________________________________'
	print 'Star-like objects in stack \t\t\t\t%i' % (len(regions_star_s))
	print 'Galaxy-like objects in stack \t\t\t\t%i' % (len(regions_galx_s))
	print 'Fraction of star-like obj in stack \t\t\t%0.3f' % (float(len(regions_star_s))/len(stack_cata[:,5]))
	print '____________________________________________________________'
	print 'Star-like objects in epoch (filtered by epoch) \t\t%i' % (len(regions_star_e))
	print 'Galaxy-like objects in epoch (filtered by epoch) \t%i' % (len(regions_galx_e))
	print 'Fraction of star-like obj in epoch \t\t\t%0.3f' % (float(len(regions_star_e))/len(cata[:,5]))
	print '____________________________________________________________'
	print 'Star-like objects in epoch (filtered by stack) \t\t%i' % (len(regions_star_er))
	print 'Galaxy-like objects in epoch (filtered by stack) \t%i' % (len(regions_galx_er))
	print 'Fraction of star-like obj in epoch \t\t\t%0.3f' % (float(len(regions_star_er))/len(repeat_e[:,5]))
	print '____________________________________________________________'


	if PLOTS:
		fig, ax = plt.subplots(2, figsize = (12,10))
		fig.suptitle(name, fontsize = 10)
		ax[0].hist(stack_cata[:,5], bins = BINS, log = True, color = 'g', alpha = 0.7, histtype = 'stepfilled', label = 'stack')
		ax[0].hist(stack_cata[mask_shape_stack,5], bins = BINS, log = True, color = 'r', alpha = 0.5, histtype = 'stepfilled', label = 'stack shape filter')
		ax[0].legend(loc = 'upper right', fontsize='xx-small')
		ax[0].set_xscale("log")

		ax[1].hist(repeat_e[:,5], bins = BINS, log = True, color = 'g', alpha = 0.7, histtype = 'stepfilled', label = 'epoch %s' % (epoch))
		ax[1].hist(repeat_e[mask_shape_repeat,5], bins = BINS, log = True, color = 'r', alpha = 0.5, histtype = 'stepfilled', label = 'epoch %s shape filter' % (epoch))
		ax[1].legend(loc = 'upper right', fontsize='xx-small')
		ax[1].set_xscale("log")
		plt.savefig('%s/%s/%s/%s_hist.png' % (webpath, field, CCD, name), dpi = 300)

	if PLOTS:
		fig, ax = plt.subplots(len(regions_ol), 4, sharey = True)#, sharex = True)
		fig.suptitle(name+' RANSAC outliers', fontsize = 10)
		for k in range(len(regions_ol)):
			y_s, x_s = int(float(regions_ol_s[k,1])), int(float(regions_ol_s[k,2]))
			stack_imag_trim = np.log10(stack_imag[x_s-25:x_s+25, y_s-25:y_s+25] + 100)
			vmin = np.percentile(stack_imag_trim, 10)
			vmax = np.percentile(stack_imag_trim, 90)
			ax[k,0].imshow(stack_imag_trim, interpolation = 'nearest', cmap = 'gray', vmin = vmin, vmax = vmax)
			ax[k,0].set_xlim(0,60)
			ax[k,0].set_ylabel('ADU = %.0f' % (float(regions_ol_s[k,4])), fontsize = 4)
			#ax[k,0].legend(loc = 'upper left', fontsize='x-small')

			y, x = int(float(regions_ol[k,1])), int(float(regions_ol[k,2]))
			imag_trim = np.log10(imag[x-25:x+25, y-25:y+25] + 100)
			vmin = np.percentile(imag_trim, 10)
			vmax = np.percentile(imag_trim, 90)
			ax[k,1].imshow(imag_trim, interpolation = 'nearest', cmap = 'gray', vmin = vmin, vmax = vmax)
			ax[k,1].set_xlim(0,60)
			ax[k,1].set_ylabel('ADU = %.0f' % (float(regions_ol[k,4])), fontsize = 4)
			#ax[k,1].legend(loc = 'upper right', fontsize='x-small')

			y_bc, x_bc = int(float(regions_ol[k,1])), int(float(regions_ol[k,2]))
			imag_trim_bc = np.log10(imag_bc[x_bc-25:x_bc+25, y_bc-25:y_bc+25] + 100)
			vmin = np.percentile(imag_trim_bc, 10)
			vmax = np.percentile(imag_trim_bc, 90)
			ax[k,2].imshow(imag_trim_bc, interpolation = 'nearest', cmap = 'gray', vmin = vmin, vmax = vmax)
			ax[k,2].set_xlim(0,60)

			y_n, x_n = int(float(regions_ol[k,1])), int(float(regions_ol[k,2]))
			imag_trim_n = np.log10(imag_next[x_n-25:x_n+25, y_n-25:y_n+25] + 100)
			vmin = np.percentile(imag_trim_n, 10)
			vmax = np.percentile(imag_trim_n, 90)
			ax[k,3].imshow(imag_trim_n, interpolation = 'nearest', cmap = 'gray', vmin = vmin, vmax = vmax)
			ax[k,3].set_xlim(0,60)

		fig.subplots_adjust(bottom = 0.1, right = 0.6, top = 0.9, hspace=0)
		for a in fig.axes: a.set_xlim(0,60)
		plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=False)
		plt.setp([a.get_yticklabels() for a in fig.axes[:]], visible=False)
		ax[k,0].set_xlabel('Stack', fontsize = 5)
		ax[k,1].set_xlabel('Epoch %s' % (epoch), fontsize = 5)
		ax[k,2].set_xlabel('Epoch %s before crbl' % (epoch), fontsize = 5)
		ax[k,3].set_xlabel('Epoch %s' % (next_epoch), fontsize = 5)
		plt.savefig('%s/%s/%s/%s_RANSAC_outliers.png' % (webpath, field, CCD, name), dpi = 300)


if __name__ == '__main__':
	field = sys.argv[1]
	CCD = sys.argv[2]
	epoch = sys.argv[3]
	run_code(field, CCD, epoch)
