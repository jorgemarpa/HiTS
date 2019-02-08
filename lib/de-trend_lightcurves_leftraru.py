import sys
import os
import glob
import re
import warnings
import numpy as np
import scipy as sc
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.optimize import leastsq

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
sharepath = '/home/apps/astro/home/jmartinez'

def run_code(field):

	warnings.filterwarnings("ignore")

	if not os.path.exists("%s/%s" % (webpath, field)):
		print "Creating field folder"
		os.makedirs("%s/%s" % (webpath, field))
	if not os.path.exists("%s/%s/de-trend" % (webpath, field)):
		print "Creating de-trend folder"
		os.makedirs("%s/%s/de-trend" % (webpath, field))

	CHIPS = np.loadtxt('%s/scripts/ccds.txt' % (jorgepath), comments = '#', dtype = str)

	all_mag_auto = np.loadtxt('%s/lightcurves/%s/%s/%s_%s_master_mag_auto.txt' % (jorgepath, field, CHIPS[0], field, CHIPS[0]), comments = '#')
	all_err_mag_auto = np.loadtxt('%s/lightcurves/%s/%s/%s_%s_master_err_mag_auto.txt' % (jorgepath, field, CHIPS[0], field, CHIPS[0]), comments = '#')
	all_mjd = np.loadtxt('%s/lightcurves/%s/%s/%s_%s_master_mjd.txt' % (jorgepath, field, CHIPS[0], field, CHIPS[0]), comments = '#')
	all_epoch = np.loadtxt('%s/lightcurves/%s/%s/%s_%s_master_epo.txt' % (jorgepath, field, CHIPS[0], field, CHIPS[0]), comments = '#', dtype = str)
	all_coord = np.loadtxt('%s/lightcurves/%s/%s/%s_%s_master_coord.txt' % (jorgepath, field, CHIPS[0], field, CHIPS[0]), comments = '#')

	all_CCD = np.empty(len(all_mag_auto), dtype='S6')
	all_CCD[:] = CHIPS[0]
	print all_CCD.shape

	for CCD in CHIPS[0:0]:
		########################################################################### open files with LC ###################################################################################

		print 'Loading files for CCD %s' % CCD
		if CCD == 'S7':
			print 'Flag CCD'
			continue
		mag_auto = np.loadtxt('%s/lightcurves/%s/%s/%s_%s_master_mag_auto.txt' % (jorgepath, field, CCD, field, CCD), comments = '#')
		err_mag_auto = np.loadtxt('%s/lightcurves/%s/%s/%s_%s_master_err_mag_auto.txt' % (jorgepath, field, CCD, field, CCD), comments = '#')
		mjd = np.loadtxt('%s/lightcurves/%s/%s/%s_%s_master_mjd.txt' % (jorgepath, field, CCD, field, CCD), comments = '#')
		epoch =  np.loadtxt('%s/lightcurves/%s/%s/%s_%s_master_epo.txt' % (jorgepath, field, CCD, field, CCD), comments = '#', dtype = str)
		coord =  np.loadtxt('%s/lightcurves/%s/%s/%s_%s_master_coord.txt' % (jorgepath, field, CCD, field, CCD), comments = '#')

		ccdss = np.empty(len(all_mag_auto), dtype='S6')
		ccdss = CCD

		print 'Number of LC %i' % len(mag_auto)

		if (len(mag_auto) > 0) & (len(err_mag_auto) > 0) & (len(mjd) > 0) & (len(epoch) > 0) & (len(coord) > 0):
			all_mag_auto = np.vstack((all_mag_auto, mag_auto))
			all_err_mag_auto = np.vstack((all_err_mag_auto, err_mag_auto))
			all_mjd = np.vstack((all_mjd, mjd))
			all_epoch = np.vstack((all_epoch, epoch))
			all_coord = np.vstack((all_coord, coord))
			all_CCD = np.append(all_CCD, ccdss)

	if field == 'Blind15A_25':
		print 'changing mjd'
		aux_1 = all_mjd[0,-1]
		aux_2 = all_mjd[0,-2]
		all_mjd[0,-1] = aux_2
		all_mjd[0,-2] = aux_1

	print '__________________________________________________________'
	print 'Number of LC %i' % all_mag_auto.shape[0]
	print 'Number of epochs %i' % all_mag_auto.shape[1]

	#all_mag_auto = all_mag_auto[:,0:25]
	#all_err_mag_auto = all_err_mag_auto[:,0:25]
	#all_mjd = all_mjd[:,0:25]

	print all_mag_auto.shape

	median_mag_lc = np.median(all_mag_auto, axis = 1)
	mask_mag = (median_mag_lc > 17) & (median_mag_lc < 21)

	trend = master_Trend(all_mag_auto[mask_mag], all_err_mag_auto[mask_mag])


	if False:
		name = '%s' % (field)
		plt.clf()
		fig, ax = plt.subplots(1, figsize = (7,5))
		fig.suptitle(name, fontsize = 15)

		ax.hist(np.median(all_mag_auto, axis = 1), bins = 100, color = 'g', alpha = 0.7, histtype = 'stepfilled', normed = False, label = 'median mag light-curves')
		ax.legend(loc = 'upper right', fontsize='xx-small')
		ax.set_xlabel('Median mag of LC')
		ax.set_ylabel('N')

		plt.savefig('%s/%s/de-trend/%s_median_mag_hist.png' % (webpath, field, field), dpi = 300, bbox_inches = 'tight')

	if True:
		name = '%s' % (field)
		plt.clf()
		fig, ax = plt.subplots(1, figsize = (7,5))
		fig.suptitle(name, fontsize = 15)

		ax.grid()
		ax.plot(all_epoch[0], trend, '.-b', label = 'Master Trend')
		#ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		ax.legend(loc = 'upper right', fontsize='xx-small')
		ax.set_xlabel('EPOCH')
		ax.invert_yaxis()

		plt.savefig('%s/%s/de-trend/%s_trend.png' % (webpath, field, field), dpi = 300, bbox_inches = 'tight')

	all_norm_mag = np.zeros(all_mag_auto.shape)
	all_norm_err_mag = np.zeros(all_err_mag_auto.shape)
	all_detr_mag = np.zeros(all_mag_auto.shape)

	print 'Normalization and de-trending'
	beta_guess = 1.0

	for k in range(0,len(all_norm_mag)):
		name = '%s_%s_%i_%i       Jorge MP' % (field, all_CCD[k], all_coord[k,0], all_coord[k,1])

		plt.clf()
		fig, ax = plt.subplots(2, figsize = (15,6), sharex = True)
		fig.suptitle(name, fontsize = 15)

		ax[0].errorbar(all_epoch[0], all_mag_auto[k], yerr = all_err_mag_auto[k], fmt = '.--', color = 'b', label = 'LC, std %f' % np.std(all_mag_auto[k]))
		ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		ax[0].legend(loc = 'upper right', fontsize='xx-small')
		ax[0].set_ylabel('MAG')
		ax[0].set_xlim(0, int(all_epoch[k,-1]) + 1)
		#ax[0].invert_yaxis()

		all_norm_mag[k,:], all_norm_err_mag[k,:] = Normalize(all_mag_auto[k], all_err_mag_auto[k])

		#ax[1].errorbar(all_epoch[0], all_norm_mag[k], yerr = all_norm_err_mag[k], fmt = '.-', color = 'b', alpha = 0.5, label = 'Normalied LC, std %f' % np.std(all_norm_mag[k]))
		#ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

		beta, cov, infodict, mesg, ier = leastsq(residual, beta_guess, args = (trend, all_norm_mag[k]), full_output = True)
		all_detr_mag[k] = all_norm_mag[k] - linear_comb(beta, trend)
		all_detr_mag[k] += 1
		all_detr_mag[k] *= np.median(all_mag_auto[k])
		print 'beta = %f' % beta
		ax[1].errorbar(all_epoch[0], all_detr_mag[k], yerr = all_err_mag_auto[k], fmt = '.--', color = 'g', label = 'De-trended LC, std %f' % np.std(all_detr_mag[k]))
		print '__________________________________________________________'

		ax[1].set_xlabel('EPOCH')
		ax[1].set_xlim(0, int(all_epoch[k,-1]) + 1)
		ax[1].legend(loc = 'upper right', fontsize='xx-small')
		#ax[1].invert_yaxis()

		fig.subplots_adjust(hspace=0)
		plt.savefig('%s/%s/de-trend/%s_%s_%i.png' % (webpath, field, field, all_CCD[k], k), dpi = 300, bbox_inches = 'tight')

	print 'Done!'


############################################################################## Calculate main trend in LC ####################################################################################
def master_Trend(LCs, errs):

	print 'Shape of LCs to de-trend: ', LCs.shape
	suma_top = np.zeros(LCs.shape[1])
	suma_bot = 0

	for k in range(len(LCs)):
		norm, err_norm = Normalize(LCs[k], errs[k])
		weight = 1./np.var(norm)
		suma_top[:] += norm[:]*weight
		suma_bot += weight

	return suma_top/suma_bot



####################################################################################### Normalize LC #########################################################################################
def Normalize(mag, err):
	mag_norm = (mag / np.mean(mag)) - 1
	#err_norm = err/np.mean(mag)
	err_norm = np.sqrt((err/mag)**2 + (np.std(mag)/np.mean(mag))**2)
	return mag_norm, err_norm


def linear_comb(beta, Tk):
	return beta*Tk


def residual(beta, Tk, Lk):
	return Lk - linear_comb(beta,Tk)


if __name__ == '__main__':
	field = sys.argv[1]
	run_code(field)
