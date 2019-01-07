## analiza los catalogos para encontrar magnitud limite
## interpolacion para encontrar la magnitud limite

from datetime import datetime
import errno
import sys
import os
import re
import warnings
import getopt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
# import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from empty_aperture_leftraru import empty_aperture_all_CCD
from astropy.table import Table, vstack
from misc_func_leftraru import *
# import seaborn as sns
from matplotlib.ticker import MaxNLocator
# sns.set(style="white", color_codes=True, context="poster")

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
sharepath = '/home/apps/astro/home/jmartinez'
number = 75      # number of bins for histograms to calculate limit
limit = 0.5       # percentage of recoveries
thresh = 1.0      # threshold of catalog
bound = 200		  # widht of frame
minarea = 1      # min area for detection

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


def get_partial_info(field, epoch, band='g', plots=False, verbose=False):

	warnings.filterwarnings("ignore")

	CHIPS = np.loadtxt('%s/info/ccds.txt' % (jorgepath), comments = '#', dtype = str)
	full_epoch = []
	cata_epoch = []
	SEEING = []
	ZP, Ag, Kg = [], [], []
	if band == 'u':
		corr = 'ZA'
	else:
		corr = 'ZP'

	if not os.path.exists("%s/%s" % (sharepath, field)):
		try:
			print "Creating field folder"
			os.makedirs("%s/%s" % (sharepath, field))
		except OSError, e:
			if e.errno != 17:
				raise
			pass

	####### Loading catalogs,one epoch all CCD and computing crossmatch with stack ##############

	for CCD in CHIPS:
		# if CCD == 'N10': break
		print 'Loading CCD %s' % (CCD)

		############# files for single epoch catalogues from unprojected images ################

		files_epoch = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final.dat"\
		            % (jorgepath, field, CCD, field, CCD, epoch, str(thresh), str(minarea))
		if not os.path.exists(files_epoch):
			print 'No file: %s' % files_epoch
			continue

		###################### import single epoch catalogue and ZP ###########################

		cata = Table.read(files_epoch, format = 'ascii')

		ZP_file = '%s/info/%s/%s/ZP_AUTO_PS_%s_%s_%s.npy' % (jorgepath, field, CCD, field, CCD, epoch)
		ZP_bool = True
		if not os.path.exists(ZP_file):
			print 'No ZP file'
			print 'Loading DECam coeff...'

			imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (astropath, field, CCD, field, CCD, epoch)
			hdufits = fits.open(imag_file)
			EXP_TIME = float(hdufits[0].header['EXPTIME'])
			CCDID = int(hdufits[0].header['CCDNUM'])
			CTE_file = np.loadtxt('%s/sextractor/zeropoint/psmFitDES-mean-%s.csv' % (jorgepath, band),
                        	delimiter = ',', skiprows = 1, usecols = [4,5,6,7,8,9,10,11,19], dtype = str)
			Ag = float(CTE_file[CCDID-1][2])
			err_Ag = float(CTE_file[CCDID-1][3])
			Kg = float(CTE_file[CCDID-1][6])
			err_Kg = float(CTE_file[CCDID-1][7])
			ZP_bool = False
		else:
			ZP.append(np.load(ZP_file)['ZP_PS'])

		########################### filtering by bounds ADU > 0 and MAG < 30 ####################

		full_epoch.append(cata)  # full catalogue
		cata = cata[(cata['FLUX_AUTO'] > 0) & (cata['MAG_AUTO_%s' % corr] < 30) & \
					(cata['X_IMAGE']>0+bound) & (cata['X_IMAGE']<2048-bound) &\
					(cata['Y_IMAGE']>0+bound) & (cata['Y_IMAGE']<4096-bound)]
		cata_epoch.append(cata)

	############################### Directories for figures ###############################

	if not os.path.exists("%s/%s" % (webpath, field)):
		try:
			print "Creating field folder"
			os.makedirs("%s/%s" % (webpath, field))
		except OSError, e:
			if e.errno != 17:
				raise
			pass
	if not os.path.exists("%s/%s/%s" % (webpath, field, epoch)):
		try:
			print "Creating field folder"
			os.makedirs("%s/%s/%s" % (webpath, field, epoch))
		except OSError, e:
			if e.errno != 17:
				raise
			pass

	######################################################################## Final catalogs ##########################################################################

	if len(full_epoch) > 0:
		full_epoch = vstack(full_epoch[:])			# unfiltered stack catalogue
		cata_epoch = vstack(cata_epoch[:])			# filtered stack catalogueatalogue
	else:
		print 'NO catalogues to work!!'
		sys.exit()

	print '_________________________________________________________________________'
	print '_________________________________________________________________________'

	################################## Values for ADU -> Mag ##################################

	fits_file = '%s/DATA/%s/N4/%s_N4_%s_image_crblaster.fits' % (astropath, field, field, epoch)
	if not os.path.exists(fits_file):
		print 'No image file: %s' % (fits_file)
	hdufits = fits.open(fits_file)
	AIRMASS = float(hdufits[0].header['AIRMASS'])
	EXP_TIME = float(hdufits[0].header['EXPTIME'])
	MJD = float(hdufits[0].header['MJD-OBS'])
	FILTER = hdufits[0].header['FILTER'][0]
	background = background_level(hdufits[0].data, 1, 1, np.median)
	if ZP_bool:
		ZP = np.vstack(ZP[:])
		ZP_PS = np.median(ZP[:,0])
	else:
		ZP_PS = - np.median(Ag) - np.median(Kg)*AIRMASS

	print 'FWHM ',np.median(cata_epoch['FWHM_IMAGE']), np.std(cata_epoch['FWHM_IMAGE'])

	print 'FLAGS', np.median(cata_epoch['FLAGS'])
	print 'FLUX_RADIUS', np.median(cata_epoch['FLUX_RADIUS']), np.std(cata_epoch['FLUX_RADIUS'])

	############################### Masking and completeness mag ###############################
	## filtering only starlike objects in stack catalogue
	if band == 'g':
		flux_rad = 3.4
		n_ap = 1000
	elif band == 'r':
		flux_rad = 5.5
		n_ap = 1000
	elif band == 'i':
		flux_rad = 5.5
		n_ap = 1000
	elif band == 'u':
		flux_rad = 3.1
		n_ap = 500
	mask_SHAPE_ce = (cata_epoch['FWHM_IMAGE'] < 20) & \
					(1-1/cata_epoch['ELONGATION'] <= 0.5) & \
					(cata_epoch['CLASS_STAR'] >= 0.) & \
					(cata_epoch['FLAGS'] < 10) & \
					(cata_epoch['FLUX_RADIUS'] < flux_rad)

	SEEING = 0.27*np.median(cata_epoch['FWHM_IMAGE'][mask_SHAPE_ce])

	######################################### Limit MAG ########################################

	std_empty_ap = empty_aperture_all_CCD(field, epoch, plots=False, verbose=False, n_aper=n_ap)
	limit_mag_empty_5 = -2.5*np.log10(5*std_empty_ap) + 2.5*np.log10(EXP_TIME) + ZP_PS
	limit_mag_empty_3 = -2.5*np.log10(3*std_empty_ap) + 2.5*np.log10(EXP_TIME) + ZP_PS

	compl_adu_sl = None
	compl_mag_sl = None

	########################################################################### Save INFO #############################################################################

	INFO = np.array((FILTER, epoch, MJD, EXP_TIME, SEEING, AIRMASS, background, \
					compl_adu_sl, compl_mag_sl, std_empty_ap, limit_mag_empty_3, \
					limit_mag_empty_5, ZP_PS), \
	dtype = {'names':['FILTER', 'EPOCH', 'MJD', 'EXP_TIME', 'SEEING', 'AIRMASS', 'BACK_LEVEL', \
					  'COMPLET_ADU', 'COMPLET_MAG', 'STD_EMPTY_AP', 'LIMIT_MAG_EA3', \
					  'LIMIT_MAG_EA5', 'ZP_PS'],
			 'formats':['S1', 'S2', 'float', 'float', 'float', 'float', 'float',\
			 'float', 'float', 'float', 'float', 'float', 'float']})
	INFO_name = '%s/info/%s/%s_%s_%s' % (jorgepath, field, field, epoch, FILTER)
	np.save(INFO_name, INFO)

	print 'Total objects in Field single-epoch \t\t%i' % (len(full_epoch))
	print '_________________________________________________________________________'
	print 'Median Seeing (arcsec) \t\t\t\t%.5f' % (np.median(SEEING))
	print 'AIRMASS \t\t\t\t\t%.3f' % (AIRMASS)
	print 'Fraction of stars-like sources in epoch \t%.3f' % \
			(float(len(cata_epoch['FLUX_AUTO'][mask_SHAPE_ce]))/len(cata_epoch['FLUX_AUTO']))
	print 'Background level (ADU) \t\t\t\t%.1f' % (background)
	print 'Limit magnitude (SNR = 3) (from empty ap)\t%f' % (limit_mag_empty_3)
	print 'Limit magnitude (SNR = 5) (from empty ap)\t%f' % (limit_mag_empty_5)
	print '_________________________________________________________________________'



def get_full_info(field, epoch, band='g', plots=False, verbose=False):

	warnings.filterwarnings("ignore")

	CHIPS = np.loadtxt('%s/info/ccds.txt' % (jorgepath), comments = '#', dtype = str)
	full_epoch, full_stack = [], []
	cata_epoch, cata_stack = [], []
	repeat_epoch, repeat_stack = [], []
	umatch_epoch = []
	false_pos = []
	total = 0
	total_match = 0
	SEEING = []
	ZP, Ag, Kg = [], [], []
	if band == 'u':
		corr = 'ZA'
	else:
		corr = 'ZP'

	if not os.path.exists("%s/%s" % (sharepath, field)):
		try:
			print "Creating field folder"
			os.makedirs("%s/%s" % (sharepath, field))
		except OSError, e:
			if e.errno != 17:
				raise
			pass

	####### Loading catalogs,one epoch all CCD and computing crossmatch with stack ##############

	for CCD in CHIPS:
		# if CCD == 'N10': break
		print 'Loading CCD %s' % (CCD)

		## skiping ccds without stack image
		stack_file = "%s/catalogues/%s/%s/%s_%s_image_stack_SUM_thresh%s_minarea%s_backsize64_cat.dat"\
		 				% (jorgepath, field, CCD, field, CCD, str(thresh), 5)
		# stack_file = "%s/catalogues/%s/%s/%s_%s_image_stack_SUM_%s_thresh%s_minarea%s_backsize64_cat.dat"\
		#  				% (jorgepath, field, CCD, field, CCD, band, str(thresh), 5)
		if not os.path.exists(stack_file):
			print 'No file: %s' % stack_file
			continue

		############################ import stack catalogue and filter  #########################

		stack_cata = Table.read(stack_file, format = 'ascii')
		full_stack.append(stack_cata)  # full catalogue
		stack_cata = stack_cata[(stack_cata['FLUX_AUTO'] > 0) & \
		 			(stack_cata['X_IMAGE']>0+bound) & (stack_cata['X_IMAGE']<2048-bound) & \
					(stack_cata['Y_IMAGE']>0+bound) & (stack_cata['Y_IMAGE']<4096-bound)]
		stack_XY = np.transpose(np.array((stack_cata['X_IMAGE'],stack_cata['Y_IMAGE'])))
		tree_XY = cKDTree(stack_XY)
		print 'Lenght of stack catalogue %i' % len(stack_cata)

		############# files for single epoch catalogues from unprojected images ################

		files_epoch = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final.dat"\
		            % (jorgepath, field, CCD, field, CCD, epoch, str(thresh), str(minarea))
		if not os.path.exists(files_epoch):
			print 'No file: %s' % files_epoch
			continue

		###################### import single epoch catalogue and ZP ###########################

		cata = Table.read(files_epoch, format = 'ascii')

		ZP_file = '%s/info/%s/%s/ZP_AUTO_PS_%s_%s_%s.npy' % (jorgepath, field, CCD, field, CCD, epoch)
		ZP_bool = True
		if not os.path.exists(ZP_file):
			print 'No ZP file'
			print 'Loading DECam coeff...'

			imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (astropath, field, CCD, field, CCD, epoch)
			hdufits = fits.open(imag_file)
			EXP_TIME = float(hdufits[0].header['EXPTIME'])
			CCDID = int(hdufits[0].header['CCDNUM'])
			CTE_file = np.loadtxt('%s/sextractor/zeropoint/psmFitDES-mean-%s.csv' % (jorgepath, band),
                        	delimiter = ',', skiprows = 1, usecols = [4,5,6,7,8,9,10,11,19], dtype = str)
			Ag = float(CTE_file[CCDID-1][2])
			err_Ag = float(CTE_file[CCDID-1][3])
			Kg = float(CTE_file[CCDID-1][6])
			err_Kg = float(CTE_file[CCDID-1][7])
			ZP_bool = False
		else:
			ZP.append(np.load(ZP_file)['ZP_PS'])

		########################### filtering by bounds ADU > 0 and MAG < 30 ####################

		full_epoch.append(cata)  # full catalogue
		cata = cata[(cata['FLUX_AUTO'] > 0) & (cata['MAG_AUTO_%s' % corr] < 30) & \
					(cata['X_IMAGE']>0+bound) & (cata['X_IMAGE']<2048-bound) &\
					(cata['Y_IMAGE']>0+bound) & (cata['Y_IMAGE']<4096-bound)]
		XY_cata = np.transpose(np.array((cata['X_IMAGE'], cata['Y_IMAGE'])))

		################## finding same objects in single epoch and stack catalogue #############

		superpos_ind = tree_XY.query(XY_cata, k = 1, distance_upper_bound= 4)[1]
		index_filter = (superpos_ind < len(stack_cata)) # dice los obj de single epoch encontrados en stack
		index = superpos_ind[index_filter]
		unmatch = np.logical_not(index_filter)			# dice los obj de single epoch que no tienen match
		repeat_stack.append(stack_cata[index]) 			# objetos encontados en stack
		repeat_epoch.append(cata[index_filter]) 		# objetos encontados en single epoch
		cata_stack.append(stack_cata)  					# filtered catalogue
		cata_epoch.append(cata)   						# filtered catalogue
		umatch_epoch.append(cata[unmatch])				#unmatched objects in epoch

		name = re.findall(r'Blind\w+\_\d+\_\w\d+\_\d+\_', files_epoch)
		false_pos.append(float(len(index))/len(cata))
		total += len(cata)
		total_match += len(index)
		SEEING.append(0.27*np.median(cata['FWHM_IMAGE']))


		print 'CATALOGUE\t\t|  OBJECTS\t|  SEEING\t|  MATCH | M/O'
		print '%s\t|  %i\t\t|  %.4f\t|  %d\t | %f' % (name[0], len(cata), \
			0.27*np.median(cata['FWHM_IMAGE']), len(index), float(len(index))/len(cata))


	############################### Directories for figures ###############################

	if not os.path.exists("%s/%s" % (webpath, field)):
		try:
			print "Creating field folder"
			os.makedirs("%s/%s" % (webpath, field))
		except OSError, e:
			if e.errno != 17:
				raise
			pass
	if not os.path.exists("%s/%s/%s" % (webpath, field, epoch)):
		try:
			print "Creating field folder"
			os.makedirs("%s/%s/%s" % (webpath, field, epoch))
		except OSError, e:
			if e.errno != 17:
				raise
			pass

	######################################################################## Final catalogs ##########################################################################

	if len(full_epoch) > 0:
		full_epoch = vstack(full_epoch[:])			# unfiltered stack catalogue
		full_stack = vstack(full_stack[:])			# unfiltered single epoch catalogue
		cata_epoch = vstack(cata_epoch[:])			# filtered stack catalogue
		cata_stack = vstack(cata_stack[:])			# filtered single epoch catalogue
		repeat_epoch = vstack(repeat_epoch[:])		# matched stack catalogue
		repeat_stack = vstack(repeat_stack[:])		# matched single epoch catalogue
		umatch_epoch = vstack(umatch_epoch[:])		# unmatched single epoch catalogue
	else:
		print 'NO catalogues to work!!'
		sys.exit()

	print '_________________________________________________________________________'
	print '_________________________________________________________________________'

	################################## Values for ADU -> Mag ##################################

	fits_file = '%s/DATA/%s/N4/%s_N4_%s_image_crblaster.fits' % (astropath, field, field, epoch)
	if not os.path.exists(fits_file):
		print 'No image file: %s' % (fits_file)
	hdufits = fits.open(fits_file)
	AIRMASS = float(hdufits[0].header['AIRMASS'])
	EXP_TIME = float(hdufits[0].header['EXPTIME'])
	MJD = float(hdufits[0].header['MJD-OBS'])
	FILTER = hdufits[0].header['FILTER'][0]
	background = background_level(hdufits[0].data, 1, 1, np.median)
	if ZP_bool:
		ZP = np.vstack(ZP[:])
		ZP_PS = np.median(ZP[:,0])
	else:
		ZP_PS = - np.median(Ag) - np.median(Kg)*AIRMASS

	print 'ZP: ', ZP_PS
	#################################### Linear Regression ###################################

	BINS = np.logspace(1, 7, number)
	name = '%s_%s_bin%s_limit%s_thresh%s_minarea%s_zp' % \
			(field, epoch, str(number), str(limit), str(thresh), str(minarea))

	ADU_slope, ADU_inter, ADUx, ADU_y, ADU_outlier = reg_RANSAC(np.log10(repeat_stack['FLUX_AUTO']), \
							np.log10(repeat_epoch['FLUX_AUTO']), \
							main_path='%s/%s/%s' % (webpath, field, epoch) , \
							pl_name = name+'_ADU', plots=plots, log=True)

	############################### Masking and completeness mag ###############################

	# compl_adu = compl_adu_bin(repeat_epoch['FLUX_AUTO'], \
	# 			np.power(cata_stack['FLUX_AUTO'],ADU_slope) * np.power(10,ADU_inter),\
	# 			BINS, pl_name = name, plots = False, file = False)
	# compl_mag = -2.5*np.log10(compl_adu) + 2.5*np.log10(EXP_TIME) + ZP_PS

	## filtering only starlike objects in stack catalogue
	if field[5:7] == '13':
		mask_SHAPE_rs = (repeat_stack['FWHM_IMAGE'] < 20) & \
						(1-1/repeat_stack['ELONGATION'] <= 0.5) & \
						(repeat_stack['CLASS_STAR'] >= 0.) & \
						(repeat_stack['FLAGS'] < 10) & \
						(repeat_stack['FLUX_RADIUS'] < 3.0)
		mask_SHAPE_re = (repeat_epoch['FWHM_IMAGE'] < 20) & \
						(1-1/repeat_epoch['ELONGATION'] <= 0.5) & \
						(repeat_epoch['CLASS_STAR'] >= 0.) & \
						(repeat_epoch['FLAGS'] < 10) & \
						(repeat_epoch['FLUX_RADIUS'] < 3.0)
		mask_SHAPE_cs = (cata_stack['FWHM_IMAGE'] < 20) & \
						(1-1/cata_stack['ELONGATION'] <= 0.5) & \
						(cata_stack['CLASS_STAR'] >= 0.) & \
						(cata_stack['FLAGS'] < 10) & \
						(cata_stack['FLUX_RADIUS'] < 3.1)
		mask_SHAPE_ce = (cata_epoch['FWHM_IMAGE'] < 20) & \
						(1-1/cata_epoch['ELONGATION'] <= 0.5) & \
						(cata_epoch['CLASS_STAR'] >= 0.) & \
						(cata_epoch['FLAGS'] < 10) & \
						(cata_epoch['FLUX_RADIUS'] < 3.1)
	if field[5:7] == '14':
		mask_SHAPE_rs = (repeat_stack['FWHM_IMAGE'] < 20) & \
						(1-1/repeat_stack['ELONGATION'] <= 0.5) & \
						(repeat_stack['CLASS_STAR'] >= 0.) & \
						(repeat_stack['FLAGS'] < 10) & \
						(repeat_stack['FLUX_RADIUS'] < 3.3)
		mask_SHAPE_re = (repeat_epoch['FWHM_IMAGE'] < 20) & \
						(1-1/repeat_epoch['ELONGATION'] <= 0.5) & \
						(repeat_epoch['CLASS_STAR'] >= 0.) & \
						(repeat_epoch['FLAGS'] < 10) & \
						(repeat_epoch['FLUX_RADIUS'] < 3.3)
		mask_SHAPE_cs = (cata_stack['FWHM_IMAGE'] < 20) & \
						(1-1/cata_stack['ELONGATION'] <= 0.5) & \
						(cata_stack['CLASS_STAR'] >= 0.) & \
						(cata_stack['FLAGS'] < 10) & \
						(cata_stack['FLUX_RADIUS'] < 3.4)
		mask_SHAPE_ce = (cata_epoch['FWHM_IMAGE'] < 20) & \
						(1-1/cata_epoch['ELONGATION'] <= 0.5) & \
						(cata_epoch['CLASS_STAR'] >= 0.) & \
						(cata_epoch['FLAGS'] < 10) & \
						(cata_epoch['FLUX_RADIUS'] < 3.4)
	elif field[5:7] == '15':
		mask_SHAPE_rs = (repeat_stack['FWHM_IMAGE'] < 20) & \
						(1-1/repeat_stack['ELONGATION'] <= 0.5) & \
						(repeat_stack['CLASS_STAR'] >= 0.) & \
						(repeat_stack['FLAGS'] < 10) & \
						(repeat_stack['FLUX_RADIUS'] < 2.9)
		mask_SHAPE_re = (repeat_epoch['FWHM_IMAGE'] < 20) & \
						(1-1/repeat_epoch['ELONGATION'] <= 0.5) & \
						(repeat_epoch['CLASS_STAR'] >= 0.) & \
						(repeat_epoch['FLAGS'] < 10) & \
						(repeat_epoch['FLUX_RADIUS'] < 2.9)
		mask_SHAPE_cs = (cata_stack['FWHM_IMAGE'] < 20) & \
						(1-1/cata_stack['ELONGATION'] <= 0.5) & \
						(cata_stack['CLASS_STAR'] >= 0.) & \
						(cata_stack['FLAGS'] < 10) & \
						(cata_stack['FLUX_RADIUS'] < 3.)
		mask_SHAPE_ce = (cata_epoch['FWHM_IMAGE'] < 20) & \
						(1-1/cata_epoch['ELONGATION'] <= 0.5) & \
						(cata_epoch['CLASS_STAR'] >= 0.) & \
						(cata_epoch['FLAGS'] < 10) & \
						(cata_epoch['FLUX_RADIUS'] < 3.)


	compl_adu_sl = compl_adu_bin(repeat_epoch['FLUX_AUTO'][mask_SHAPE_rs], \
				np.power(cata_stack['FLUX_AUTO'][mask_SHAPE_cs],ADU_slope) * np.power(10,ADU_inter), \
				BINS, pl_name = name+'_starlike', plots = False, file = False, \
				params = [EXP_TIME, ZP_PS])
	compl_mag_sl = -2.5*np.log10(compl_adu_sl) + 2.5*np.log10(EXP_TIME) + ZP_PS
	print compl_adu_sl, compl_mag_sl
	#sys.exit()

	SEEING = 0.27*np.median(repeat_epoch['FWHM_IMAGE'][mask_SHAPE_rs])

	########################################### ROC curve #######################################

	ADU_arange = np.logspace(0,5,250)
	TPR = np.zeros(len(ADU_arange))
	FPR = np.zeros(len(ADU_arange))
	for k in range(len(ADU_arange)):
		## filtering all object with ADU > limite
		limite = ADU_arange[k]
		cum_cata_stack = cata_stack[(np.power(cata_stack['FLUX_AUTO'],ADU_slope) * np.power(10,ADU_inter) > 310)]
		cum_cata_epoch = cata_epoch[(cata_epoch['FLUX_AUTO'] > limite)]
		cum_repe_epoch = repeat_epoch[(repeat_epoch['FLUX_AUTO'] > limite)]
		FPR[k] = 1 - (float(len(cum_repe_epoch)))/len(cum_cata_epoch)
		TPR[k] = (float(len(cum_repe_epoch)))/len(cum_cata_stack)

	cum_cata_stack = cata_stack[(np.power(cata_stack['FLUX_AUTO'],ADU_slope) * np.power(10,ADU_inter) > 310)]
	cum_cata_epoch = cata_epoch[(cata_epoch['FLUX_AUTO'] > compl_adu_sl)]
	cum_repe_epoch = repeat_epoch[(repeat_epoch['FLUX_AUTO'] > compl_adu_sl)]
	FPR_limit = 1 - (float(len(cum_repe_epoch)))/len(cum_cata_epoch)
	TPR_limit = (float(len(cum_repe_epoch)))/len(cum_cata_stack)

	#FPR_limit = (FPR_limit - np.min(FPR))/(np.max(FPR) - np.min(FPR))
	#TPR_limit = (TPR_limit - np.min(TPR))/(np.max(TPR) - np.min(TPR))
	#FPR_norm = (FPR - np.min(FPR))/(np.max(FPR) - np.min(FPR))
	#TPR_norm = (TPR - np.min(TPR))/(np.max(TPR) - np.min(TPR))

	######################################### Limit MAG ########################################
	if band == 'g':
		n_ap = 1000
	elif band == 'r':
		n_ap = 1000
	elif band == 'i':
		n_ap = 1000
	elif band == 'u':
		n_ap = 500

	std_empty_ap = empty_aperture_all_CCD(field, epoch, plots=False, verbose=False, n_aper=n_ap)
	limit_mag_empty_5 = -2.5*np.log10(5*std_empty_ap) + 2.5*np.log10(EXP_TIME) + ZP_PS
	limit_mag_empty_3 = -2.5*np.log10(3*std_empty_ap) + 2.5*np.log10(EXP_TIME) + ZP_PS

	########################################################################### Save INFO #############################################################################

	INFO = np.array((FILTER, epoch, MJD, EXP_TIME, SEEING, AIRMASS, background, \
					compl_adu_sl, compl_mag_sl, std_empty_ap, limit_mag_empty_3, \
					limit_mag_empty_5, ZP_PS), \
	dtype = {'names':['FILTER', 'EPOCH', 'MJD', 'EXP_TIME', 'SEEING', 'AIRMASS', 'BACK_LEVEL', \
					  'COMPLET_ADU', 'COMPLET_MAG', 'STD_EMPTY_AP', 'LIMIT_MAG_EA3', \
					  'LIMIT_MAG_EA5', 'ZP_PS'],
			 'formats':['S1', 'S2', 'float', 'float', 'float', 'float', 'float',\
			 'float', 'float', 'float', 'float', 'float', 'float']})
	INFO_name = '%s/info/%s/%s_%s_%s' % (jorgepath, field, field, epoch, FILTER)
	np.save(INFO_name, INFO)

	print 'Total objects in Field single-epoch \t\t%i' % (total)
	print 'Total objects in Field matched \t\t\t%i' % (total_match)
	print 'Ratio of Recoveries \t\t\t\t%.5f' % (np.mean(false_pos))
	print '_________________________________________________________________________'
	print 'Median Seeing (arcsec) \t\t\t\t%.5f' % (np.median(SEEING))
	print 'AIRMASS \t\t\t\t\t%.3f' % (AIRMASS)
	print 'Seeing with star-like (arcsec) \t\t\t%.3f' % \
			(0.27*np.median(repeat_epoch['FWHM_IMAGE'][mask_SHAPE_rs]))
	print 'Fraction of stars-like sources in stack \t%.3f' % \
			(float(len(cata_stack['FLUX_AUTO'][mask_SHAPE_cs]))/len(cata_stack['FLUX_AUTO']))
	print 'Fraction of stars-like sources in epoch \t%.3f' % \
			(float(len(cata_epoch['FLUX_AUTO'][mask_SHAPE_ce]))/len(cata_epoch['FLUX_AUTO']))
	print 'Background level (ADU) \t\t\t\t%.1f' % (background)
	print 'Completeness magnitude (%.2f)\t\t\t%.3f' % (limit, compl_mag_sl)
	print 'Limit magnitude (SNR = 3) (from empty ap)\t%f' % (limit_mag_empty_3)
	print 'Limit magnitude (SNR = 5) (from empty ap)\t%f' % (limit_mag_empty_5)
	print '_________________________________________________________________________'

    ############################################## plots ##############################################
	if plots:

		print 'Ploting Hist and RANSAC Figures'
		fig, ax = plt.subplots(2, figsize = (16,9))
		fig.suptitle(name, fontsize = 20)

		ax[0].hist(np.power(cata_stack['FLUX_AUTO'],ADU_slope) * np.power(10,ADU_inter), bins = BINS, log = True, color = 'g', alpha = 0.7, histtype = 'stepfilled', normed = False, label = 'Full Stack')
		ax[0].hist(cata_epoch['FLUX_AUTO'], bins = BINS, log = True, color = 'r', alpha = 0.5, histtype = 'stepfilled', normed = False, label = 'Single epoch')
		ax[0].hist(repeat_epoch['FLUX_AUTO'], bins = BINS, log = True, color = 'y', alpha = 0.5, histtype = 'stepfilled', normed = False, label = 'match single epoch')
		ax[0].hist(np.power(cata_stack['FLUX_AUTO'][mask_SHAPE_cs],ADU_slope) * np.power(10,ADU_inter), bins = BINS, log = True, color = 'c', alpha = 0.9, histtype = 'step', normed = False, label = 'Full Stack, shape filter')
		ax[0].hist(repeat_epoch['FLUX_AUTO'][mask_SHAPE_rs], bins = BINS, log = True, color = 'm', alpha = 0.9, histtype = 'step', normed = False, label = 'match single epoch, shape filter')
		#ax[0].hist(repeat_epoch[ADU_outlier,5], bins = BINS, log = True, color = 'b', alpha = 0.2, histtype = 'stepfilled', normed = False, label = 'outliers RANSAC')
		ax[0].axvline(compl_adu, color = 'k', ls = '--', label = 'Limit count %10.0f' % compl_adu)
		ax[0].axvline(compl_adu_sl, color = 'k', ls = '-', label = 'Limit count only starlike %10.0f' % compl_adu_sl)
		ax[0].set_xscale("log")
		ax[0].legend(loc = 'upper right', fontsize='xx-small')
		ax[0].set_xlabel('ADU')
		ax[0].set_ylabel('N')

		ax[1].scatter(np.log10(repeat_stack['FLUX_AUTO']), np.log10(repeat_epoch['FLUX_AUTO']), marker = '.', color = 'b')
		#ax[1].scatter(np.log10(repeat_stack[ADU_outlier,5]), np.log10(repeat_epoch[ADU_outlier,5]), marker = '.', color = 'r')
		ax[1].plot(ADUx, ADU_y, color = 'r', label = 'RANSAC %.5f, %.5f' % (ADU_slope, ADU_inter))
		ax[1].legend(loc = 'lower right', fontsize='x-small')
		ax[1].set_xlabel('stack (log ADU)')
		ax[1].set_ylabel('single epoch (log ADU)')
		ax[1].grid(which='major', axis='x', linewidth=0.5, linestyle='-', color='0.8')
		ax[1].grid(which='major', axis='y', linewidth=0.5, linestyle='-', color='0.8')

		plt.savefig('%s/%s/%s/%s_2x2.png' % (webpath, field, epoch, name), dpi = 300)
		plt.clf()
		plt.close()

    ############################################ ROC curve ############################################

	if False:

		print 'Ploting ROC curve'
		fig, ax = plt.subplots(2,2, figsize = (15,12))
		fig.suptitle('ROC Curve for %s_%s_thresh%s_area%s' % (field, epoch, str(thresh), str(minarea)), fontsize = 20)
		## ROC curve
		ax[0,0].plot(FPR, TPR, '-k', label = 'SE detections')
		ax[0,0].plot(FPR_limit, TPR_limit, '.r', label = 'Values at limit ADU %.0f' % (compl_adu_sl))
		ax[0,0].set_xlabel('FPR')
		ax[0,0].set_ylabel('TPR')
		ax[0,0].legend(loc= 'lower right', fontsize = 'x-small')
		## FPR/TPR

		ax[0,1].plot(ADU_arange, FPR[:]/TPR[:], '-k', label = 'SE detections')
		ax[0,1].axvline(compl_adu_sl, color = 'r', ls = '-', label = 'Limit adu %.0f' % (compl_adu_sl))
		ax[0,1].set_xlabel('ADU')
		ax[0,1].set_xscale('log')
		ax[0,1].set_ylabel('FPR/TPR')
		ax[0,1].legend(loc= 'upper left', fontsize = 'x-small')
		## FPR vs ADU
		ax[1,0].plot(ADU_arange, FPR, '-k', label = 'SE detections')
		ax[1,0].axvline(compl_adu_sl, color = 'r', ls = '-', label = 'Limit adu %.0f' % (compl_adu_sl))
		ax[1,0].set_xlabel('ADU')
		ax[1,0].set_xscale('log')
		ax[1,0].set_ylabel('FPR')
		ax[1,0].legend(loc= 'lower left', fontsize = 'x-small')
		## TPR vs ADU
		ax[1,1].plot(ADU_arange, TPR, '-k', label = 'SE detections')
		ax[1,1].axvline(compl_adu_sl, color = 'r', ls = '-', label = 'Limit adu %.0f' % (compl_adu_sl))
		ax[1,1].set_xlabel('ADU')
		ax[1,1].set_xscale('log')
		ax[1,1].set_ylabel('TPR')
		ax[1,1].legend(loc= 'lower left', fontsize = 'x-small')

		for axis in ax.flat :
			axis.grid(which='major', axis='x', linewidth=0.5, linestyle='-', color='0.8')
			axis.grid(which='major', axis='y', linewidth=0.5, linestyle='-', color='0.8')

		plt.savefig('%s/%s/%s/%s_%s_ROC_thresh%s_minarea%s.png' % (webpath, field, epoch, field, epoch, str(thresh),
		 str(minarea)), dpi = 300)
		plt.clf()
		plt.close()

##############################################################################################



############################### Functions ###################################################
########################## Found Completeness ADU ###########################################


def compl_adu_bin(data1, data2, bins, pl_name, plots=False, file=False, params=[None]):
	hist1, bins1 = np.histogram(data1, bins = bins)
	hist2, bins2 = np.histogram(data2, bins = bins, range= (data1.min(),data1.max()))
	ratio = hist1.astype(float)/hist2.astype(float)
	x = np.zeros(len(ratio))
	print ratio

	for k in range(len(x) - 1):
		x[k] = (bins1[k] + bins1[k+1])/2

	## looking limit at ratio > 80 percent
	compl_adu = 0
	for j in range(1,len(ratio)):
		if x[j] > 100 and ratio[j] > limit and ratio[j-1] < 1. and ratio[j-1] < limit and x[j] < 1e4:
			xx = np.array([x[j-1], x[j]])
			yy = np.array([ratio[j-1], ratio[j]])
			f = interp1d(yy, xx)
			compl_adu = f(limit)
			break

	if file:
		ratio_name = '%s/%s/%s_%s_ratio_complet' % (sharepath, field, field, epoch)
		np.save(ratio_name, np.array((x, ratio)))

	## plotin histograms + ratio + limit adus
	if params[0] != None:
		print params
		def adu_to_mag(adu):
			return -2.5*np.log10(adu) + 2.5*np.log10(params[0]) + params[1]

	if plots:
		fig, ax = plt.subplots(2, figsize=(7,5))
		#fig.suptitle('FRatio of ADU distribution %s'% (pl_name), fontsize = 10)
		ax[0].hist(data2, bins = bins, log = True, color = 'k', alpha = 1, \
					histtype = 'step', normed = False, label = 'Stack',lw=2)
		ax[0].hist(data1, bins = bins, log = True, color = 'b', alpha = 1, \
					histtype = 'step', normed = False, label = 'Single epoch',lw=2)
		ax[0].axvline(compl_adu, color = 'g', ls = '--',lw=1)
		ax[0].set_xscale('log')
		ax[0].set_yscale('log')
		ax[0].set_ylabel('N')
		ax[0].legend(loc = 'upper left', fontsize='xx-small')
		ax[0].set_xlim(10,100000)
		ax[0].xaxis.set_visible(False)
		ax[1].plot(adu_to_mag(x), ratio, color = 'k', ls = '-', lw=2)
		ax[1].axvline(adu_to_mag(compl_adu), color = 'g', ls = '--',lw=1)
		ax[1].text(adu_to_mag(compl_adu), 1.75,'%.3f' % adu_to_mag(compl_adu), color='g', rotation='vertical')
		ax[1].set_xscale('linear')
		ax[1].set_yscale('linear')
		ax[1].set_xlabel(r'$m_{g}$')
		ax[1].set_ylabel('Ratio')
		ax[1].set_ylim(0,2)
		ax[1].axhline(limit, color = 'r', lw=1)
		ax[1].text(26.2, 0.82, '80%', color='r')
		ax[1].set_xlim(adu_to_mag(10),adu_to_mag(100000))
		nbins = len(ax[1].get_xticklabels())
		ax[1].yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
		#ax[1].grid(which='major', axis='x', linewidth=0.5, linestyle='-', color='0.8')
		#ax[1].grid(which='major', axis='y', linewidth=0.5, linestyle='-', color='0.8')
		ax[0].invert_xaxis()
		ax[1].invert_xaxis()
		fig.subplots_adjust(hspace=0)
		plt.savefig('%s/%s/%s/%s_hist_adu.pdf' % (webpath, field, epoch, pl_name), \
					dpi = 600, format='pdf', bbox_inches='tight')
		plt.clf()
		plt.close()

	if True:
		np.save('%s/data1.npy' % jorgepath, data1)
		data2.dump('%s/data2.npy' % jorgepath)
		np.save('%s/bins.npy' % jorgepath, bins)
		np.save('%s/compl_adu.npy' % jorgepath, compl_adu)
		np.save('%s/x_mag.npy' % jorgepath, adu_to_mag(x))
		np.save('%s/ratio.npy' % jorgepath, ratio)
		np.save('%s/compl_mag.npy' % jorgepath, adu_to_mag(compl_adu))
		np.save('%s/limit.npy' % jorgepath, limit)

	return compl_adu


def background_level(data, sig, ite, cenfunc):
	maskedarr = sigma_clip(data, sigma=sig, iters=ite, cenfunc=cenfunc)
	back_level = np.median(data[~maskedarr.mask])
	return back_level


if __name__ == '__main__':

	startTime = datetime.now()
	field = 'Blind15A_01'
	FILTER = 'g'
	epoch = '02'

	try:
		optlist, args = getopt.getopt(sys.argv[1:], 'F:b:e:')
	except getopt.GetoptError, err:
		print help
		sys.exit()

	for o, a in optlist:
		if o in ('-F'):
			field = str(a)
		elif o in ('-b'):
 			FILTER = str(a)
		elif o in ('-e'):
 			epoch = str(a)
		else:
			continue

	print 'Field: ', field
	print 'Filter: ', FILTER
	print 'Epoch: ', epoch

	epochs = np.loadtxt('%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, FILTER), dtype = str)
	if epoch in epochs[:,0]:
		print 'Epoch %s is observe in filter %s' % (epoch, FILTER)
	else:
		print 'Wrong epoch/ilter combination'
		sys.exit()

	if FILTER in ['g','u']:
		get_full_info(field, epoch, band=FILTER, plots=False)
	else:
		get_partial_info(field, epoch, band=FILTER, plots=False)

	print 'It took', (datetime.now()-startTime), 'seconds'
	print '_______________________________________________________________________'
	print 'Done!'
