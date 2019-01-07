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

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
sharepath = '/home/apps/astro/home/jmartinez'

thresh = 1.0      # threshold of catalog
minarea = 1      # min area for detection

# 0 = number
# 1,2 = X,Y
# 3,4 = RA,DEC
# 5,6 = FLUX-ERR_AUTO
# 7, 8 = MAG-ERR_AUTO
# 9 = FLUX_RADIUS
# 10 = FWHM
# 11 = CLASS_STAR
# 12 = ELONGATION
# 13 = FLAG
# 14 = A_IMAGE
# 15 = B_IMAGE
# 16 = THETA_IMAGE
# 17 = KRON_RADIUS


def plot_lcs(field, CCD, FILTER, corr):

    # Creting figure folders.
    if not os.path.exists("%s/%s" % (webpath, field)):
        print "Creating field folder"
        os.makedirs("%s/%s" % (webpath, field))
    if not os.path.exists("%s/%s/%s" % (webpath, field, CCD)):
        print "Creating CCD folder"
        os.makedirs("%s/%s/%s" % (webpath, field, CCD))

    # Loading file list
    if corr == 'all':
        file_l_af = np.sort(
            glob.glob(
                '%s/lightcurves/%s/%s/%s_%s_%s_*_af.dat' %
                (jorgepath, field, CCD, field, CCD, FILTER)), kind='mergesort')
        file_l_zp = np.sort(
            glob.glob(
                '%s/lightcurves/%s/%s/%s_%s_%s_*_zp.dat' %
                (jorgepath, field, CCD, field, CCD, FILTER)), kind='mergesort')

        print 'Ploting LC'
        for file_af, file_zp in zip(file_l_af, file_l_zp):

            LC_af = np.loadtxt(file_af)
            LC_zp = np.loadtxt(file_zp)
            name = os.path.basename(file_af).replace('_af.dat', '')
            print 'Ploting LC %s' % name

            fig, ax = plt.subplots(1, figsize=(12, 4))
            fig.suptitle(name, fontsize=15)

            ax.grid()
            ax.errorbar(LC_af[:, 4], LC_af[:, 0], yerr=LC_af[:, 1],
                        fmt='o-', color='b',
                        alpha=0.7, label=r'$a_{flux} \quad \sigma = %.5f$' %
                        np.std(LC_af[:, 0]))
            ax.errorbar(LC_zp[:, 4], LC_zp[:, 0], yerr=LC_zp[:, 1],
                        fmt='o-', color='g',
                        alpha=0.7, label=r'$ZP \quad \sigma = %.5f$' %
                        np.std(LC_zp[:, 0]))
            # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.legend(loc='upper right', fontsize='small')
            ax.set_ylabel('g')
            ax.set_xlabel('Epoch')
            # ax.set_ylim(np.max(LC_zp[:,0])+.5, np.min(LC_zp[:,0])-.5)
            ax.set_xlim(np.min(LC_af[:, 4]) - 1, np.max(LC_af[:, 4]) + 1)

            plt.savefig(
                '%s/%s/%s/%s_LC.png' %
                (webpath, field, CCD, name), dpi=300, bbox_inches='tight')
            plt.close()

    else:
        file_l = np.sort(
            glob.glob(
                '%s/lightcurves/%s/%s/%s_%s_*_%s.dat' %
                (jorgepath, field, CCD, field, CCD, corr)), kind='mergesort')

        print 'Ploting LC'
        for file_ in file_l:

            LC = np.loadtxt(file_)
            name = os.path.basename(file_).replace('_%s.dat' % corr, '')
            print 'Ploting LC %s' % name

            fig, ax = plt.subplots(1, figsize=(12, 4))
            fig.suptitle(name, fontsize=15)

            ax.grid()
            ax.errorbar(LC[:, 4], LC[:, 0], yerr=LC[:, 1],
                        fmt='o-', color='b', alpha=0.7,
                        label=r'$%s \quad \sigma = %.5f$' %
                        (corr.upper(), np.std(LC[:, 0])))
            # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.legend(loc='upper right', fontsize='small')
            ax.set_ylabel('g')
            ax.set_xlabel('Epoch')
            ax.set_xlim(np.min(LC[:, 4]) - 1, np.max(LC[:, 4]) + 1)

            plt.savefig(
                '%s/%s/%s/%s_LC.png' %
                (webpath, field, CCD, name), dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    field = sys.argv[1]
    ccd = sys.argv[2]
    filter = sys.argv[3]
    corr = sys.argv[4]
    plot_lcs(field, ccd, filter, corr)
