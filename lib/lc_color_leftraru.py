help = '''
Create LC files for g and r filter

	-F : String with field
	-C : String with CCD
    -a[zp|af] : method of photometry correction, default is 'af'
	-1 : Strng with filter 1, default is 'g'
	-2 : Strng with filter 2, default is 'r'

'''
import sys
import os
import getopt
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from misc_func_leftraru import *

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
webpath = '/home/apps/astro/data/WEB/TESTING/jmartinez'
thresh = 1.0      # threshold of catalog
minarea = 1      # min area for detection

def colors(field, CCD, correction, filter1, filter2):

    ## Loading files with epochs per filter
    epochs_c1 = np.loadtxt('%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, filter1), dtype={'names': ('EPOCH', 'MJD'), 'formats': ('S2', 'f4')}, comments='#')
    epochs_c2 = np.loadtxt('%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, filter2), dtype={'names': ('EPOCH', 'MJD'), 'formats': ('S2', 'f4')}, comments='#')

    ## check transfromation
    if False:
        ## Loading catalogues per filter
        cata_file_c1 = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_%s.dat" % (jorgepath, field, CCD, field, CCD, epochs_c1['EPOCH'][0], str(thresh), str(minarea), correction)
        if not os.path.exists(cata_file_c1):
            print 'No catalog file: %s' % (cata_file_c1)
            sys.exit()
        cata_c1 = np.loadtxt(cata_file_c1, comments='#')
        cata_c1 = cata_c1[(cata_c1[:,5] > 0)]
        cata_file_c2 = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_%s.dat" % (jorgepath, field, CCD, field, CCD, epochs_c2['EPOCH'][0], str(thresh), str(minarea), correction)
        if not os.path.exists(cata_file_c2):
            print 'No catalog file: %s' % (cata_file_c2)
            sys.exit()
        cata_c2 = np.loadtxt(cata_file_c2, comments='#')
        cata_c2 = cata_c2[(cata_c2[:,5] > 0)]

        print 'Shape of catalogues %s: ' % (filter1), cata_c1.shape
        print 'Shape of catalogues %s: ' % (filter2), cata_c2.shape

        ## crossmatch between 2 epochs to check transformation
        c1_XY = np.transpose(np.array((cata_c1[:,1], cata_c1[:,2])))
        c2_XY = np.transpose(np.array((cata_c2[:,1], cata_c2[:,2])))
        tree_c1 = cKDTree(c1_XY)
        superpos_ind = tree_c1.query(c2_XY, k = 1, distance_upper_bound = 2)[1]
        index_filter = (superpos_ind < len(c1_XY))                                  # objects with corssmatch in c2 r
        index = superpos_ind[index_filter]                                          # objects with corssmatch in c1 g
        print 'Number of objets with corssmatch: %i' % len(index)

        ## Plots os positions after crossmatch
        if False:
            print 'Ploting positions...'
            name = '%s_%s_%s%s-%s%s' % (field, CCD, filter1, epochs_c1['EPOCH'][0], filter2, epochs_c2['EPOCH'][0])
            fig, ax = plt.subplots(1, figsize = (16,9))
            fig.suptitle(name, fontsize = 15)

            ax.scatter(cata_c1[:,2], cata_c1[:,1], marker = '*', c = 'g', alpha = 0.3)
            ax.scatter(cata_c1[index,2], cata_c1[index,1], marker = '*', c = 'g', alpha = 0.5, label = 'g')
            ax.scatter(cata_c2[:,2], cata_c2[:,1], marker = '.', c = 'r', alpha = 0.3)
            ax.scatter(cata_c2[index_filter,2], cata_c2[index_filter,1], marker = '.', c = 'r', alpha = 0.5, label = 'r')
            ax.set_xlabel('y [pix]')
            ax.set_ylabel('x [pix]')
            ax.set_ylim(0, 2048)
            ax.set_xlim(0, 4096)
            ax.legend(loc = 'best', fontsize='xx-small')

            plt.savefig('%s/%s/%s/%s_match_pix.png' % (webpath, field, CCD, name), dpi = 300)

    ## Loading index files
    print 'Loading index files'
    indx_file_c1 = "%s/lightcurves/%s/%s/%s_%s_%s_master_index.txt" % (jorgepath, field, CCD, field, CCD, filter1)
    if not os.path.exists(indx_file_c1):
        print 'No master index file: %s' % (indx_file_c1)
        sys.exit()
    master_c1 = np.loadtxt(indx_file_c1, comments = '#')

    ## cleaning objects with no all the epochs
    mask_all_c1 = []
    for obj in master_c1:
        mask_all_c1.append(all(obj >= 0))
    mask_all_c1 = np.array(mask_all_c1)


    indx_file_c2 = "%s/lightcurves/%s/%s/%s_%s_%s_master_index.txt" % (jorgepath, field, CCD, field, CCD, filter2)
    if not os.path.exists(indx_file_c2):
        print 'No master index file: %s' % (indx_file_c2)
        sys.exit()
    master_c2 = np.loadtxt(indx_file_c2, comments = '#')

    ## cleaning objects with no all the epochs
    mask_all_c2 = []
    for obj in master_c2:
        mask_all_c2.append(all(obj >= 0))
    mask_all_c2 = np.array(mask_all_c2)

    print 'Shape of index matrix (after clean) %s: ' % (filter1), master_c1[mask_all_c1].shape
    print 'Shape of index matrix (after clean) %s: ' % (filter2), master_c2[mask_all_c2].shape

    mag_c1 = np.zeros(master_c1[mask_all_c1].shape)
    mag_err_c1 = np.zeros(master_c1[mask_all_c1].shape)
    mag_c2 = np.zeros(master_c2[mask_all_c2].shape)
    mag_err_c2 = np.zeros(master_c2[mask_all_c2].shape)
    flux_c1 = np.zeros(master_c1[mask_all_c1].shape)
    flux_err_c1 = np.zeros(master_c1[mask_all_c1].shape)
    flux_c2 = np.zeros(master_c2[mask_all_c2].shape)
    flux_err_c2 = np.zeros(master_c2[mask_all_c2].shape)

    pos_c1 = np.zeros((len(master_c1[mask_all_c1]),2))
    pos_c2 = np.zeros((len(master_c2[mask_all_c2]),2))

    ## Saiving mag, err_mag and position of objects with all epochs
    for epo in range(len(epochs_c1)):
        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_%s.dat" % (jorgepath, field, CCD, field, CCD, epochs_c1['EPOCH'][epo], str(thresh), str(minarea), correction)
        if not os.path.exists(cata_file):
            print 'No catalog file: %s' % (cata_file)
            continue
        cata = np.loadtxt(cata_file, comments='#')
        ii = np.array(master_c1[mask_all_c1, epo], dtype = int)
        mag_c1[:,epo] = cata[ii, 7]
        mag_err_c1[:,epo] = cata[ii, 8]
        flux_c1[:,epo] = cata[ii, 5]
        flux_err_c1[:,epo] = cata[ii, 6]
        if epo == 0:
            pos_c1[:,0] = cata[ii, 1]
            pos_c1[:,1] = cata[ii, 2]

    for epo in range(len(epochs_c2)):
        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_%s.dat" % (jorgepath, field, CCD, field, CCD, epochs_c2['EPOCH'][epo], str(thresh), str(minarea), correction)
        if not os.path.exists(cata_file):
            print 'No catalog file: %s' % (cata_file)
            continue
        cata = np.loadtxt(cata_file, comments='#')
        ii = np.array(master_c2[mask_all_c2, epo], dtype = int)
        mag_c2[:,epo] = cata[ii, 7]
        mag_err_c2[:,epo] = cata[ii, 8]
        flux_c2[:,epo] = cata[ii, 5]
        flux_err_c2[:,epo] = cata[ii, 6]
        if epo == 0:
            pos_c2[:,0] = cata[ii, 1]
            pos_c2[:,1] = cata[ii, 2]


    ## crossmatch between two filters
    tree_c1 = cKDTree(pos_c1)
    superpos_ind = tree_c1.query(pos_c2, k = 1, distance_upper_bound = 2)[1]
    index_filter = (superpos_ind < len(pos_c1))                                   # objects with corssmatch in c1 g
    index = superpos_ind[index_filter]                                            # objects with corssmatch in c2 r
    print 'Number of objets with corssmatch: %i' % len(index)

    repeat_c1 = [pos_c1[index], flux_c1[index], flux_err_c1[index], mag_c1[index], mag_err_c1[index]]
    repeat_c2 = [pos_c2[index_filter], flux_c2[index_filter], flux_err_c2[index_filter], mag_c2[index_filter], mag_err_c2[index_filter]]

    if True:
        print 'Ploting positions...'
        name = '%s_%s_%s%s-%s%s' % (field, CCD, filter1, epochs_c1['EPOCH'][0], filter2, epochs_c2['EPOCH'][0])
        fig, ax = plt.subplots(1, figsize = (12,7))
        fig.suptitle(name, fontsize = 15)

        ax.scatter(pos_c1[:,1], pos_c1[:,0], marker = '*', c = 'g', alpha = 0.5, label = 'g')
        ax.scatter(pos_c2[:,1], pos_c2[:,0], marker = '.', c = 'r', alpha = 0.5, label = 'r')
        ax.set_xlabel('y [pix]')
        ax.set_ylabel('x [pix]')
        ax.set_ylim(0, 2048)
        ax.set_xlim(0, 4096)
        ax.legend(loc = 'best', fontsize='xx-small')

        plt.savefig('%s/%s/%s/%s_match_pix.png' % (webpath, field, CCD, name), dpi = 300)

    median_mag_c1 = np.median(repeat_c1[3], axis = 1)           # mag filter1 = g
    median_mag_c2 = np.median(repeat_c2[3], axis = 1)           # mag filter2 = r

    repeat_c1.append(median_mag_c1 - median_mag_c2)
    repeat_c2.append(median_mag_c1 - median_mag_c2)

    ## sort objects by y-coordinate, to have LC, colors and coord in the same order
    idx_sort_c1 = np.argsort(repeat_c1[0][:,1])
    for k in range(len(repeat_c1)):
        repeat_c1[k] = [ repeat_c1[k][i] for i in idx_sort_c1 ]

    ## printing magnitudes and colors
    if False:
        for m1, m2, c in zip(median_mag_c1, median_mag_c2, repeat_c1[-1]):
            print '%s = %f\t%s = %f\t %s-%s = %f' %(filter1, m1, filter2, m2, filter1, filter2, c)

    ## ploting CMD
    if True:
        print 'Ploting Color-Mag diag...'
        name = '%s_%s_%s%s-%s%s' % (field, CCD, filter1, epochs_c1['EPOCH'][0], filter2, epochs_c2['EPOCH'][0])
        fig, ax = plt.subplots(2)
        fig.suptitle(name, fontsize = 15)

        ax[0].hist(repeat_c1[-1], bins = 25, histtype = 'step')
        ax[0].set_xlabel('%s-%s' % (filter1, filter2))
        ax[0].set_ylabel('Frequency')

        ax[1].scatter(repeat_c1[-1], median_mag_c1, marker = 'o', c = 'b', alpha = 0.7)
        ax[1].set_xlabel('%s-%s' % (filter1, filter2))
        ax[1].set_ylabel('%s' % (filter1))

        plt.savefig('%s/%s/%s/%s_CMD.png' % (webpath, field, CCD, name), dpi = 300)

    ## Saving LC and position files
    if filter1 == 'g':
        print 'Saving LC, coordinates and colors files'
        for p_xy, mag, mag_e, flx, flx_e in zip(repeat_c1[0], repeat_c1[3], repeat_c1[4], repeat_c1[1], repeat_c1[2]):
            #print '%04i %04i' % (p_xy[1], p_xy[0])
            np.savetxt('%s/lightcurves/%s/%s/%s_%s_%s_%04i_%04i_%s.dat' % (jorgepath, field, CCD, field, CCD, filter1, p_xy[1], p_xy[0], correction), np.array([mag, mag_e, flx, flx_e, epochs_c1['EPOCH'], epochs_c1['MJD']] , dtype = float).T, fmt = ['%0.4f', '%0.4f', '%10.1f', '%8.1f', '%02i', '%f'], delimiter = '\t')
        np.savetxt('%s/lightcurves/%s/%s/%s_%s_%s_xy.txt' % (jorgepath, field, CCD, field, CCD, filter1), repeat_c1[0], fmt = ['%.4f', '%.4f'], delimiter = '\t')
        np.savetxt('%s/lightcurves/%s/%s/%s_%s_color_%s-%s.txt' % (jorgepath, field, CCD, field, CCD, filter1, filter2), repeat_c1[-1], fmt = ['%.4f'], delimiter = '\t')


if __name__ == '__main__':
    field = ''
    CCD = ''
    filter1 = 'g'
    filter2 = 'r'
    correction = 'af'

    #read command line option.
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'h:F:C:1:2:a:')
    except getopt.GetoptError, err:
        print help
        sys.exit()

    for o, a in optlist:
        if o in ('-h'):
            print help
            sys.exit()
        elif o in ('-F'):
            field = str(a)
        elif o in ('-C'):
            CCD = str(a)
        elif o in ('-a'):
            correction = str(a)
        elif o in ('-1'):
            filter1 = str(a)
        elif o in ('-2'):
            filter2 = str(a)
        else:
            continue

    print correction
    print 'Filter 1:', filter1
    print 'Filter 2:', filter2
    colors(field, CCD, correction, filter1, filter2)
    print 'Done!'
