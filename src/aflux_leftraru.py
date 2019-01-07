import sys
import os
import glob
import re
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from misc_func_leftraru import *

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
thresh = 1.0      # threshold of catalog
minarea = 1      # min area for detection

def calculate_aflux(field, CCD, ref):

    if not os.path.exists("%s/info/%s/%s" % (jorgepath, field, CCD)):
        try:
            print "Creating CCD folder"
            os.makedirs("%s/info/%s/%s" % (jorgepath, field, CCD))
        except OSError, e:
            if e.errno != 17:
                raise
            pass

    if not os.path.exists("%s/%s" % (webpath, field)):
        try:
            print "Creating field folder"
            os.makedirs("%s/%s" % (webpath, field))
        except OSError, e:
            if e.errno != 17:
                raise
            pass
    if not os.path.exists("%s/%s/%s" % (webpath, field, CCD)):
        try:
            print "Creating CCD folder"
            os.makedirs("%s/%s/%s" % (webpath, field, CCD))
        except OSError, e:
            if e.errno != 17:
                raise
            pass

    ## Getting epochs for the field
    epochs = np.loadtxt('%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field, field, 'g'), dtype={'names': ('EPOCH', 'MJD'), 'formats': ('S2', 'f4')}, comments='#')

    ## Loading all catalogues
    print 'Loading catalogues'
    all_cata, aflux_old = [], []
    for epo in epochs['EPOCH']:
        print 'Epoch: ', epo
        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_cat.dat" % (jorgepath, field, CCD, field, CCD, epo, str(thresh), str(minarea))
        if not os.path.exists(cata_file):
            print 'No catalog file: %s' % (cata_file)
            continue
        cata = np.loadtxt(cata_file, comments='#')
        #print cata[:,13]
        #cata = cata[(cata[:,5] > 1000) & (cata[:,5] < 10e6) & (cata[:,13] < 10) & (cata[:,9] < np.median(cata[:,9])+1.) & (cata[:,1] > 200) & (cata[:,1] < 2048-200) & (cata[:,2] > 200) & (cata[:,2] < 4096-200)]
        cata = cata[(cata[:,5] > 1000) & (cata[:,5] < 10e6)  & (cata[:,1] > 200) & (cata[:,1] < 2048-200) & (cata[:,2] > 200) & (cata[:,2] < 4096-200)]

        if epo != '02':
            mtch_file = '%s/info/%s/%s/match_%s_%s_%s-02.npy' % (jorgepath, field, CCD, field, CCD, epo)
            if not os.path.exists(mtch_file):
                print 'No match file: %s' % (mtch_file)
                continue
            match_coef = np.load(mtch_file)
            aflux_old.append(match_coef[0])
            print 'Applying transformation to pix coordinates'

            for k in range(len(cata)):
                cata[k,1], cata[k,2] = applyinversetransformation(match_coef[3], cata[k,1], cata[k,2], match_coef[4:])

        all_cata.append(cata)
        if epo == ref:
            ref_cata = cata

    print '_________________________________________________________'
    print 'Total of epochs: %i' % (len(all_cata))

    ## Creating tree for ref epoch
    ref_XY = np.transpose(np.array((ref_cata[:,1],ref_cata[:,2])))
    tree_XY = cKDTree(ref_XY)

    ## Calculating aflux with reference epoch
    aflux_new, aflux_epoch, zp = [], [], []
    for epo, cata in zip(epochs['EPOCH'], all_cata):
        if epo == ref:
            continue
        print 'Compearing epoch %s with ref' % (epo)
        XY_cata = np.transpose(np.array((cata[:,1], cata[:,2])))
        print 'Total of object in epoch %i' % (len(cata))

        print 'Crossmatching with reference epoch'
        superpos_ind = tree_XY.query(XY_cata, k = 1, distance_upper_bound=4)[1]
        index_filter = (superpos_ind < len(ref_cata)) # dice los obj de single epoch encontrados en stack
        index = superpos_ind[index_filter]
        ref_match = ref_cata[index] 			# objetos encontados en ref epoch
        epoch_match = cata[index_filter] 		# objetos encontados en single epoch

        print 'Total of matched objects: %i, %i' % (len(ref_match), len(epoch_match))

        print 'Fitting linear relation with RANSAC'
        name = 'aflux_%s_%s_%s-%s' % (field, CCD, epo, ref)
        slope, inter, X_fit, Y_fit, outlier, rms = linear_reg_RANSAC(ref_match[:,5], epoch_match[:,5], ref_match[:,6], epoch_match[:,6], main_path='%s/%s/%s' % (webpath, field, CCD), pl_name = name, plots = True, log = True)
        #print 'aflux_old = %f, rms = %f' % (match_coef[0], match_coef[1])
        print 'aflux_new = %f, rms = %f' % (slope, rms)
        aflux_new.append(slope)
        aflux_epoch.append(int(epo))
        zp.append(inter)
        aflux_info = np.array((slope, rms, inter),
    		dtype = {'names':['aflux', 'rms', 'inter'],
    		'formats':['float', 'float', 'float']})
        np.save('%s/info/%s/%s/%s' % (jorgepath, field, CCD, name), aflux_info)
        print '_________________________________________________________'


    name_evo = 'a_flux_evolution_ref%s' % (ref)
    fig, ax = plt.subplots(1)
    fig.suptitle(name_evo, fontsize = 13)
    if ref == '02':
        ax.plot(aflux_epoch, aflux_old, 'bo-', alpha = 0.5, label = 'Old')
    ax.plot(aflux_epoch, aflux_new, 'g*-', alpha = 0.5, label = 'New')
    ax.legend(loc = 'upper right', fontsize='xx-small')
    ax.set_xlabel('Time index')
    ax.set_ylabel('aflux')
    ax.set_xlim(0, np.max(aflux_epoch)+1)

    ax2 = ax.twinx()
    ax2.plot(aflux_epoch, zp, 'r.-', label = 'Zero Point')
    ax2.legend(loc = 'lower right', fontsize='x-small')
    ax2.set_ylabel('Zero Point')

    plt.savefig('%s/%s/%s/%s.png' % (webpath, field, CCD, name_evo), dpi = 300)


if __name__ == '__main__':
    field = sys.argv[1]
    CCD = sys.argv[2]
    ref = sys.argv[3]
    calculate_aflux(field, CCD, ref)
    print 'Done!'
