#!/usr/bin/python

# busqueda de objetos en 2 catalagos usando cKDTree
# match y matriz final (table) con los indices.
# transformacion de orden superior
# problema de archivos faltantes resuelto

import sys
import os
import getopt
from datetime import datetime
import numpy as np
from scipy.spatial import cKDTree
import warnings
from astropy.table import Table

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'
sharepath = '/home/apps/astro/home/jmartinez'

thresh = 1.0      # threshold of catalog
minarea = 3      # min area for detection
deg2rad = 0.0174532925
rad2deg = 57.2957795

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


def run_crossmatch_lc(field, CCD, FILTER, kind='final',
                      startTime=datetime.now()):

    warnings.filterwarnings("ignore")

    ##########################################################################

    if not os.path.exists("%s/lightcurves/" % (jorgepath)):
        print "Creating lightcurve folder"
        os.makedirs("%s/lightcurves/" % (jorgepath))
    if not os.path.exists("%s/lightcurves/%s" % (jorgepath, field)):
        print "Creating field folder"
        os.makedirs("%s/lightcurves/%s" % (jorgepath, field))
    if not os.path.exists("%s/lightcurves/%s/%s" % (jorgepath, field, CCD)):
        print "Creating CCD folder"
        os.makedirs("%s/lightcurves/%s/%s" % (jorgepath, field, CCD))

    ##########################################################################

    epochs_file = '%s/info/%s/%s_epochs_%s.txt' % (jorgepath, field,
                                                   field, FILTER)
    if not os.path.exists(epochs_file):
        print 'No epochs file: %s' % (epochs_file)
        sys.exit()
    epochs = np.loadtxt(epochs_file, comments='#', dtype=str)

    if epochs.shape == (2,):
        epochs = epochs.reshape(1, 2)

    INFO = []
    epoch_c = []
    tree = []
    X_Y = []

    print 'Loading catalogues (%s) files, creating tree structure' % (kind)
    no_epoch = 0
    for epoch in epochs:
        print 'Epoch %s' % epoch[0]

        # catalogues

        cata_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
            (jorgepath, field, CCD, field, CCD,
             epoch[0], str(thresh), str(minarea))
        if not os.path.exists(cata_file):
            print 'No catalog file: %s' % (cata_file)
            no_epoch += 1
            continue
        # cata = np.loadtxt(cata_file, comments='#')
        cata = Table.read(cata_file, format='ascii')

        # epoch_c has all the catalogues, each element of epoch_c contain the
        # catalogue of a given epoch
        epoch_c.append(cata)
        cata_XY = np.transpose(np.array((cata['X_IMAGE_REF'],
                                         cata['Y_IMAGE_REF'])))
        # X_Y has the pix coordinates of each catalogue
        X_Y.append(cata_XY)
        # X_Y has the pix coordinates of each catalogue in tree structure
        tree.append(cKDTree(cata_XY))

        # INFO of epochs

        INFO.append(epoch)

    if len(epoch_c) == 0:
        print 'No catalogues for this CCD'
        sys.exit()

    INFO = np.asarray(INFO)
    print '____________________________________________________________________'

    # compare each all catalogues to find same

    # master has the final index matrix, rows are objects and columns epochs
    # if master_cat[i][j] = -1 then no match for this object i in epoch j
    master_cat = np.ones((1, len(epoch_c)), dtype=np.int) * (-1)

    # comparar todas las epocas entre ellas buscando matching
    for TIME in range(len(epoch_c)):

        print 'Length of catalog %s = %i' % (INFO[TIME, 0], len(X_Y[TIME]))

        aux_cat = np.ones((len(X_Y[TIME]), len(epoch_c)), dtype=np.int) * (-1)
        aux_cat[:, TIME] = np.arange(len(X_Y[TIME]))

        if TIME < len(epoch_c):
            for time in range(TIME + 1, len(epoch_c)):

                print 'comparing epoch %s with epoch %s' % (INFO[TIME, 0],
                                                            INFO[time, 0])
                # find for nn
                aux_dist = tree[time].query(X_Y[TIME], k=1,
                                            distance_upper_bound=5)
                aux_cat[:, time] = aux_dist[1]
                # busca y reemplaza por -1 los que no encontro
                mask_no = np.where(aux_cat[:, time] == len(X_Y[time]))
                aux_cat[mask_no, time] = -1
                print 'max: ', np.max(aux_dist[0][~np.isinf(aux_dist[0])])
                # mask_yes es un arreglo con los indices de los encontrados
                # mask_yes tiene los indices de los objetos encontrados en
                # epoch[time] y tiene largo de len(epoch[TIME])
                mask_yes = aux_cat[np.where(aux_cat[:, time] > 0), time]

                print 'objects with match = %i' % len(mask_yes[0])

        # quitar de aux_cat los ya encontrados en las iteraciones anteriores.
        if TIME > 0:
            to_remove = []
            for q in range(len(aux_cat[:, TIME])):

                repited = np.where(aux_cat[q, TIME] == master_cat[:, TIME])[0]
                if len(repited) > 0:
                    to_remove.append(q)

            aux_cat = np.delete(aux_cat, to_remove, 0)

        # concatenate the aux_catalog to the master catalog
        master_cat = np.vstack((master_cat, aux_cat))
        print 'objects added = %i' % len(aux_cat)
        aux_cat = 0
        print '_______________________________________________________________'

    master_cat = np.delete(master_cat, 0, 0)
    print 'Total of objects = %i' % len(master_cat)

    ##########################################################################

    if kind == 'final':
        np.savetxt("%s/lightcurves/%s/%s/%s_%s_%s_master_index.txt" %
                   (jorgepath, field, CCD, field, CCD, FILTER), master_cat,
                   fmt='%04i', delimiter='\t')
    elif kind == 'temp':
        np.savetxt("%s/catalogues/%s/%s/temp_%s_%s_%s_master_index.txt" %
                   (jorgepath, field, CCD, field, CCD, FILTER), master_cat,
                   fmt='%04i', delimiter='\t')

    # Create lig

    print 'Total of epochs %i' % len(epochs)
    print 'Effective epochs %i' % (len(epochs) - no_epoch)

    print 'It took', (datetime.now() - startTime), 'seconds'
    print '___________________________________________________________________'


if __name__ == '__main__':
    startTime = datetime.now()

    field = ''
    CCD = ''
    FILTER = 'g'

    # read command line option.
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'h:F:C:b:')
    except getopt.GetoptError as err:
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
        elif o in ('-b'):
            FILTER = str(a)
        else:
            continue

    print 'Field: ', field
    print 'CCD: ', CCD
    print 'Filter: ', FILTER
    run_crossmatch_lc(field, CCD, FILTER, startTime=startTime)

    print 'Done!'
