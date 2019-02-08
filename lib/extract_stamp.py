#!/usr/bin/python

import sys
import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.table import Table, vstack
import seaborn as sb

warnings.filterwarnings("ignore")
sb.set(style="white", color_codes=True, context="notebook", font_scale=1.2)

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'

thresh = 1.0      # threshold of catalog
minarea = 3       # min area for detection
deg2rad = 0.0174532925
rad2deg = 57.2957795
axisY = 4094
axisX = 2046


def get_stamps_lc(field, CCD, FILTER, epoch, row_pix, col_pix,
                  dx=100, verbose=False):

    name = '%s_%s_%s_%s_%s_%s' % (field, CCD, col_pix, row_pix, FILTER, epoch)
    print 'Saving stamps for... ', name

        # fits
    imag_file = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % (astropath,
                                                                 field,
                                                                 CCD,
                                                                 field,
                                                                 CCD,
                                                                 epoch)
    if not os.path.exists(imag_file):
        if verbose:
            print '\t\tNo image file: %s' % (imag_file)
        return
    hdu = fits.open(imag_file)
    data = hdu[0].data

    # # loading catalogs
    # cat_file = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_final-scamp.dat" % \
    #     (jorgepath, field, CCD, field, CCD,
    #      epoch[0], str(thresh), str(minarea))
    # if not os.path.exists(cat_file):
    #     if verbose:
    #         print '\t\tNo catalog file for worst epoch: %s' % (cat_file)
    #     continue
    # cata = Table.read(cat_file, format='ascii')
    # cata_XY = np.transpose(np.array((cata['X_IMAGE_REF'],
    #                                  cata['Y_IMAGE_REF'])))
    # tree_XY = cKDTree(cata_XY)
    #
    # # quering asked position
    # XY_obj = np.transpose(np.array((col_pix, row_pix)))
    # dist, indx = tree_XY.query(XY_obj, k=1, distance_upper_bound=5)
    # if np.isinf(dist):
    #     if verbose:
    #         print '\t\tNo match in epoch %s' % epoch[0]
    #         continue

        # position in non projected coordinates, i.e. loaded image
    stamp = data[row_pix - dx: row_pix + dx,
                 col_pix - dx: col_pix + dx]

    plt.figure(figsize=(10,10))
    plt.imshow(stamp, interpolation="nearest", cmap='gray', origin='lower')
    plt.title(name)
    plt.axes.get_xaxis().set_visible(False)
    plt.axes.get_yaxis().set_visible(False)
    plt.savefig('%s/lightcurves/findingcharts/%s.pdf' % (jorgepath, name),
                tight_layout=True, pad_inches=0.01,
                bbox_inches='tight')
    plt.close()

    print 'Done!'
    return



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help="mode: xypix/radec/id/list",
                        required=False, default='xypix', type=str)
    parser.add_argument('-F', '--field', help="HiTS field",
                        required=False, default='Blind15A_01', type=str)
    parser.add_argument('-C', '--ccd', help="HiTS ccd",
                        required=False, default='N1', type=str)
    parser.add_argument('-x', '--xcoord', help="x coordinate",
                        required=False, default=0, type=float)
    parser.add_argument('-y', '--ycoord', help="y coordinate",
                        required=False, default=0, type=float)
    parser.add_argument('-b', '--band', help="filter band",
                        required=False, default='g', type=str)
    parser.add_argument('-e', '--epoch', help="epoch index",
                        required=False, default='02', type=str)
    parser.add_argument('-i', '--id',
                        help="object id or file name with list of ids",
                        required=False, default='', type=str)
    parser.add_argument('-s', '--dx', help="stamp half-size",
                        required=False, default=100, type=int)
    args = parser.parse_args()
    band = args.band
    epoch = args.epoch
    dx = args.dx

    if args.mode in 'xypix':
        field = args.field
        ccd = args.ccd
        row_pix = int(args.ycoord)
        col_pix = int(args.xcoord)
        print field, ccd, row_pix, col_pix

        get_stamps_lc(field, ccd, band, epoch, row_pix, col_pix,
                      dx=dx, verbose=True)


    if args.mode in 'radec':
        field = args.field
        ccd = args.ccd
        row_pix = args.ycoord
        col_pix = args.xcoord
        print field, ccd, row_pix, col_pix

        get_stamps_lc(field, ccd, band, epoch, row_pix, col_pix,
                      dx=dx, verbose=True)


    if args.mode == 'id':
        print args.id
        field, ccd, col_pix, row_pix = re.findall(
            r'(\w+\d+\w?\_\d\d?)\_(\w\d+?)\_(\d+)\_(\d+)', args.id)[0]
        row_pix = int(row_pix)
        col_pix = int(col_pix)
        print field, ccd, row_pix, col_pix

        get_stamps_lc(field, ccd, band, epoch, row_pix, col_pix,
                      dx=dx, verbose=True)


    if args.mode == 'list':

        print 'File: ', args.id
        ID_table = pd.read_csv(args.id, compression='gzip')
        ID_table.set_index('internalID', inplace=True)
        print '# of sources: ', ID_table.shape[0]
        IDs = ID_table.index.values
        fail = []
        for kk, ids in enumerate(IDs):
            field, ccd, col_pix, row_pix = re.findall(
                r'(\w+\d+\w?\_\d\d?)\_(\w\d+?)\_(\d+)\_(\d+)', ids)[0]
            row_pix = int(row_pix)
            col_pix = int(col_pix)
            print kk, field, ccd, col_pix, row_pix

            if (col_pix < dx) or (row_pix < dx) or \
               (axisX - col_pix < dx) or (axisY - row_pix < dx):
                continue
            if ccd == 'S7':
                continue

            try:
                get_stamps_lc(field, ccd, band, epoch,
                              row_pix, col_pix, dx=dx, verbose=True)
            except:
                fail.append(ids)
                print '_____________________'
                continue
            print '_____________________'
            # break
        print 'Fail: ', fail
        thefile = open('/home/jmartinez/HiTS/HiTS-Leftraru/temp/%s_fail.txt'
                       % (field), 'w')
        thefile.writelines( "%s\n" % item for item in fail)
        thefile.close()
        print 'Done!'
