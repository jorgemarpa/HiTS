#!/usr/bin/python

import sys
import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from scipy.spatial import cKDTree
from astropy.io import fits
from matplotlib.colors import LogNorm
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
                  dx=100, title='', verbose=False):

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
    scale = hdu[0].header['PIXSCAL1']

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

    if dx > row_pix or dx > col_pix:
        dx = np.min([dx, row_pix, col_pix])
    if dx > axisY - row_pix or dx > axisX - col_pix:
        dx = np.min([dx, axisY - row_pix, axisX - col_pix])

    stamp = data[row_pix - dx: row_pix + dx,
                 col_pix - dx: col_pix + dx]
    v_min = np.percentile(stamp.ravel(), 30)
    v_max = stamp[dx, dx]
    fontprops = fm.FontProperties(size=14, family='monospace')
    fig, ax = plt.subplots(figsize=(10,10))
    plt.title('%s | %s' % (title, name))
    ax.imshow(stamp, interpolation="nearest", cmap='gray', origin='lower',
              norm=LogNorm(vmin=v_min, vmax=v_max))
    scalebar = AnchoredSizeBar(ax.transData, 222,
                               r'     %.0f$^{\prime}$'% ((222.2 * scale)/60.)+
                               '\n'+
                               r'[%.2f$^{\prime\prime}$/pix]' % (scale),
                               loc=8, color='white',
                               frameon=False, sep=5,
                               size_vertical=1, fontproperties=fontprops)
    ax.add_artist(scalebar)
    ax.set_xlabel('pix')
    ax.set_ylabel('pix')
    # plt.show()
    plt.savefig('%s/figures/findingcharts/%s.pdf' % (jorgepath, name))
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
    parser.add_argument('-t', '--title', help="figure title",
                        required=False, default='', type=str)
    args = parser.parse_args()
    band = args.band
    epoch = args.epoch
    dx = args.dx
    title = args.title

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
                      dx=dx, title=title, verbose=True)


    if args.mode == 'list':

        print 'File: ', args.id
        ID_table = pd.read_csv(args.id)
        ID_table.set_index('internalID', inplace=True)
        print '# of sources: ', ID_table.shape[0]
        IDs = ID_table.index.values
        fail = []
        for kk, ids in enumerate(IDs):
            field, ccd, col_pix, row_pix = re.findall(
                r'(\w+\d+\w?\_\d\d?)\_(\w\d+?)\_(\d+)\_(\d+)', ids)[0]
            row_pix = int(row_pix)
            col_pix = int(col_pix)
            title = ID_table.loc[ids, 'SDSS12']
            print kk, field, ccd, col_pix, row_pix, title
            if ccd == 'S7':
                fail.append(ids)
                continue

            try:
                get_stamps_lc(field, ccd, band, epoch, row_pix, col_pix,
                              title=title, dx=dx, verbose=True)
            except:
                fail.append(ids)
                print '_____________________'
                continue
            print '_____________________'
            # break
        if len(fail) > 0:
            print 'Fail: ', fail
            thefile = open('/home/jmartinez/HiTS/HiTS-Leftraru/temp/%s_fail.txt'
                           % (field), 'w')
            thefile.writelines( "%s\n" % item for item in fail)
            thefile.close()
        print 'Done!'
