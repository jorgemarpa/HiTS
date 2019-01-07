import numpy as np
from astropy.io import fits
import sys
import os
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from sklearn.neighbors import KDTree
import pandas as pd
import getopt
import glob

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'

mac_path = '/Users/jorgetil/Astro/HITS'


def split_PS1(HiTS_year, center_file=False):

    if not center_file:
        FIELDS = np.loadtxt('%s/info/fields_%s.txt' % (jorgepath, HiTS_year), comments = '#', dtype = str)
        CCDS = np.loadtxt('%s/info/ccds.txt' % (jorgepath), comments = '#', dtype = str)

        print FIELDS

        #radec_cats = []
        radec_fits = []
        for field in FIELDS:
            print '\r', field
            file_name = '%s/data/DATA/%s/%s/%s_%s_02_image_crblaster.fits' % (astropath,
                                                field, CCDS[0], field, CCDS[0])
            hdu = fits.open(file_name)
            radec = SkyCoord(ra=hdu[0].header['RA'], dec=hdu[0].header['DEC'],
                               unit=(u.hour, u.deg), frame='icrs')
            radec_fits.append([radec.ra.degree, radec.dec.degree])
            print radec.ra.degree, radec.dec.degree

            #HiTS_path = '%s/tables/%s/%s_HiTS_10_table.csv' % (jorgepath, field, field)
            #HiTS_data = pd.read_csv(HiTS_path)
            #radec_cats.append([HiTS_data.raMedian.mean(),HiTS_data.decMedian.mean()])
            #print HiTS_data.raMedian.mean(), HiTS_data.decMedian.mean()

            print '_____________________________________________________'

        radec_fits_a = np.array(radec_fits)
        to_tile = np.array([FIELDS, radec_fits_a[:,0], radec_fits_a[:,1]])
        np.savetxt('%s/info/%s_center_coord.txt' % (jorgepath, HiTS_year), to_tile.T, delimiter='\t', fmt='%s')

    else:
        center_file_path = '%s/INFO/%s_center_coord.txt' % (mac_path, HiTS_year)
        centers = np.loadtxt(center_file_path, dtype=str)
        FIELDS = list(centers[:,0])
        radec_fits = np.array(centers[:,1:], dtype=float)

    print FIELDS
    print radec_fits

    PS1_path = '%s/PanSTARRS/PS1_x_HiTS.fits' % (mac_path)
    PS1_hdu = fits.open(PS1_path)
    cols = PS1_hdu[1].columns.names
    PS1_table = Table(PS1_hdu[1].data, names=cols)

    tree = KDTree(np.array([PS1_table['raMean'], PS1_table['decMean']]).T)

    for k in range(len(radec_fits)):
        print '%s' % FIELDS[k], radec_fits[k][0], radec_fits[k][1], '|',

        idx_query = tree.query_radius([radec_fits[k][0], radec_fits[k][1]], 1.2 )[0]

        print PS1_table['raMean'][idx_query].mean(), PS1_table['decMean'][idx_query].mean()

        out_path = '%s/PanSTARRS/VOT/%s_PS1.vot' % (mac_path, FIELDS[k])
        PS1_table[idx_query].write(out_path, table_id=FIELDS[k], format='votable', overwrite=True)

    print 'split is Done!'


def concatenate_PS1():

    parts = np.sort(glob.glob("%s/PS1_x_HiTS15A_*_jorgemarpa.fit" % (jorgepath)), kind='mergesort')
    print parts

    frames = []
    for part in parts:
        print part
        hdu_aux = fits.open(part)
        col_names = hdu_aux[1].columns.names
        tab = Table(hdu_aux[1].data)
        frames.append(tab)

    print frames
    frames = vstack(frames)

    frames.write('%s/data/SHARED/PanSTARRS/HiTS_PS1.vot' % (astropath), format='votable', overwrite=True)

    print 'Concatenation Done!'


if __name__ == '__main__':

    mode = 'concat'
    year = 'Blind14A'

    if len(sys.argv) == 1:
        print 'option mode needed...'
        sys.exit()

    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'M:Y:')
    except getopt.GetoptError, err:
        print help
        sys.exit()

    for o, a in optlist:
        if o in ('-M'):
            mode = str(a)
        elif o in ('-Y'):
            year = str(a)
        else:
            continue

    print mode, year

    if mode == 'concat':
        print 'Concatenating PS1 files'
        concatenate_PS1()

    elif mode == 'split':
        print 'Spliting PS1 catalog into HiTS %s fields' % (year)
        split_PS1(year, center_file=True)
