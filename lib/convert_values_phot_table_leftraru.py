import sys
import os
import getopt
from datetime import datetime
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'
pix_scale = 0.27

def convert_units(field):

    file_name = '%s/tables/%s/%s_phot_table.csv' % (jorgepath, field, field)
    if not os.path.exists(file_name):
        print '\tNo table file...'
        sys.exit()
    table = pd.read_csv(file_name)
    cols_names = table.columns.tolist()
    obs_bands = [x[0] for x in cols_names if 'N' in x]
    cols_names.remove('internalID')
    cols_names.remove('Unnamed: 0')
    cols_names.remove('X')
    cols_names.remove('Y')

    #print table[cols_names].head(5)

    mag_to_flx = lambda x: 3631. * np.power(10, -.4 * x) * 10**6
    sigma_mag_to_flx = lambda x: .4 * np.log(10) * x

    for band in obs_bands:
        print band,
        cols_names.remove('%sClassStar' % band)

        table['%sFluxRadius' % band] *= pix_scale
        table['%sFWHM' % band] *= pix_scale
        table['%sKronRadius' % band] *= pix_scale

        phot_type = ['Kron', 'Ap1', 'Ap2', 'Ap3', 'Ap4', 'Ap5']
        for tp in phot_type:
            print '\t',tp
            table['%sMedian%sFlux' % (band, tp)]    = mag_to_flx(table['%sMedian%sMag' % (band, tp)].values)
            # table['%sMedian%sFluxStd' % (band, tp)] = mag_to_flx(table['%sMedian%sMagStd' % (band, tp)].values)
            table['%sMedian%sFluxErr' % (band, tp)] = table['%sMedian%sFlux' % (band, tp)] * sigma_mag_to_flx(table['%sMedian%sMagErr' % (band, tp)].values)

            cols_names.remove('%s%sFlux' % (band, tp))
            cols_names.remove('%s%sFluxErr' % (band, tp))

            cols_names.insert(cols_names.index('%sMedian%sMag' % (band, tp)), '%sMedian%sFlux' % (band, tp))
            # cols_names.insert(cols_names.index('%sMedian%sMag' % (band, tp)), '%sMedian%sFluxStd' % (band, tp))
            cols_names.insert(cols_names.index('%sMedian%sMag' % (band, tp)), '%sMedian%sFluxErr' % (band, tp))

    #print cols_names
    #print len(cols_names)
    #print table[['gMedianKronMag', 'gMedianKronMagErr' , 'gMedianKronFlux', 'gMedianKronFluxErr']].head(20)
    out_name = '%s/tables/%s/%s_public_phot_table.csv' % (jorgepath, field, field)
    print out_name
    table[cols_names].to_csv(out_name)


if __name__ == '__main__':
    startTime = datetime.now()
    field = 'Blind15A_01'

    if len(sys.argv) == 1:
        print help
        sys.exit()

    # read command line option.
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'F:')
    except getopt.GetoptError, err:
        print help
        sys.exit()

    for o, a in optlist:
        if o in ('-F'):
            field = str(a)
        else:
            continue

    convert_units(field)
    print 'It took', (datetime.now()-startTime), 'seconds'
    print '___________________________________________________________________'
