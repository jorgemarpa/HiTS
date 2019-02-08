import sys
import os
import getopt
from datetime import datetime
import numpy as np
from astropy.table import Table
from astropy import units as u

jorgepath = '/Users/jorgetil/Astro/HITS/'
pix_scale = 0.2637

def mag_to_flx(x):
    # return flux density in Jy
    return 3631. * np.power(10, -.4 * x) * 10**6

def magStd_to_flxStd(x, dx):
    flx_p = mag_to_flx(x + dx)
    flx_m = mag_to_flx(x - dx)
    return np.abs((flx_p - flx_m) / 2.)

def sigma_mag_to_flx(x):
    return .4 * np.log(10) * x

def convert_units(field):

    for k in [1,2,3,4,5,6,7,8]:
        file_name = '%s/tables/%s_full_photometry_public_%i.fits' % (jorgepath, field, k)
        print file_name
        if not os.path.exists(file_name):
            print '\tNo table file...'
            sys.exit()
        table = Table.read(file_name, format='fits')
        cols_names = table.colnames
        print 'N cols    : ', len(cols_names)
        print 'cols names: ', cols_names
        obs_bands = [x[0] for x in cols_names if 'N' in x]
        # cols_names.remove('internalID')
        # cols_names.remove('Unnamed: 0')
        # cols_names.remove('X')
        # cols_names.remove('Y')

        table['X'].unit = u.pix
        table['Y'].unit = u.pix

        for band in obs_bands:
            print band,
            # cols_names.remove('%sClassStar' % band)

            table['%sFluxRadius' % band] *= pix_scale
            table['%sFluxRadius' % band].unit = u.arcsec
            table['%sFWHM' % band] *= pix_scale
            table['%sFWHM' % band].unit = u.arcsec
            table['%sKronRadius' % band] *= pix_scale
            table['%sKronRadius' % band].unit = u.arcsec

            phot_type = ['Kron', 'Ap1', 'Ap2', 'Ap3', 'Ap4', 'Ap5']
            for tp in phot_type:
                print '\t', tp

                # table.remove_columns('%sMedian%sFlux' % (band, tp))
                # table.remove_columns('%sMedian%sFluxErr' % (band, tp))
                # table.remove_columns('%sMedian%sFluxStd' % (band, tp))

                table['%sMedian%sFlux' % (band, tp)] = mag_to_flx(table['%sMedian%sMag' % (band, tp)])
                table['%sMedian%sFluxStd' % (band, tp)] = table['%sMedian%sFlux' % (band, tp)] * sigma_mag_to_flx(table['%sMedian%sMagStd' % (band, tp)])
                table['%sMedian%sFluxErr' % (band, tp)] = table['%sMedian%sFlux' % (band, tp)] * sigma_mag_to_flx(table['%sMedian%sMagErr' % (band, tp)])

                table['%sMedian%sMag' % (band, tp)].unit = u.mag
                table['%sMedian%sMagStd' % (band, tp)].unit = u.mag
                table['%sMedian%sMagErr' % (band, tp)].unit = u.mag

                table['%sMedian%sFlux' % (band, tp)].unit = u.Jy
                table['%sMedian%sFluxStd' % (band, tp)].unit = u.Jy
                table['%sMedian%sFluxErr' % (band, tp)].unit = u.Jy

                cols_names.insert(cols_names.index('%sMedian%sMag' % (band, tp)), '%sMedian%sFlux' % (band, tp))
                cols_names.insert(cols_names.index('%sMedian%sMag' % (band, tp)), '%sMedian%sFluxErr' % (band, tp))
                cols_names.insert(cols_names.index('%sMedian%sMag' % (band, tp)), '%sMedian%sFluxStd' % (band, tp))

        print table[:5]
        print table['gMedianKronFlux', 'gMedianKronFluxErr', 'gMedianKronFluxStd',
                    'gMedianKronMag',  'gMedianKronMagErr',  'gMedianKronMagStd'][:5]
        print 'N cols    : ', len(cols_names)
        print 'cols names: ', cols_names
        out_name = '%s/tables/%s_full_photometry_public_unist_%i.fits' % (jorgepath, field, k)
        print out_name
        table[cols_names].write(out_name, format='fits', overwrite=True)

        # if k == 1:
        #     break


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
