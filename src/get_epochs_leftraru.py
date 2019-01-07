import sys
import os
import glob
import re
import numpy as np
import getopt
from astropy.io import fits

help = '''
Get epoch for a given field and filter

    -F : String with field
    -B : String with filter: 'g', 'r', 'i' or 'all'

'''

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'


def run_code(field, band, field_list):
    file_list = np.sort(glob.glob(
        "%s/DATA/%s/N4/%s_N4_*_image_crblaster.fits" %
        (astropath, field, field)), kind='mergesort')
    epochs = []
    MJD = []
    airmass, seeing = [], []
    for cata in file_list:
        hdu = fits.open(cata)
        FILTER = hdu[0].header['FILTER'][0]
        SCALE = float(hdu[0].header['PIXSCAL1'])
        FWHM_p = float(hdu[0].header['FWHM'])
        SEEING = FWHM_p * SCALE
        if band == 'all':
            epochs.append(re.findall(
                r'(\w+\d+\w?)\_(\d\d?)\_(\w\d+?)\_(.*?)\_', cata)[0][3])
            MJD.append(hdu[0].header['MJD-OBS'])
        else:
            if FILTER == band:
                epochs.append(re.findall(
                    r'(\w+\d+\w?)\_(\d\d?)\_(\w\d+?)\_(.*?)\_', cata)[0][3])
                MJD.append(hdu[0].header['MJD-OBS'])
                airmass.append(hdu[0].header['AIRMASS'])
                seeing.append(FWHM_p * SCALE)

    if not os.path.exists("%s/info/%s" % (jorgepath, field)):
        print "Creating field folder"
        os.makedirs("%s/info/%s" % (jorgepath, field))

    file_epochs = '%s/info/%s/%s_epochs_%s.txt' % (
        jorgepath, field, field, band)
    print np.vstack((airmass, seeing)).T
    to_save = np.vstack((epochs, MJD))
    if to_save.shape == (2, 0):
        print 'No epochs for this filter...'
        if field_list:
            return
        # sys.exit()
    header = 'EPOCH\t MJD'
    print to_save.T
    np.savetxt(file_epochs, to_save.T, fmt='%s', delimiter='\t')
    print 'Done!'


if __name__ == '__main__':
    field = ''
    band = ''
    field_list_file = ''
    field_list = False

    # read command line option.
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'h:F:B:f:')
    except getopt.GetoptError as err:
        print help
        sys.exit()

    for o, a in optlist:
        if o in ('-h'):
            print help
            sys.exit()
        elif o in ('-F'):
            field = str(a)
        elif o in ('-B'):
            band = str(a)
        elif o in ('-f'):
            field_list_file = str(a)
            field_list = True
        else:
            continue

    print band
    if field_list:
        list_field = np.loadtxt(field_list_file, comments='#', dtype=str)
        for campo in list_field:
            print campo
            run_code(campo, band, field_list)
    else:
        run_code(field, band, field_list)
