#!/usr/bin/env python

import catbuilder as cb
import argparse
import pp
import os
import shutil
import sys
import glob
import re

import pyslurm

import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii

import astromatic_wrapper as aw

import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'

parser = argparse.ArgumentParser()
parser.add_argument('-F', '--field', help="HiTS field", required=True)
parser.add_argument('-C', '--ccd', help="HiTS ccd", required=False)
parser.add_argument('-e', '--epoch', help="HiTS epoch", required=True)
parser.add_argument('-m', '--mosaic', help="do mosaic", required=False,
                    default=False)
args = parser.parse_args()
print args

if not args.mosaic:
    print 'Runing SCAMP by CCD..'
    print 'Extracting catalog...'
    file_path = "%s/DATA/%s/%s/%s_%s_%s_image_crblaster.fits" % \
                (astropath, args.field, args.ccd, args.field,
                 args.ccd, args.epoch)
    name = os.path.basename(file_path.replace('.fits', ''))
    image = cb.Image(file_path)

    GAIN = image.getHeader(key='GAINA')
    RON = image.getHeader(key='RDNOISEA')
    SAT = image.getHeader(key='SATURATA')
    SCALE = image.getHeader(key='PIXSCAL1')
    FWHM_p = image.getHeader(key='FWHM')
    SEEING = FWHM_p * SCALE
    FILTER = image.getHeader()['FILTER'][0]

    print GAIN, RON, SAT, SCALE

    sex_config = {"DETECT_MINAREA": 3,
                  "WEIGHT_GAIN": "Y",
                  "PHOT_AUTOPARAMS": "2.5,3.5",
                  "DETECT_THRESH": 1.0,
                  "ANALYSIS_THRESH": 1.5,
                  "BACK_SIZE": 64,
                  "SATUR_LEVEL": SAT,
                  "GAIN": GAIN,
                  "SEEING_FWHM": SEEING,
                  "PIXEL_SCALE": SCALE}
    cata = image.getCatalog(type="FITS_LDAC", config=sex_config,
                            preserve_workdir=False)

    print len(cata['ldac_catalog'])

    print 'Runing SCAMP...'
    s = cb.scamp([cata['ldac_catalog']], params={"ASTREF_CATALOG": "GAIA-DR1",
                                                 "SAVE_REFCATALOG": "Y",
                                                 "FULLOUTCAT_NAME":
                                                 "full_catalog.cat",
                                                 "MERGEDOUTCAT_TYPE":
                                                 "FITS_LDAC"},
                 preserve_workdir=False)
    p = s.run().get()
    calibrated = p['data']['full_catalog_1.cat']
    print 'saving catalog...'
    if isinstance(calibrated, Table):
        out = '%s/catalogues/%s/%s/%s_scamp.dat' % (jorgepath, args.field,
                                                    args.ccd, name)
        ascii.write(calibrated, out)

    head = p["heads"][0]
    print 'saving astronmetry...'
    head.totextfile('%s/info/%s/%s/scamp_astrometry_%s_%s_%s.dat' %
                    (jorgepath, args.field, args.ccd,
                     args.field, args.ccd, args.epoch), overwrite=True)

    print 'Done!'

else:
    print 'Runing SCAMP as mosaic...'
    file_image = np.sort(glob.glob("%s/DATA/%s/*/%s_*_%s_image_crblaster.fits*"
                                   % (astropath, args.field, args.field,
                                      args.epoch)), kind='mergesort')
    print file_image
    mosaic_cata = []
    for ima in file_image[:10]:
        image = cb.Image(ima)

        GAIN = image.getHeader()['GAINA']
        RON = image.getHeader()['RDNOISEA']
        SAT = image.getHeader()['SATURATA']
        SCALE = image.getHeader()['PIXSCAL1']
        FWHM_p = image.getHeader()['FWHM']
        SEEING = FWHM_p * SCALE
        FILTER = image.getHeader()['FILTER'][0]

        sex_config = {"DETECT_MINAREA": 1,
                      "WEIGHT_GAIN": "Y",
                      "PHOT_AUTOPARAMS": "2.5,3.5",
                      "DETECT_THRESH": 1.0,
                      "ANALYSIS_THRESH": 1.5,
                      "BACK_SIZE": 64,
                      "SATUR_LEVEL": SAT,
                      "GAIN": GAIN,
                      "SEEING_FWHM": SEEING,
                      "PIXEL_SCALE": SCALE}
        cata = image.getCatalog(type="FITS_LDAC", config=sex_config)
        print cata.info()
        print cata[2].header
        sys.exit()
