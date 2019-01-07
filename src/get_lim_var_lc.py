#!/usr/bin/python

import sys
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

jorgepath = '/home/jmartinez/HiTS'


def give_me_value(field, ccd, epoch):
    data = pd.read_csv('%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh1.0_minarea3_backsize64_final-scamp.dat'
                       % (jorgepath, field, ccd, field, ccd, epoch),
                       delimiter='\t')
    print '\t\t', data.shape
    err = data[(data.FLAGS == 0) &
               (data.CLASS_STAR > .9)].MAGERR_AUTO_ZP.median()
    print '\t\t', err
    return err

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--field', help="Field",
                        required=True, default='Blind15A_01', type=str)
    args = parser.parse_args()

    # fields = np.loadtxt('%s/info/fields_%s.txt' % (jorgepath, 'Blind15A'),
    #                     dtype=str)
    ccds = np.loadtxt('%s/info/ccds.txt' % (jorgepath), dtype=str)

    # for j, field in enumerate(fields):
    if True:
        field = args.field
        print field
        epochs = np.loadtxt('%s/info/%s/%s_epochs_%s.txt' %
                            (jorgepath, field, field, 'g'),
                            dtype={'names': ('EPOCH', 'MJD'),
                                   'formats': ('S2', 'f4')}, comments='#')
        for k, ccd in enumerate(ccds):
            print '\t', ccd
            lc_std = []
            for l, epoch in enumerate(epochs):
                try:
                    std = give_me_value(field, ccd, epoch[0])
                except:
                    continue
                lc_std.append([epoch[0], epoch[1], std])

            lc_std = np.array(lc_std)
            print lc_std.shape
            print lc_std
            np.save('/home/jmartinez/HiTS/info/lc_stats/%s_%s_std_lc.npy' %
                    (field, ccd), lc_std)
            # break
        # break

    print 'Done!'
