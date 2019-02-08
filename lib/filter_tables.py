import os
import glob
import sys
import numpy as np
import pandas as pd

def filter_table(file_name):
    if not os.path.exists(file_name):
        print 'No table file !!'
    table = pd.read_csv(file_name)

    print 'Number of rows: %i' % (table.shape[0])

    filter_table = table.query('FLUX_RADIUS <= 10 and FLAGS <= 10 and FWHM <= 10\
                               ELLIPTICITY <= 0.5 and Mean > 15 and Median_g > 15')
    print 'Number of rows: %i' % (filter_table.shape[0])
    out_name = file_name.replace('.csv', '_filter.csv')
    filter_table.to_csv(out_name, compression = 'gzip')

if __name__ == '__main__':
    file_name = sys.argv[1]
    filter_table(file_name)
