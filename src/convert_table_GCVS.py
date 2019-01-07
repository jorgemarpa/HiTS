import numpy as np
import sys
import glob
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

def run_code(table_file):
    print 'File: %s' % (table_file)

    line = []
    with open(table_file) as input:
        for data in input:
            data = data.strip()
            if data[0] != '#':
                data = data.split('|')
                if len(data) != 15:
                    print len(data)
                    print np.char.strip(data)
                line.append(np.char.strip(data))

    line = np.asarray(line)
    line = line.reshape((len(line), 15))

    mask = (line[:,2] != '')
    print line.shape
    line = line[mask]
    print line.shape

    radec = []
    for j in line[:,2]:
        if j == '': print 'si'
        coord = '%s %s %s %s %s %s' % (j[:2], j[2:4], j[4:8], j[8:11], j[11:13], j[13:15])
        radec.append(coord)
        #print '______________________'

    radec = SkyCoord(radec, unit=(u.hourangle, u.deg))
    print radec
    print line.shape

    ra = radec.ra.degree
    dec = radec.dec.degree

    radec_degree = np.column_stack((ra,dec))
    table = np.hstack((line, radec_degree))

    cols = ['NNo', 'GCVS', 'J2000.0', 'Type', 'Max', 'Min I', 'Min II', 'Filter', 'Epoch', 'Year', 'Period', 'M-m', 'Spectrum', 'References', 'Other design', 'RA', 'DEC']
    df = pd.DataFrame(table, columns = cols)
    for k in range(len(df)):
        period = df['Period'][k].strip()
        print '__%s__' % period
        print type(period)
        df['Period'][k] = float(period)

    out = table_file.replace('.dat', '.csv')
    df.to_csv(out)

if __name__ == '__main__':
    table_file = sys.argv[1]

    run_code(table_file)
