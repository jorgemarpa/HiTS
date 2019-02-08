import os
import sys
import glob
import numpy as np
import pandas as pd
from astropy.table import Table
import getopt

jorgepath = '/home/jmartinez/HiTS'
webpath = '/home/apps/astro/WEB/TESTING/jmartinez'


def concat_feat_CCD(field, CCDS):

    frames = []
    for ccd in CCDS:
        print '\t%s' % ccd
        file_name = '%s/features/%s/%s_%s_prepro.csv' % (jorgepath,
                                                         field, field, ccd)
        if not os.path.exists(file_name):
            print '\t\tNo feature file...'
            continue
        aux = pd.read_csv(file_name, compression='gzip')
        # print aux.shape
        frames.append(aux)

    frames = pd.concat(frames, axis=0)
    frames.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    print frames.shape
    return frames


def concat_features(FIELDS, CCDS):

    frames = []
    for field in FIELDS:
        print 'Field: %s' % (field)
        file_name = '%s/features/%s' % (jorgepath, field)
        if not os.path.exists(file_name):
            print '\tNo Field file...'
            print '___________________________'
            continue
        aux = concat_feat_CCD(field, CCDS)
        frames.append(aux)
        # if field == 'Blind15A_04': break
        print '___________________________'

    year = FIELDS[0][:8]
    print year
    data_frame = pd.concat(frames, axis=0)
    data_frame.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    print data_frame.head()
    print 'Total of LC: %i' % data_frame.shape[0]
    print 'Total of Features: %i' % data_frame.shape[1]
    data_frame.to_csv('%s/features/%s_prepro_features.csv' % (jorgepath, year),
                      compression='gzip')


def concat_tables_fields(FIELDS):

    frames = []
    year = FIELDS[0].replace('_01', '')

    for field in FIELDS:
        print 'Field: %s' % (field)
        file_name = '%s/tables/%s/%s_phot_n3_table.csv' % (
            jorgepath, field, field)
        if not os.path.exists(file_name):
            print '\tNo table file...'
            continue
        aux = pd.read_csv(file_name, compression='gzip')
        print aux.columns.values
        frames.append(aux)
        print '___________________________'

    data_frame = pd.concat(frames, axis=0)
    data_frame.drop('Unnamed: 0', axis=1, inplace=True)
    print data_frame.columns.values
    num_obs = [x for x in data_frame.columns.values if 'N' in x]
    print num_obs
    for ja in num_obs:
        data_frame[ja].fillna(value=0, inplace=True)
        data_frame[ja] = data_frame[ja].astype(int)
    data_frame.fillna(value=-999, inplace=True)
    print data_frame.columns.values
    print 'Total of sources: %i' % data_frame.shape[0]
    print 'Total of columns: %i' % data_frame.shape[1]
    # data_frame.to_csv('%s/tables/%s_1_phot_table.csv' % (jorgepath, year),
    # compression = 'gzip')
    astrotable = Table.from_pandas(data_frame)
    astrotable.write('%s/tables/%s_phot_n3_table.fits' % (jorgepath, year),
                     overwrite=True)


def concat_tables_ccds(FIELD, CCDs):

    frames = []
    for ccd in CCDs:
        print 'CCD: %s' % (ccd)
        file_name = '%s/catalogues/%s/%s/%s_%s_HiTS_n3_table.csv' % (
            jorgepath, FIELD, ccd, FIELD, ccd)
        print file_name
        if not os.path.exists(file_name):
            print '\tNo table file...'
            continue
        aux = pd.read_csv(file_name)
        frames.append(aux)
        print '___________________________'

    data_frame = pd.concat(frames, axis=0)
    print data_frame.columns
    print 'Total of sources: %i' % data_frame.shape[0]
    print 'Total of sources: %i' % data_frame.drop_duplicates(
        subset=['ID']).shape[0]
    print 'Total of columns: %i' % data_frame.shape[1]
    # sys.exit()

    if not os.path.exists("%s/tables/%s" % (jorgepath, FIELD)):
        print "Creating field folder"
        os.makedirs("%s/tables/%s" % (jorgepath, FIELD))

    data_frame.to_csv('%s/tables/%s/%s_phot_n3_table.csv' %
                      (jorgepath, FIELD, FIELD), compression='gzip')
    # astrotable = Table.from_pandas(data_frame)
    # astrotable.write('%s/tables/%s/%s_phot_n5_table.fits' %
    #                  (jorgepath, FIELD, FIELD), overwrite=True)


def join_table_feat(year):

    table_file = '%s/tables/%s_tables.csv' % (jorgepath, year)
    if not os.path.exists(table_file):
        print '\tNo table file...'
        sys.exit()
    feat_file = '%s/features/%s_prepro_features.csv' % (jorgepath, year)
    if not os.path.exists(feat_file):
        print '\tNo feature file...'
        sys.exit()

    table = pd.read_csv(table_file)
    table.drop('Unnamed: 0', axis=1, inplace=True)
    table.drop_duplicates(subset='ID', inplace=True)
    feat = pd.read_csv(feat_file)
    feat.drop('Unnamed: 0', axis=1, inplace=True)
    feat.drop_duplicates(subset='ID', inplace=True)

    print 'Table shape:\t', table.shape
    # print table.columns
    print 'Features shape:\t', feat.shape
    # print feat.columns

    result = pd.merge(table, feat, on='ID', how='inner')
    result = result[result['Occu_g'] >= 15]
    # print result[:50]
    print 'Result shape:\t', result.shape
    result.to_csv('%s/tables/%s_tables+feat.csv' %
                  (jorgepath, year), compression='gzip')


if __name__ == '__main__':

    data_table = 'phot'
    all_fields = False
    year = 'Blind15A'

    if len(sys.argv) == 1:
        print help
        sys.exit()

    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'T:F:y:')
    except getopt.GetoptError as err:
        print help
        sys.exit()

    for o, a in optlist:
        if o in ('-T'):
            data_table = str(a)
        elif o in ('-F'):
            if a == 'True':
                all_fields = True
            else:
                all_fields = str(a)
        elif o in ('-y'):
            year = str(a)
        else:
            continue

    if all_fields == True:
        FIELDS = np.loadtxt('%s/info/fields_%s.txt' %
                            (jorgepath, year), dtype=str)
    else:
        FIELDS = all_fields

    CCDS = np.loadtxt('%s/info/ccds.txt' % (jorgepath), dtype=str)

    print FIELDS

    if data_table == 'feat':
        print 'Concatening features files...'
        concat_features(FIELDS, CCDS)

    elif data_table == 'phot':
        print 'Concatening tables files...'
        if all_fields == True:
            concat_tables_fields(FIELDS)
        else:
            concat_tables_ccds(FIELDS, CCDS)

    elif data_table == 'table+feat':
        print 'Joining Table and Features...'
        join_table_feat(year)
