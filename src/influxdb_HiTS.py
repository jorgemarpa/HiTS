import os
import re
import sys
import glob
import time
import getopt
import socket
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from astropy.table import Table
from datetime import datetime
from influxdb import InfluxDBClient
from scipy.spatial import cKDTree
from astropy import units as u
import json


# mainpath = '/Users/jorgetil/Astro/HITS'
jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro/data'


def data_ingestion(field, ccd, hostname='localhost', db_name='HiTS-test'):

    HiTS_path = '%s/catalogues/%s/%s/%s_%s_HiTS_10_table.csv' % (jorgepath, field, ccd, field, ccd)
    HiTS_data = pd.read_csv(HiTS_path)
    HiTS_data.drop('Unnamed: 0', axis=1, inplace=True)
    HiTS_data.drop_duplicates(subset=['ID'],inplace=True)
    HiTS_data.set_index('ID', inplace=True)

    print HiTS_data.shape
    # HiTS_data.head(5)

    cata_list = np.sort(glob.glob('%s/catalogues/%s/%s/%s_%s_*_image_crblaster_thresh1.0_minarea1_backsize64_final.dat' %
                         (jorgepath, field, ccd, field, ccd)))
    catalogs = []
    epochs = []
    filter_list = []
    tree_radec = []
    tree_pix = []
    epochs_g = np.loadtxt('%s/info/%s/%s_epochs_g.txt' % (jorgepath, field, field))
    epochs_r = np.loadtxt('%s/info/%s/%s_epochs_r.txt' % (jorgepath, field, field))
    epochs_i = np.loadtxt('%s/info/%s/%s_epochs_i.txt' % (jorgepath, field, field))
    #print epochs_g
    #print epochs_r
    #print epochs_i
    for file in cata_list:
        epo = file.replace('%s/catalogues/%s/%s/%s_%s_' % (jorgepath, field, ccd, field, ccd),'')[:2]
        print '\r', epo,
        band = ''
        if int(epo) in epochs_g[:,0]:
            band = 'g'
            pos = np.where(epochs_g[:,0] == int(epo))[0][0]
            print pos,
            mjd = epochs_g[pos,1]
        elif int(epo) in epochs_r[:,0]:
            band = 'r'
            pos = np.where(epochs_r[:,0] == int(epo))[0][0]
            print pos,
            mjd = epochs_r[pos,1]
        elif int(epo) in epochs_i[:,0]:
            band = 'i'
            pos = np.where(epochs_i[:,0] == int(epo))[0][0]
            print pos,
            mjd = epochs_i[pos,1]
        epochs.append([epo, mjd])
        filter_list.append(band)
        print band

        cata_aux = Table.read(file, format = 'ascii')
        catalogs.append(cata_aux)

        # coord = SkyCoord(ra=cata_aux['RA'].tolist(), dec=cata_aux['DEC'].tolist(),
        #                 frame='icrs', unit=u.degree)
        #tree_radec.append(coord)
        tree_pix.append(cKDTree(np.array([cata_aux['X_IMAGE'].tolist(), cata_aux['Y_IMAGE'].tolist()]).T))


    client = InfluxDBClient(hostname, 8086, 'root', 'root', db_name)
    #client.create_database('HiTS-test')

    master_name = 'sources'
    obs_name = 'obs_%s' % field

    t_0 = time.time()
    sources = []
    time_index_db = []
    delta_obj = 0
    count_obs = 0
    if ccd[0] == 'N':
        seed = int(ccd[1:])
    else:
        seed = int(ccd[1:]) + 30
    master_index = 100000 * (seed - 1)
    print 'initial index ', master_index

    for i,idx in enumerate(HiTS_data.index.tolist()):
        print '\r', i, idx,
        #print int(HiTS_data.loc[idx,'gN'] + HiTS_data.loc[idx,'rN'] + HiTS_data.loc[idx,'iN'])

        X = HiTS_data.loc[idx,'X']
        Y = HiTS_data.loc[idx,'Y']
        internalID = HiTS_data.loc[idx,'internalID']

        obj_serie = []
        times = []
        gN, rN, iN = 0,0,0
        obj_off = 10 * i # offset between object of 10 days

        for k, cata in enumerate(catalogs):
            #query = SkyCoord(ra=ra, dec=dec, frame='icrs', unit=u.degree)
            #pos, d2d, d3d = query.match_to_catalog_3d(tree_radec[k])

            dist, poss = tree_pix[k].query([X,Y], k = 1, distance_upper_bound = 1)
            #print pos, poss
            #print d2d.arcsec, dist

            if poss == len(cata): continue
            count_obs += 1
            master_index += 1

            if filter_list[k] == 'g':
                gN += 1
            elif filter_list[k] == 'r':
                rN += 1
            elif filter_list[k] == 'i':
                iN += 1

            mjd_idx = epochs[k][1] + int(delta_obj + obj_off)
            times.append(mjd_idx)
            time_index_db.append(mjd_idx)

            mjd_idx *= 10**9
            #print int(mjd_idx)
            obj = {"measurement": obs_name,
                    "time"    : int(master_index),
                    "fields"  :{"mjd": epochs[k][1],
                                "ra" : cata['RA'][poss],
                                "dec": cata['DEC'][poss],
                                "KronMag"   : cata['MAG_AUTO_ZP'][poss],
                                "KronMagErr": cata['MAGERR_AUTO_ZP'][poss],
                                "Flag": cata['FLAGS'][poss],
                                "FluxRadius": cata['FLUX_RADIUS'][poss],
                                "FWHM": cata['FWHM_IMAGE'][poss],
                                "Ellipticity": 1 - 1./cata['ELONGATION'][poss],
                                "KronRadius": 1 - 1./cata['KRON_RADIUS'][poss]
                                },
                    "tags"    :{"ID"     : idx,
                                "filter" : filter_list[k],
                                "internalID" : internalID
                               }
                  }
            obj_serie.append(obj)
        client.write_points(obj_serie)

        # source = {"measurement": master_name,
        #           "time"       : i+1,
        #           "fields"     :{"ID" : idx,
        #                         "raMedian" : HiTS_data.loc[idx,'raMedian'],
        #                         "decMedian": HiTS_data.loc[idx,'decMedian'],
        #                         "allN"     : count,
        #                         "gN"       : gN,
        #                         "rN"       : rN,
        #                         "iN"       : iN,
        #                         "gMedianKronMag"   : HiTS_data.loc[idx,'gMedianKronMag'],
        #                         "gMedianKronMagErr": HiTS_data.loc[idx,'gMedianKronMagErr']
        #                         },
        #            "tags"    :{"ID" : idx
        #                       }
        #           }
        # sources.append(source)
        #all_times.append(times)
        delta_obj += np.max(times) - np.min(times)
        #print 'elapsed time: ', np.max(times) - np.min(times)


    # client.write_points(sources)
    print '_______________'
    t_1 = time.time()
    write_time = (t_1-t_0)
    print 'Points in obs table: ', count_obs
    print("Write time: %f" % write_time)



def test_queries():

    query_list = ['SELECT * FROM "sources"',
              'SELECT "ID", "KronMag", "filter" FROM "obs"',
              'SELECT * FROM "sources" WHERE "allN" > 25',
              'SELECT * FROM "obs" WHERE "filter" = \'i\'',
              'SELECT MEDIAN("KronMag") AS mea_mag FROM "obs" WHERE "ID" = \'HiTS094609-020243\' AND "Flag" = 0 AND "filter" = \'g\'',
              'select median("KronMag") from "obs" group by *']
    measurments = ['sources','obs', 'sources', 'obs', 'obs', 'obs']

    for query, measure in zip(query_list, measurments):
        print query
        get_ipython().magic(u'time result = client.query(query)')

        print pd.DataFrame(list(result.get_points(measurement=measure)))
        print '________________________________________________________________________________________'


if __name__ == '__main__':
    startTime = datetime.now()
    field = 'Blind15A_01'
    ccd = 'N1'
    test = False
    db_hostname = ''
    drop = False
    db_name = 'HiTS-test'

    if len(sys.argv) == 1:
        print help
        sys.exit()

	#read command line option.
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'F:c:t:i:d:')
    except getopt.GetoptError, err:
        print help
        sys.exit()

    for o, a in optlist:
        if o in ('-F'):
            field = str(a)
        elif o in ('-c'):
            ccd = str(a)
        elif o in ('-i'):
            db_hostname = str(a)
        elif o in ('-t'):
            if str(a) == 'True':
                test = True
            else:
                test = False
        elif o in ('-d'):
            if str(a) == 'True':
                drop = True
            else:
                drop = False
        else:
            continue

    if drop:
        client = InfluxDBClient(db_hostname, 8086, 'root', 'root', db_name)
        print 'Droping database %s' % (db_name)
        client.drop_database(db_name)

    if test:
        print 'Field: ', field
        print 'CCD  : ', ccd
        print 'local IP adress   : ', socket.gethostbyname(socket.gethostname())
        print 'DB host node name : ', db_hostname
        data_ingestion(field, ccd, hostname=db_hostname)
        sys.exit()
    data_ingestion(field, ccd)
    print 'It took', (datetime.now()-startTime), 'seconds'
    print '_______________________________________________________________________'
