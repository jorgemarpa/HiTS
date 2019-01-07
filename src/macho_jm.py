import pandas as pd
import numpy as np
import sys
import glob
import os
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
import FATS

mainpath = '/Users/jorgetil/Astro/HITS'

def calculate_tile(lcs_path, class_, save=False):

    member_names = np.sort(glob.glob('%s/lc_*.mjd' % (lcs_path)), kind='mergesort')
    # member_names = [member.name for member in members]
    done_ids = []
    done_features  = []
    for member_name in member_names[:1]:
        present = -1 # 0 = Both, 1 = B, 2 = R
        prefix = None
        id_ = None
        if member_name.endswith("B.mjd"):
            prefix = member_name[:-5]
            id_ = prefix[prefix.rfind('_')+1:-1]
            if prefix + 'R.mjd' in member_names:
                present = 0
            else:
                present = 1
        elif member_name.endswith("R.mjd"):
            prefix = member_name[:-5]
            id_ = prefix[prefix.rfind('_')+1:-1]
            if prefix + 'B.mjd' in member_names:
                present = 0
            else:
                present = 2
        prefix = os.path.basename(prefix)
        if id_ is not None and not id_ in done_ids:
            # try:
            if present == 0:
                # light_curve_file_string_B = tar.extractfile(tar.getmember(prefix + "B.mjd")).read()
                # light_curve_file_string_R = tar.extractfile(tar.getmember(prefix + "R.mjd")).read()
                lc_B = pd.read_csv('%s/%sB.mjd' % (lcs_path, prefix), header=2, delimiter=' ')
                lc_R = pd.read_csv('%s/%sR.mjd' % (lcs_path, prefix), header=2, delimiter=' ')
                features = calculate_lightcurve_2_filters(lc_B, lc_R)
            # elif present == 1:
            #     light_curve_file_string = tar.extractfile(tar.getmember(prefix + "B.mjd")).read()
            #     lc = pd.read_csv(StringIO(light_curve_file_string), header=2, delimiter=' ')
            #     features = calculate_lightcurve_1_filter(lc)
            # elif present == 2:
            #     light_curve_file_string = tar.extractfile(tar.getmember(prefix + "R.mjd")).read()
            #     lc = pd.read_csv(StringIO(light_curve_file_string), header=2, delimiter=' ')
            #     features = calculate_lightcurve_1_filter(lc)
            print(id_)
            done_ids.append(id_)
            done_features.append(features)
            # except IndexError:
                # print('Index error on {0}'.format(id_))
            # except:
                # print('Error on {0}'.format(id_))
    df = pd.DataFrame(done_features, index=done_ids, columns=['Amplitude','AndersonDarling','Autocor_length',
            'Beyond1Std','CAR_mean','CAR_sigma','CAR_tau','Color','Con','Eta_color','Eta_e',
            'FluxPercentileRatioMid20','FluxPercentileRatioMid35','FluxPercentileRatioMid50',
            'FluxPercentileRatioMid65','FluxPercentileRatioMid80','Freq1_harmonics_amplitude_0',
            'Freq1_harmonics_amplitude_1','Freq1_harmonics_amplitude_2','Freq1_harmonics_amplitude_3',
            'Freq1_harmonics_rel_phase_0','Freq1_harmonics_rel_phase_1','Freq1_harmonics_rel_phase_2',
            'Freq1_harmonics_rel_phase_3','Freq2_harmonics_amplitude_0','Freq2_harmonics_amplitude_1',
            'Freq2_harmonics_amplitude_2','Freq2_harmonics_amplitude_3','Freq2_harmonics_rel_phase_0',
            'Freq2_harmonics_rel_phase_1','Freq2_harmonics_rel_phase_2','Freq2_harmonics_rel_phase_3',
            'Freq3_harmonics_amplitude_0','Freq3_harmonics_amplitude_1','Freq3_harmonics_amplitude_2',
            'Freq3_harmonics_amplitude_3','Freq3_harmonics_rel_phase_0','Freq3_harmonics_rel_phase_1',
            'Freq3_harmonics_rel_phase_2','Freq3_harmonics_rel_phase_3','LinearTrend','MaxSlope','Mean',
            'Meanvariance','MedianAbsDev','MedianBRP','PairSlopeTrend','PercentAmplitude',
            'PercentDifferenceFluxPercentile','PeriodLS','Period_fit','Psi_CS','Psi_eta','Q31','Q31_color',
            'Rcs','Skew','SlottedA_length','SmallKurtosis','Std','StetsonJ','StetsonK','StetsonK_AC','StetsonL'])
    label = class_[class_.find('/')+1:]
    df['Type'] = label
    if save:
        df.to_csv('%s/features.csv' % (lcs_path), sep=',')
    return df

def calculate_lightcurve_2_filters(lc_B, lc_R):
    #We import the data
    [time, mag, error] = lc_B.values.T
    [time2, mag2, error2] = lc_R.values.T

    #We preprocess the data
    preproccesed_data = FATS.Preprocess_LC(mag, time, error)
    [mag, time, error] = preproccesed_data.Preprocess()

    preproccesed_data = FATS.Preprocess_LC(mag2, time2, error2)
    [mag2, time2, error2] = preproccesed_data.Preprocess()

    [aligned_mag, aligned_mag2, aligned_time, aligned_error, aligned_error2] = \
    FATS.Align_LC(time, time2, mag, mag2, error, error2)

    lc = np.array([mag, time, error, mag2, aligned_mag, aligned_mag2, aligned_time, aligned_error, aligned_error2])

    a = FATS.FeatureSpace(featureList=['PeriodLS', 'Period_fit',
                                'StructureFunction_index_21', 'StructureFunction_index_31', 'StructureFunction_index_32'])
    a = a.calculateFeature(lc)
    print a.result(method='dict')
    return a.result(method='array')

def calculate_lightcurve_1_filter(lc_X):
    #We open the ligth curve in two different bands
    #We import the data
    [time, mag, error] = lc_X.values.T
    #We preprocess the data
    preproccesed_data = FATS.Preprocess_LC(mag, time, error)
    [mag, time, error] = preproccesed_data.Preprocess()
    lc = np.array([mag, time, error])
    fs = FATS.FeatureSpace(Data=['magnitude','time','error'], featureList=None)
    features = fs.calculateFeature(lc).result(method='array')
    complete_features = features[0:7] + [0] + [features[7]] + [0] + features[8:52] + [0] + features[52:56] + [0] + features[56:59] + [0]
    return complete_features

if __name__ == '__main__':
    class_ = sys.argv[1]
    lcs_path = '%s/MACHO/%s' % (mainpath, class_)
    calculate_tile(lcs_path, class_, True)
