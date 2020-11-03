#!/usr/bin/env python

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import pickle
from collections import OrderedDict


def get_sepsis_score(data, model):
    features = model['features_xgboost_Greg']
    min_dict_plausibility = model['min_dict_plausibility_Greg']
    max_dict_plausibility = model['max_dict_plausibility_Greg']
    feat_median = model['feat_median']

    cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
        'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
        'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
        'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
        'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
        'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
        'HospAdmTime', 'ICULOS']

    # apply plausbility check
    df = pd.DataFrame(data, columns=cols)
    
    cols_plausible = []
    for feat in cols:
        df[feat+'_plausible'] = df[feat].values
        cols_plausible.append(feat+'_plausible')

    # create new features after the plausibility check
    data_plausible = df[cols_plausible].values
    seqLen = data.shape[0]
    isMeasured = ~np.isnan(data_plausible)
    # measurement rate of each feature
    measurement_rate = np.sum(isMeasured,axis=0)/seqLen
    # whether each feature is ever measured
    ever_measured = (measurement_rate>0).astype(float)
    # whether each feature is currently measured
    currently_measured = (isMeasured[-1]).astype(float)
    # hours since last measured
    idx_lastMeasured = isMeasured.cumsum(0).argmax(0)
    hours_since_last_measured = np.ones(data.shape[1])*np.nan
    idxFeat_everMeasured = list(np.where(ever_measured)[0])
    hours_since_last_measured[idxFeat_everMeasured] = \
        (seqLen-idx_lastMeasured)[idxFeat_everMeasured]
    # imputed values: forward filling + median imputation
    imputed_val = np.array([data_plausible[idx_lastMeasured[j],j] for j in \
        range(40)])
    notMeasured = np.isnan(imputed_val)
    imputed_val[notMeasured] = 0
    imputed_val += notMeasured*feat_median

    x_in = OrderedDict()
    for idx, feat in enumerate(cols):
        x_in[feat] = data[-1,idx]
        x_in[feat+'_plausible'] = data_plausible[-1,idx]
        x_in[feat+'_currently_measured'] = currently_measured[idx]
        x_in[feat+'_imputed'] = imputed_val[idx]
        x_in[feat+'_ever_measured'] = ever_measured[idx]
        x_in[feat+'_hours_since_last_measured'] = hours_since_last_measured[idx]
        x_in[feat+'_measurement_rate'] = measurement_rate[idx]
    x_in_sel = np.array([x_in[item] for item in features])[None,:]
    score = 0.
    for model_index in range(10):
        # extract the model and threshold
        clf = model['xgbModel_list_Greg'][model_index]
        threshold = model['xgbThreshold_list_Greg'][model_index]
        pred_prob = clf.predict_proba(x_in_sel)[0,1]
        score += int(pred_prob >= threshold)
    score = score/10.
    label = int(score>(0.3+1e-3))
    return score, label

def load_sepsis_model():
    # the list of xgb models and thresholds
    xgbModel_list_Greg, xgbThreshold_list_Greg = [], []
    
    for testFold in range(10):
        # Load Greg xgboost models
        input_file = './trained_models/xgboost_output_model_test_fold_' + str(testFold) + '_with_threshold.pkl'
        pkl_file = open(input_file, 'rb')
        [model, threshold] = pickle.load(pkl_file)
        pkl_file.close()
        xgbModel_list_Greg.append(model)
        xgbThreshold_list_Greg.append(threshold)

    # Load Plausibility check and feature statistics for Greg xgboost model
    df_plausibility = pd.read_csv('trained_models/Plausibility_Check.csv')
    df_feature_statistics_Greg = pd.read_csv('./trained_models/feature_statistics.csv')
    
    cols_p = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
              'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
              'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
              'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
              'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
              'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
              'HospAdmTime', 'ICULOS']
    
    min_dict_plausibility = dict()
    max_dict_plausibility = dict()
    for f in cols_p:
        dfp = df_plausibility[df_plausibility['Parameter'] == f]
        try:
            minval = dfp['Plausibility2_min'].values[0]
            maxval = dfp['Plausibility2_max'].values[0]
            min_dict_plausibility[f] = minval
            max_dict_plausibility[f] = maxval
        except:
            min_dict_plausibility[f] = np.nan
            max_dict_plausibility[f] = np.nan
    
    # load feature median values
    feat_median = np.loadtxt("./trained_models/medianValues_vitalLab.csv")
    
    #Load feature names used by Greg's xgboost model
    input_file = 'trained_models/xgb_features.pkl'
    pkl_file = open(input_file, 'rb')
    features_xgboost_Greg = pickle.load(pkl_file)
    pkl_file.close()

    return {'xgbModel_list_Greg' : xgbModel_list_Greg, \
         'xgbThreshold_list_Greg' : xgbThreshold_list_Greg, \
         'plausibility_Greg' : df_plausibility, \
         'feature_statistics_Greg' : df_feature_statistics_Greg, \
         'features_xgboost_Greg' : features_xgboost_Greg, \
         'min_dict_plausibility_Greg' : min_dict_plausibility,  \
         'max_dict_plausibility_Greg' : max_dict_plausibility, \
         'feat_median': feat_median}
