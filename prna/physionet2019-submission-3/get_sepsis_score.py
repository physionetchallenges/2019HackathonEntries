#!/usr/bin/env python

import numpy as np
from model import RitsModel, TCN
import torch
from process import parse_dataMatrix, collate_fn, attributes
from utils_jr import to_var
import pandas as pd
from xgboost import XGBClassifier
import pickle
from collections import OrderedDict


flag_useCuda = False
device = torch.device('cpu')
if flag_useCuda:
    torch.cuda.set_device(0)
    device = torch.device('cuda')

def get_sepsis_score(data, model):
    # input for xgboost-greg and xgboost-saman
    features = model['features_xgboost_Greg']
    min_dict_plausibility = model['min_dict_plausibility']
    max_dict_plausibility = model['max_dict_plausibility']
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

    # input feature vector of xgb-greg
    x_greg = np.array([x_in[item] for item in features])[None,:]
    # input feature vector of xgb-saman
    x_saman = np.array([x_in[item+'_imputed'] for item in cols])[None,:]

    # apply 10-xgb ensemble to Greg and Saman's models
    score_greg, score_saman = 0., 0.
    for model_index in [3, 7, 9, 8, 0]:
        # prediction by xgb-greg
        clf_greg = model['gregModel_list'][model_index]
        threshold_greg = model['gregThreshold_list'][model_index]
        pred_prob_greg = clf_greg.predict_proba(x_greg)[0,1]
        score_greg += int(pred_prob_greg >= threshold_greg)
        # prediction by xgb-saman
        clf_saman = model['xgbModel_list'][model_index]
        threshold_saman = model['xgbThreshold_list'][model_index]
        pred_prob_saman = clf_saman.predict_proba(x_saman)[0,1]
        score_saman += int(pred_prob_saman >= threshold_saman)
    label_greg = int(score_greg/5. > (0.2+1e-3))
    label_saman = int(score_saman/5. > (0.2+1e-3))

    # normalize isMeasured variables
    # only take the last 10 measurements (due to the receptive field of TCN)
    idx_start = max(0, data.shape[0]-10)
    x_isMeasured = ~np.isnan(data[idx_start:,range(0,34)])
    x_isMeasured_norm = torch.from_numpy((x_isMeasured-model['isMeasured_mean'])/\
        model['isMeasured_std']).float().to(device)

    # run RITS imputation: apply log-transform before running RITS
    df = pd.DataFrame(data, columns=attributes)
    rec = parse_dataMatrix(df, False, model['rits_mean'], model['rits_std'], \
        log_transform=True)
    seq = collate_fn([rec])
    # get the number of models
    n_model = len(model['ritsModel_list'])
    assert(n_model==2)
    # get the sequence length
    seqLen = data.shape[0]
    # initialize the predicted score as the prediction of Greg and Saman
    score = label_greg + label_saman
    with torch.no_grad():
        seq_var = to_var(seq, device=device)
        for idx in range(n_model):
            rits_model = model['ritsModel_list'][idx]
            rits_threshold = model['ritsThreshold_list'][idx]
            tcn_model = model['tcnModel_list'][idx]
            tcn_threshold = model['tcnThreshold_list'][idx]

            # RITS prediction
            _, ret = rits_model.run_on_batch([seq_var], None)
            score += ((ret[0]['predictions'][0][-1]*(1-torch.exp(-0.58*torch.arange(\
                seqLen-1,seqLen).float().to(device))))>rits_threshold).float().item()
            
            # TCN prediction
            imputation = ret[0]['imputations']
            # concatenate imputed data (of last 10 measurements) and isMeasured
            x_tensor = torch.cat((imputation[0][idx_start:], x_isMeasured_norm), 1)
            # get the prediction of the last element
            pred_tcn = tcn_model.forward(x_tensor)[-1].item()
            score += (pred_tcn>tcn_threshold)
    # average score from 6 models
    score = score/6.
    label = int(score>(1./3+1e-3))
    return score, label

def load_sepsis_model():
    # the list of xgb-saman, xgb-greg, rits, tcn models and thresholds
    xgbModel_list, xgbThreshold_list = [], []
    gregModel_list, gregThreshold_list = [], []
    ritsModel_list, ritsThreshold_list = [], [] 
    tcnModel_list, tcnThreshold_list = [], []

    # load 1) 10 models from xgb-saman; and 2) 10 models from xgb-greg
    for testFold in range(10):
        # load xgb-saman models and thresholds
        with open("./trained_models/XGBmodel_testFold"+str(testFold)+".pkl",\
            "rb") as pkl_file:
            xgbModel_list.append(pickle.load(pkl_file))
        xgbThreshold_list.append(np.loadtxt("./trained_models/XGBthreshold_testFold"+\
            str(testFold)+".txt").item())
        # load xgb-greg
        input_file = './trained_models/xgboost_output_model_test_fold_'+\
            str(testFold)+'_with_threshold.pkl'
        with open(input_file, "rb") as pkl_file:
            greg_model, greg_threshold = pickle.load(pkl_file)
            gregModel_list.append(greg_model)
            gregThreshold_list.append(greg_threshold)

    # load RITS-log, TCN
    for testFold in [9, 3]:
        # load RITS models
        rits_model = RitsModel(40, 64, 0.5, device=device)
        path_rits = "./trained_models/RITSLog_testFold"+str(testFold)+".pkl"
        if flag_useCuda:
            rits_model.load_state_dict(torch.load(path_rits)['model_state_dict'])
        else:
            rits_model.load_state_dict(torch.load(path_rits, map_location=\
                lambda storage, loc: storage)['model_state_dict'])
        rits_model = rits_model.eval()
        if flag_useCuda:
            rits_model = rits_model.cuda()
        ritsModel_list.append(rits_model)
        ritsThreshold_list.append(np.loadtxt("./trained_models/RITSLogthreshold_testFold"+\
            str(testFold)+".txt").item())

        # load TCN models
        tcn_model = TCN(74, 1, [100], kernel_size=5, dropout=0.25)
        path_tcn = "./trained_models/TCNmodel_testFold"+str(testFold)+".pt"
        if flag_useCuda:
            tcn_model.load_state_dict(torch.load(path_tcn))
        else:
            tcn_model.load_state_dict(torch.load(path_tcn, map_location=\
                lambda storage, loc: storage))         
        tcn_model = tcn_model.eval()
        if flag_useCuda:
            tcn_model = tcn_model.cuda()
        tcnModel_list.append(tcn_model)
        tcnThreshold_list.append(np.load("./trained_models/TCNthreshold_testFold"+\
            str(testFold)+".npy")[0])

    # load median values used by xgboost
    feat_median = np.loadtxt("./trained_models/medianValues_vitalLab.csv")

    # load normalization mean, std for 34 vital+lab imputed by rits
    rits_mean = pd.read_csv("./trained_models/means_log.csv", header=None, \
        names=['param','value']).value.values
    rits_std = pd.read_csv("./trained_models/stds_log.csv", header=None, \
        names=['param','value']).value.values

    # load normalization mean, std for binary variables used by TCN
    isMeasured_mean = np.loadtxt("./trained_models/isMeasured_mean.csv")
    isMeasured_std = np.loadtxt("./trained_models/isMeasured_std.csv")

    # Load Plausibility check and feature statistics for Greg xgboost model
    df_plausibility = pd.read_csv('trained_models/Plausibility_Check.csv')
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
    # Load feature names used by Greg's xgboost model
    with open('trained_models/xgb_features.pkl', 'rb') as pkl_file:
        features_xgboost_Greg = pickle.load(pkl_file) 

    return {'xgbModel_list':xgbModel_list, 'xgbThreshold_list':xgbThreshold_list,\
        'ritsModel_list':ritsModel_list, 'ritsThreshold_list':ritsThreshold_list,\
        'tcnModel_list':tcnModel_list, 'tcnThreshold_list':tcnThreshold_list, \
        'feat_median':feat_median, 'rits_mean':rits_mean, 'rits_std':rits_std, \
        'isMeasured_mean':isMeasured_mean, 'isMeasured_std':isMeasured_std, \
        'gregModel_list': gregModel_list, 'gregThreshold_list':gregThreshold_list, \
        'df_plausibility': df_plausibility, 'features_xgboost_Greg':features_xgboost_Greg, \
        'min_dict_plausibility': min_dict_plausibility, \
        'max_dict_plausibility': max_dict_plausibility}
    
