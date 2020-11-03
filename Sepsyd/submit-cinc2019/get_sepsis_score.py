#!/usr/bin/env python

import numpy as np
import xgboost as xgb
import pickle
#from model import LSTMsepsis
#from convModel import SepsisResNet
#import torch

varmeans = [84.58144338298742, 97.19395453339598, 36.977228240795384, 123.75046539637763, 82.40009988667639, 63.83055577034239, 18.72649785557987, 32.95765667291276, -0.6899191871174756, \
    24.075480562219358, 0.5548386348703284, 7.37893402619616, 41.02186880800917, 92.65418774854838, 260.22338482309493, 23.91545210569777, 102.48366144100076, 7.557530849328269, 105.82790991400108, \
        1.5106993531749389, 1.8361772575250843, 136.9322832898959, 2.646666023259181, 2.05145021490337, 3.544237652686153, 4.135527970939283, 2.114059461561731, 8.290099451999183, 30.79409334002751, 10.43083278791528, \
            41.231193461563706, 11.446405019759258, 287.38570591681315, 196.01391078961922, 62.00946887985519, 0.5592690422043409, 0.49657112470087744, 0.5034288752991226, -56.12512176894499, 26.994992301299437]
#varstds = [16.940419525356408, 2.908791565579173, 0.7803155565261937, 21.520551371399666, 15.044027397831087, 12.572755512817777, 5.395746029477869, 32, 4.286615170367131, 4.396157637432313, 0.18586829571925814, 0.07187656247404031, 8.996538990763671, 12.226073643364902, 1025.6112845802568, 20.154280585677434, 147.35641261897763, 0.819294062656942, 5.939883907159667, 1.5275309407915263, 4.629487005150567, 51.583313424071335, 2.3289583573870667, 0.3900247652587193, 1.4456138356096124, 0.6327154679428191, 5.242532026427244, 11.40695476095943, 4.874624921305417, 1.746014191005458, 23.964020613181315, 7.562615204886337, 158.61888749956518, 109.24308542907367, 16.133622114212926, 0.49392247045481574, 0.49994957884701907, 0.49994957884701907, 155.86483791787572, 28.190922444201767]
varstds = [17.3252, 2.9369, 0.77, 23.2316, 16.3418, 13.956, 5.0982, 7.9517, 4.2943, 4.3765, 11.1232, 0.0746, 9.2672, 10.893, 855.7468, 19.9943, 120.1227, 2.4332, 5.8805, 1.8056, 3.6941, 51.3107, 2.5262, 0.3979, 1.4233, 0.6421, 4.3115, 24.8062, 5.4917, 1.9687, 26.2177, 7.731, 153.0029, 103.6354, 16.3862, 0.4965, 0.5, 0.5, 162.2569, 29.0054]
varlogstds = [0.2069, 0.0338, 0.0209, 0.1862, 0.1929, 0.2133, 0.2811, 0.2632, 0.1, 0.1, 0.1, 0.0102, 0.2117, 0.1413, 1.3713, 0.6972, 0.6114, 0.6183, 0.0564, 0.6841, 1.4805, 0.3181, 0.6703, 0.1821, 0.3854, 0.1478, 1.0199, 2.723, 0.1788, 0.1896, 0.425, 0.5176, 0.5046, 0.5522, 0.3135, 0.1, 0.1, 0.1, 0.1, 0.9707]
varlogmeans = [4.4166, 4.576, 3.6101, 4.8009, 4.3928, 4.1334, 2.8926, 3.4633, 0.1, 0.1, 0.1, 1.9986, 3.6911, 4.5201, 4.104, 2.92, 4.3874, 1.9011, 4.6602, 0.1002, -0.5519, 4.8652, 0.7091, 0.7016, 1.1922, 1.4085, 0.0335, -0.8605, 3.4115, 2.327, 3.6051, 2.3098, 5.5347, 5.1412, 4.0841, 0.1, 0.1, 0.1, 0.1, 2.8862]
varmaxes = [280.0, 100.0, 50.0, 300.0, 300.0, 300.0, 100.0, 100.0, 100.0, 55.0, 4000.0, 7.93, 100.0, 100.0, 9961.0, 268.0, 3833.0, 27.9, 145.0, 46.6, 37.5, 988.0, 31.0, 9.8, 18.8, 27.5, 49.6, 440.0, 71.7, 32.0, 250.0, 440.0, 1760.0, 2322.0, 100.0, 1.0, 1.0, 1.0, 23.99, 336.0]
varmins = [20.0, 20.0, 20.9, 20.0, 20.0, 20.0, 1.0, 10.0, -32.0, 0.0, -50.0, 6.62, 10.0, 23.0, 3.0, 1.0, 7.0, 1.0, 26.0, 0.1, 0.01, 10.0, 0.2, 0.2, 0.2, 1.0, 0.1, 0.01, 5.5, 2.2, 12.5, 0.1, 34.0, 1.0, 14.0, 0.0, 0.0, 0.0, -5366.86, 1.0]


def get_sepsis_score(data, model):
    #print("New data")
    #print(data.shape)
    #print(data)
    data = np.copy(data)
    mask = np.ones(data.shape)
    nanIdx = np.where(np.isnan(data))
    delta = np.zeros(data.shape)
    vitaldelta = np.zeros((len(delta),7))
    variance = np.zeros((len(delta),8))
    vari_threehour = np.zeros((len(delta), 8))
    six_delta = np.zeros((len(delta), 8))
    three_delta = np.zeros((len(delta), 8))
    six_acc = np.zeros((len(delta), 8))
    three_acc = np.zeros((len(delta), 8))
    maxes = np.zeros((len(delta), 34))
    mins = np.zeros((len(delta), 34))
    three_max = np.zeros((len(delta), 34))
    three_min = np.zeros((len(delta), 34))
    swing = np.zeros((len(delta), 34))
    three_swing = np.zeros((len(delta), 34))

    if(len(delta)==1):
        grad1 = np.zeros((1,8))
        grad2 = np.zeros((1,8))
    else:
        grad1 = np.gradient(data[:,:8], axis=0)
        grad2 = np.gradient(grad1, axis=0)


    data[nanIdx] = np.take(varmeans, nanIdx[1])
    mask[nanIdx] = 0
    forward = np.copy(data[0,:])
    for t in range(data.shape[0]):
        for i in range(39):
            if mask[t, i]==1:
                forward[i] = data[t,i]
            else:
                data[t,i] = forward[i]
        z = 0 if t - 6 < 0 else t-6
        three = 0 if t-3<0 else t-3
        if t>0:
            #delta[t,:] = data[t,:]-data[t-1,:] # + delta[t-1, :] #Per step delta or all-time delta
            variance[t,:] = np.var(data[z:t+1, :8],axis=0)
            #vitaldelta[t,:] = delta[t,:7]
            six_delta[t,:] = np.mean(grad1[z:t+1, :],axis=0)
            three_delta[t,:] = np.mean(grad1[three:t+1,:],axis=0)
            six_acc[t,:] = np.mean(grad2[z:t+1,:],axis=0)
            three_acc[t,:] = np.mean(grad2[three:t+1,:],axis=0)

            vari_threehour[t, :] = np.var(data[three:t+1, :8], axis=0)
        maxes[t,:] = np.max(data[z:t+1, :34],axis=0)
        mins[t,:] = np.max(data[z:t+1, :34],axis=0)
        three_max[t,:] = np.max(data[three:t+1, :34],axis=0)
        three_min[t,:] = np.min(data[three:t+1, :34],axis=0)
        swing[t,:] = maxes[t,:]-mins[t,:]
        three_swing[t,:] = maxes[t,:]-mins[t,:]
    for i in range(0, 39):
        #if i in [14,15,16,19,20,22,25,26,27,30,31,32]:
        if i in [2, 11, 14, 15, 16, 17, 20, 21, 22, 23, 27, 28, 31, 32, 33, 34]:
            data[:,i] = 10*(np.log(data[:,i])-varlogmeans[i])/varlogstds[i]
        else:
            data[:,i] = 10*(data[:,i]-varmeans[i])/varstds[i]
    #data = np.concatenate((data, delta), axis=1)
    data = np.concatenate((data, mask), axis=1) #INCLUDE mask
    data = np.concatenate((data, variance, vari_threehour), axis=1)
    data = np.concatenate((data, maxes, mins, three_max, three_min), axis=1)
    data = np.concatenate((data, swing, three_swing), axis=1)
    #data = np.concatenate((data, data_entropy), axis=1)
    #data = np.concatenate((data, GRU_out), axis=1)
    data = np.concatenate((data, six_delta, six_acc, three_delta, three_acc), axis=1)

    #print(data)
    #print(data.shape)
    t = len(data)-1
    row = list(data[t, :])
    #print(len(row))
    row = np.array([row])
    #print(row.shape)

    dtest = xgb.DMatrix(row)
    pred_prob = model.predict(dtest)
    disc = pred_prob > 0.5

    return pred_prob, disc

def load_sepsis_model():
    #model = SepsisResNet(140, 1)
    #model.load_state_dict(torch.load("sepConvvFpos40L1h140b10lr0.0001k6ep2.pth"))

    #model = pickle.load(open("f120d4e02n8010val434.pickle.dat", "rb")) #409
    model = pickle.load(open("7dayUUfold2.pickle.dat", "rb"))
    return model
