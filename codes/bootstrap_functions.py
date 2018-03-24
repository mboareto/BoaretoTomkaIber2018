import pandas as pd 
import numpy as np


def resample(X):
    n = np.shape(X)[0]
    resample_i = np.random.randint(0, n, n)
    X_resample = X[resample_i,:]
    return X_resample


def bstrap(X, xr, nsampling, npoly, log_scale=True):
    mx = np.zeros((len(xr), nsampling))
    mp = np.zeros((npoly+1, nsampling))
    for i in range(nsampling):
        Xs = resample(X)
        if log_scale:
            mp[:,i] = np.polyfit(np.log(Xs[:,0]), np.log(Xs[:,1]), npoly)
            mx[:,i] = np.exp(np.polyval(mp[:,i],  np.log(xr)))
        else:
            mp[:,i] = np.polyfit(Xs[:,0], Xs[:,1], npoly)
            mx[:,i] = np.polyval(mp[:,i], xr)
    return mx


def fit_parameters(d, npoly=3, nsampling=1000, case='in-vivo'):
    if case=='in-vivo':
        # fit the relationship between time and somite stage
        d['x_nSS'] = d['nSS'].values[:,0]                        # time in min
        d['y_nSS'] = d['nSS'].values[:,1]                        # number of somites
        d['p_nSS'] = np.polyfit(d['x_nSS'], d['y_nSS'], npoly)   # fit relationship time and somite stage
        d['tSS'  ] = np.polyval(d['p_nSS'], d['t'])              # converts the time to somite stage

        # use the previous fitted relationship to convert growth rate estimative from min to somite stage
        d['G'].values[:,0] = np.polyval(d['p_nSS'], d['G'].values[:,0])

        # bootstrap fitting of growth properties
        G = bstrap(d['G'].values, d['tSS'], nsampling, npoly, log_scale=True)
        f = bstrap(d['f'].values, d['tSS'], nsampling, npoly, log_scale=True)
        d['g_bstrap'] = G*f
        d['g_m'] = np.mean(G*f, axis=1)
        d['g_s'] = np.std( G*f, axis=1)
        d['G_m'] = np.mean(G,   axis=1)
        d['G_s'] = np.std( G,   axis=1)
        d['f_m'] = np.mean(f,   axis=1)
        d['f_s'] = np.std( f,   axis=1)

        d['fit_L'] = np.mean(bstrap(d['L'].values, d['t'], nsampling, npoly), axis=1)
        L = np.zeros(len(d['t']))
        L_ef = np.zeros(len(d['t']))                   # L_ef represents the tail bud growth rate
        L_ef[0] = d['fit_L'][0]
        for j in range(1, len(d['t'])):
            L_ef[j] = L_ef[j-1] + d['g_m'][j]*d['dt']
        d['L_ef'] = L_ef.astype(int)

        d['fit_PSM'] = np.mean(bstrap(d['PSM'].values, d['tSS'], nsampling, npoly, log_scale=True), axis=1)
        d['fit_seg'] = np.mean(bstrap(d['seg'].values, d['tSS'], nsampling, npoly, log_scale=True), axis=1)

        # getting fitted values of PSM and somite size for discrete somite stages 
        d['i_tSS'] = np.unique(np.floor(d['tSS']))
        d['i_seg'] = np.zeros(len(d['i_tSS']))
        d['i_PSM'] = np.zeros(len(d['i_tSS']))
        for i in range(len(d['i_tSS'])):
            d['i_PSM'][i] = np.mean(d['fit_PSM'][np.floor(d['tSS'])==d['i_tSS'][i]])
            d['i_seg'][i] = np.mean(d['fit_seg'][np.floor(d['tSS'])==d['i_tSS'][i]])
            
    elif case=='ex-vivo':
        d['len_bstrap'] = bstrap(d['len'].values, d['rPSM'], nsampling, npoly)
        d['vel_bstrap'] = bstrap(d['vel'].values, d['rPSM'], nsampling, npoly)
        d['len_m'] = np.mean(d['len_bstrap'], axis=1)
        d['len_s'] = np.std( d['len_bstrap'], axis=1)
        d['vel_m'] = np.mean(d['vel_bstrap'], axis=1)
        d['vel_s'] = np.std( d['vel_bstrap'], axis=1)

