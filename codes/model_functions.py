from scipy.optimize import minimize
import seaborn as sns
import pandas as pd 
import pylab as plt
import numpy as np
import copy
import timeit


# ----------- get the the values of du/dx -----------------    
def dudx(Dxt, t, p, t0=0, dt0=0, case='in-vivo'):
    if case=='in-vivo':
        f0  = np.exp(-1.0/(p['gamma']*dt0))
        w0s = 2*np.pi/(p['T0']*f0)
        A   = f0*np.exp( Dxt/p['a'] )
        w   = w0s/np.exp(Dxt/p['b'] )
        phi = w*Dxt + w0s*t0
        return A*( w*(Dxt/p['b'] + (w0s/w - 1.0))*np.cos(phi) - (1.0/p['a'])*np.sin(phi) )*dt0
    elif case=='ex-vivo':
        A = np.exp( Dxt/p['a'] )
        w = 2*np.pi/(p['T0']*np.exp( Dxt/p['b'] ))
        return A*((1.0/p['b'])*w*t*np.cos(w*t) - (1.0/p['a'])*np.sin(w*t))

# ----------- get the the points where du/dx = theta -----------------
def xsegment(d, p, case='in-vivo'):
    xs = []
    if case=='in-vivo':
        dt0 = 1.0/d['g']                                          
        for i in range(len(d['t'])-1):
#             j  = np.arange(i, len(d['t']), 1)
            j  = np.arange(i, np.min([i+3000, len(d['t'])]), 1)
            Dt = d['t'][j] - d['t'][i]                          
            du = dudx(Dt, d['t'][j], p, t0=d['t'][i], dt0=dt0[i])
            if np.sum(du>p['theta']):
                js = np.argmax(du>p['theta'])-1
                Dx = d['L_ef'][i+js] - d['L_ef'][i]                
                xs += [[d['t'][i+js], d['L_ef'][i], Dx, Dt[js]]]
    elif case=='ex-vivo':
        for t in p['tr']:
            Dx  = np.arange(0, p['L0'], 1)
            du  = dudx(Dx, t, p, case=case)
            if np.sum(du>p['theta']):
                js = np.argmax(du>p['theta'])
                xs += [[t, p['L0']-Dx[js]]]
    return np.array(xs)

# ----------- get the the segmentation points -----------------
def ixsegment(xs, case='in-vivo'):    
    xi = 0
    ti = 0
    ixs = []
    if case=='in-vivo':
        for i in range(1, len(xs[:,0])-1):
            if ((xs[i,0]-xs[i+1,0])<-10):
                ixs += [i]
                xi = xs[i,1]
                ti = xs[i,0]
    elif case=='ex-vivo':
        for i in range(1, len(xs[:,0])-1):
            if ((xs[i,1]-xs[i+1,1])<-50):
                ixs += [i+1]
                xi = xs[i,1]
                ti = xs[i,0]
    return np.array(ixs)   

# ----------- return the PSM length, somite size and period of somitogenesis -----------------
def model_properties(d, p):
    xs  = xsegment(d, p) 
    ixs = ixsegment(xs)   
    lxs = xs[ixs,1][1:] - xs[ixs,1][:-1]
    T = xs[ixs,0][1:] - xs[ixs,0][:-1]
    if len(ixs)>2:
        return [np.mean(xs[ixs,2]), np.mean(lxs), np.mean(T)]
    else:
        return [0, 0, 0]

# ----------- cost function to be minized to find the parameters that fit the data the best -----------------
def cost_function(x, d, p, params, case='in-vivo'):
    pcopy = copy.copy(p)
    for k in range(len(params)):
        pcopy[params[k]]  = x[k]*p[params[k]]

    xs = xsegment(d, pcopy, case=case)
    if len(xs)==0:
        return 1.0
    ixs = ixsegment(xs, case=case) 
    
    if case=='in-vivo':
        r_somites = np.arange(d['rSS'][0], d['rSS'][1]+1, 1)
        if (len(ixs)+d['SS0'])<=(d['rSS'][1]):
            return 1.0
        m_seg_len = xs[ixs,1][1:] - xs[ixs,1][:-1]      
        m_psm_len = xs[ixs,2]
        t_model = np.arange(d['SS0'], d['SS0']+len(ixs))

        # in order to compare the model with the data, lets put both in the same SS range.
        mask1 = np.in1d(t_model, r_somites)
        m_psm_len = m_psm_len[mask1]
        m_seg_len = m_seg_len[mask1[:-1]]

        mask2 = np.in1d(d['i_tSS'], r_somites)
        d_psm_len = d['i_PSM'][mask2]
        d_seg_len = d['i_seg'][mask2]

        if (len(m_psm_len)!=len(d_psm_len)) or (len(m_seg_len)!=len(d_seg_len)):
            return 1.0
        return 0.5*(np.mean((m_psm_len/d_psm_len - 1.0)**2) + np.mean((m_seg_len/d_seg_len - 1.0)**2))
                             
    elif case=='ex-vivo':
        psm = p['L0']-xs[ixs,1][1:]
        l   = xs[ixs,1][1:]-xs[ixs,1][:-1]
        v   = psm/(xs[ixs,0][1:] - xs[ixs,0][:-1])
        m_len = np.polyval(np.polyfit(psm, l, 2), d['rPSM'])
        m_vel = np.polyval(np.polyfit(psm, v, 2), d['rPSM'])
        return 0.5*(np.mean((m_len/d['len_m'] - 1.0)**2) + np.mean((m_vel/d['vel_m'] - 1.0)**2))

# ----------- parameter of the minimize function -----------------
def minimize_bstrap(D, p, params, xratio, bounds, maxiter=10, case='in-vivo'):
    if len(D)==2:
        out = minimize(cost_function_both, xratio, args=(D, p, params), 
                       bounds=bounds, method='L-BFGS-B', options={'eps': 1e-2, 'maxiter': maxiter})
        return out
    elif len(D)==1:
        out = minimize(cost_function     , xratio, args=(D[0], p, params, case), 
                       bounds=bounds, method='L-BFGS-B', options={'eps': 1e-2, 'maxiter': maxiter})
        return out

# ----------- cost function to minimize fit from two experimental condictions -----------------
def cost_function_both(x, d, p, params):
    return cost_function(x, d[0], p, params)+cost_function(x, d[1], p, params)
