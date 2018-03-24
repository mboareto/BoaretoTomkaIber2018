import seaborn as sns
import pandas as pd 
import pylab as plt
import numpy as np
import copy
import bootstrap_functions as bootf
import model_functions as modelf
from scipy.stats import norm

cmap = plt.cm.YlGnBu
plt.rcParams['font.family'] = 'Arial'
sns.set_style('white')
sns.set_context('notebook', font_scale=1.6, rc={'lines.linewidth': 2.0})
aspect_ratio = [6, 4.5]


def SStomin(d, ax=None):
    if ax==None:
        fig = plt.subplots(1, 1, figsize=(aspect_ratio[0], aspect_ratio[1]), sharey=False)
        ax  = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
    ax.plot(d['x_nSS']/(24*60.0), d['y_nSS'],'o--', ms=12,  c=cmap(0.45), label='experiment')
    ax.plot(d['t']/(24*60.0),     d['tSS'],    '-', lw=1.5, c=cmap(0.8),  label='fit')
    ax.set_ylabel('Somite stage')
    ax.set_xlabel('Time (embryonic days)')
    ax.legend(frameon=True, loc=2)

    
def growth_properties(d1, d2, c=[cmap(0.8), cmap(0.45)], save_fig=None):
    fig = plt.subplots(2, 2, figsize=(2*aspect_ratio[0], 2*aspect_ratio[1]), sharey=False)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)
    ax1.text(-.15, 1.0, 'A)', transform=ax1.transAxes, size=22)
    ax2.text(-.15, 1.0, 'B)', transform=ax2.transAxes, size=22)
    ax3.text(-.15, 1.0, 'C)', transform=ax3.transAxes, size=22)
    ax4.text(-.15, 1.0, 'D)', transform=ax4.transAxes, size=22)
    for i in range(2):
        d = d1 if i==0 else d2

        ax1.plot(d['G'].values[:,0], d['G'].values[:,1], 'o', ms=8, c=c[i], alpha=0.8, label='experiment')
        ax1.fill_between(d['tSS'], d['G_m']-1.96*d['G_s'], d['G_m']+1.96*d['G_s'], alpha=0.2, facecolor=c[i])
        ax1.plot(d['tSS'], d['G_m'], lw=2.0, c=c[i])   
        ax1.set_xlim([0, 60])
        ax1.set_ylim([0, 5])
        ax1.set_xlabel('Somite stage')
        ax1.set_ylabel('Axial growth rate (um/min)')

        bstrap_fitting(d['L'].values/np.array([24*60, 1000]), d['t']/(24*60), c=c[i], ax=ax2, return_m=False)
        ax2.set_xticks(range(8, 15, 1))
        ax2.set_ylabel('Axial length (mm)')
        ax2.set_xlabel('Time (embryonic days)')
        
        ax3.plot(d['f'].values[:,0], d['f'].values[:,1], 'o', ms=8, c=c[i], alpha=0.8, label='experiment')
        ax3.fill_between(d['tSS'], d['f_m']-1.96*d['f_s'], d['f_m']+1.96*d['f_s'], alpha=0.2, facecolor=c[i])
        ax3.plot(d['tSS'], d['f_m'], lw=2.0,c=c[i], alpha=1.0)   
        ax3.set_xlim([0, 60])
        ax3.set_ylim([0.3, 1.05])
        ax3.set_xlabel('Somite stage')
        ax3.set_ylabel('PSM/axial growth ratio')

        ax4.plot(d['tSS'], d['g_m'], lw=2.0, c=c[i], alpha=1.0)   
        ax4.fill_between(d['tSS'], d['g_m']-1.96*d['g_s'], d['g_m']+1.96*d['g_s'], alpha=0.2, facecolor=c[i])
        ax4.set_xlim([0, 60])
        ax4.set_ylim([0, 2.5])
        ax4.set_xlabel('Somite stage')
        ax4.set_ylabel('Tail bud growth rate (um/min)')
    plt.tight_layout()
    if save_fig!=None:
        plt.savefig(save_fig, format='pdf')

        
def segmentation_properties(d1, d2, nsampling=1000, npoly=5, c=[cmap(0.8), cmap(0.45)], save_fig=None):
    fig = plt.subplots(1, 2, figsize=(2*aspect_ratio[0], aspect_ratio[1]), sharey=False)
    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)
    ax1.text(-.19, 1.0, 'A)', transform=ax1.transAxes, size=22)
    ax2.text(-.19, 1.0, 'B)', transform=ax2.transAxes, size=22)
    for i in range(2):
        d = d1 if i==0 else d2

        bstrap_fitting(d['PSM'].values, d['tSS'], npoly=npoly, nsampling=nsampling, c=c[i], ax=ax1)
        ax1.set_xlabel('somite stage')
        ax1.set_ylabel('PSM length (um)')
        ax1.set_ylim([200, 1600])

        bstrap_fitting(d['seg'].values, d['tSS'], npoly=npoly, nsampling=nsampling, c=c[i], ax=ax2)
        ax2.set_xlabel('somite stage')
        ax2.set_ylabel('segment length (um)')
        ax2.set_ylim([20, 200])
    plt.tight_layout()
    if save_fig!=None:
        plt.savefig(save_fig, format='pdf')

        
def bstrap_fitting(X, xr, npoly=3, nsampling=1000, plot=True, return_m=True, c='g', ax=None):
    M = bootf.bstrap(X, xr, nsampling, npoly)
    M_m = np.mean(M,axis=1)
    M_s = np.std(M, axis=1)
    if plot:
        if ax==None:
            fig = plt.subplots(2, 2, figsize=(2*aspect_ratio[0], 2*aspect_ratio[1]), sharey=False)
            ax  = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
        ax.plot(X[:,0], X[:,1], 'o', ms=8, c=c, alpha=0.8, label='experiment')
        ax.plot(xr, M_m, lw=2.0, c=c, alpha=1.0)   
        ax.fill_between(xr, M_m-1.96*M_s, M_m+1.96*M_s, alpha=0.2, facecolor=c)      # area with 95% CI

        
def sensitive_analysis(d, p, list_pars, delta=[0.2], size=5, aspect=1.4, colors=[cmap(.8),cmap(.45),cmap(.1)],
                       xlabel=None, ylim=[-10, 25], save_fig=None):
    p0 = modelf.model_properties(d, p)
    v = ['PSM length', 'Somite size', 'Segm. period']
    m = []
    for par in list_pars:
        ppar = p[par] if par!='g' else d[par]
        for dp in delta:
            if par=='g':
                d[par]    = (1.0+dp)*ppar*np.ones(len(d['t']))
                d['L_ef'] = d['g']*d['t']
            else:
                p[par] = (1.0+dp)*ppar
            aux = modelf.model_properties(d, p)
            for i in range(3):
                m += [[par, v[i], 100*(aux[i]-p0[i])/p0[i], dp]]
            if par=='g':
                d[par]    = ppar*np.ones(len(d['t']))
                d['L_ef'] = d['g']*d['t']
            else:
                p[par] = ppar
    pD = pd.DataFrame(m, columns=['parameter', 'variable', 'value', 'delta'])

    g = sns.factorplot(x='parameter', y='value', hue='variable', data=pD, legend_out=False, 
                       palette=colors, size=size, aspect=aspect, kind='bar')
    g.set_ylabels('Output change (%)')
    plt.legend(frameon=True, loc=1)
    plt.ylim(ylim)
    if xlabel != None:
        plt.xticks(range(len(xlabel)), xlabel, size=20)
    if save_fig!=None:
        plt.savefig(save_fig, format='pdf')
        
        

def segmentation_points(d, p, save_fig=None):
    xs  = modelf.xsegment(d, p)                            
    ixs = modelf.ixsegment(xs)    

    fig = plt.subplots(1, 1, figsize=(7, 5), sharey=False)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
    plt.plot(d['t'], d['L_ef'], '-', c=cmap(0.7), lw=1.5, label='Posterior end')
    plt.plot(xs[:,0], xs[:,1], 'o', ms=6, c=cmap(0.15), alpha=0.5, label=r'$du/dx=\theta$')
    plt.plot(xs[ixs, 0], xs[ixs, 1], 'o', ms=8, c=cmap(0.5), label='Segmentation point')
    plt.xlabel('time (min)')
    plt.ylabel('length (um)')
    plt.legend(frameon=True, loc=2)
    plt.tight_layout()
    if save_fig!=None:
        plt.savefig(save_fig, format='pdf')

# --------------------------------------------------------------------------------------------   
    
    
# --------------------------------------------------------------------------------------------   

        
def modelfit(D, p, c=[cmap(0.8), cmap(0.45)],save_fig=None):
    
    fig = plt.subplots(len(D), 3, figsize=(3*aspect_ratio[0], aspect_ratio[1]*len(D)), sharey=False)
    ax = []
    for i in range(len(D)):
        ax  += [[plt.subplot2grid((len(D), 3), (i, 0), colspan=1, rowspan=1), 
                 plt.subplot2grid((len(D), 3), (i, 1), colspan=1, rowspan=1), 
                 plt.subplot2grid((len(D), 3), (i, 2), colspan=1, rowspan=1)]]
    
    for i in range(len(D)):
        d = D[i]
        xs  = modelf.xsegment(d, p)                              # getting the points where du/dx=theta
        ixs = modelf.ixsegment(xs)                               # getting segmentation points
        r_somites = np.arange(d['SS0'], d['SS0']+len(ixs), 1)
    
        psm_len = xs[ixs,2]                                 # estimating PSM length
        seg_len = xs[ixs,1][1:] - xs[ixs,1][:-1]            # estimating segmentation length

        t = d['t']-d['t0']

        ax[i][0].plot(d['nSS'].values[:,0]-d['t0'], d['nSS'].values[:,1], 'o', ms=8, alpha=0.8, c=c[i])
        ax[i][0].plot((xs[ixs,0]-d['t0'])[:41], r_somites[:41], '-', c=c[i], lw=2.0)
        ax[i][0].set_ylabel('somite stage')
        ax[i][0].set_xlabel('Time (min)')

        ax[i][1].plot(d['PSM'].values[:,0], d['PSM'].values[:,1], 'o', ms=8, c=c[i], alpha=0.8, label='experiment')
        ax[i][1].plot(r_somites[:41],  psm_len[:41], '-', c=c[i], lw=2.0)
        ax[i][1].set_xlabel('somite stage')
        ax[i][1].set_ylabel('PSM length (um)')
        ax[i][1].set_ylim([200, 1500])

        ax[i][2].plot(d['seg'].values[:,0], d['seg'].values[:,1],  'o', ms=8, c=c[i], alpha=0.8, label='experiment')
        ax[i][2].plot(r_somites[1:][:41], seg_len[:41], '-', c=c[i], lw=2.0)       
        ax[i][2].set_xlabel('somite stage')
        ax[i][2].set_ylabel('segment length (um)')
        ax[i][2].set_ylim([0, 250])
    plt.tight_layout()
    if save_fig!=None:
        plt.savefig(save_fig, format='pdf')
        
        
def bstrapfit(d1, d2, p, outx, params, c=[cmap(0.80), cmap(0.45)], ax=None, save_fig=None, case='in-vivo'):
    fig = plt.subplots(2, 2, figsize=(2*aspect_ratio[0], 2*aspect_ratio[1]), sharey=False)
    ax =[[plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1), 
          plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)],
         [plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1), 
          plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)]]
    
    ax[0][0].text(-.17, 0.97, 'A)',transform=ax[0][0].transAxes, size=20)
    ax[0][1].text(-.17, 0.97, 'B)',transform=ax[0][1].transAxes, size=20)
    ax[1][0].text(-.17, 0.97, 'C)',transform=ax[1][0].transAxes, size=20)
    ax[1][1].text(-.17, 0.97, 'D)',transform=ax[1][1].transAxes, size=20)

    if case=='in-vivo':
        for j in range(2):
            d = d1 if j==0 else d2
            r_somites  = np.arange(d['rSS'][0], d['rSS'][1]+1, 1)
            n_somites  = d['rSS'][1]-d['rSS'][0]+1
            n_sampling = np.shape(outx[j])[0]

            psm_len = np.zeros((n_somites, n_sampling))
            seg_len = np.zeros((n_somites, n_sampling))
            pcopy = copy.copy(p)
            dcopy = copy.copy(d)
            for i in range(n_sampling):
                for k in range(len(params)):
                    pcopy[params[k]]  = outx[j][i,k]*p[params[k]]

                dcopy['g'] = d['g_bstrap'][:,i]
                xs = modelf.xsegment(dcopy, pcopy)
                ixs = modelf.ixsegment(xs)
                t_model = np.arange(d['SS0'], d['SS0']+len(ixs))

                mask1 = np.in1d(t_model, r_somites)
                psm_len[:,i] = xs[ixs,2][mask1]
                l = xs[ixs,1][1:] - xs[ixs,1][:-1]
                seg_len[:,i] = l[mask1[:-1]] 

            m = np.mean(psm_len, axis=1)
            s = np.std( psm_len, axis=1)
            ax[0][j].plot(r_somites, m, lw=2.0,c=c[j], alpha=1.0) 
            ax[0][j].fill_between(r_somites, m-1.96*s, m+1.96*s, alpha=0.2, facecolor=c[j])
            ax[0][j].plot(d['PSM'].values[:,0], d['PSM'].values[:,1], 'o', ms=6, c=c[j], label='experiment')
            ax[0][j].set_xlabel('somite stage')
            ax[0][j].set_ylabel('PSM length (um)')
            ax[0][j].set_ylim([300, 1500])

            m = np.mean(seg_len, axis=1)
            s = np.std( seg_len, axis=1)
            ax[1][j].plot(r_somites, m, lw=2.0,c=c[j], alpha=1.0) 
            ax[1][j].fill_between(r_somites, m-1.96*s, m+1.96*s, alpha=0.2, facecolor=c[j])
            ax[1][j].plot(d['seg'].values[:,0], d['seg'].values[:,1], 'o', ms=6, c=c[j], label='experiment')
            ax[1][j].set_xlabel('somite stage')
            ax[1][j].set_ylabel('segment length (um)')
            ax[1][j].set_ylim([0, 250])
        
    elif case=='ex-vivo':
        for j in range(2):
            d = d1 if j==0 else d2
            if outx==None:
                m = d['len_m'] 
                s = d['len_s'] 
            else:
                n_sampling = np.shape(outx[j])[0]
                ml = np.zeros((len(d['rPSM']), n_sampling))
                mv = np.zeros((len(d['rPSM']), n_sampling))
                pcopy = copy.copy(p)
                for i in range(n_sampling):
                    for k in range(len(params)):
                        pcopy[params[k]]  = outx[j][i,k]*p[params[k]]

                    xs = modelf.xsegment(d, pcopy, case=case)
                    ixs = modelf.ixsegment(xs, case=case)
                    psm = p['L0']-xs[ixs,1][1:]
                    l = xs[ixs,1][1:]-xs[ixs,1][:-1]
                    v = psm/(xs[ixs,0][1:] - xs[ixs,0][:-1])
                    ml[:,i] = np.polyval(np.polyfit(psm, l, 2), d['rPSM'])
                    mv[:,i] = np.polyval(np.polyfit(psm, v, 2), d['rPSM'])
                    
                m = np.mean(ml, axis=1)
                s = np.std( ml, axis=1)
            ax[0][j].plot(d['rPSM'], m, lw=2.0, c=c[j], alpha=1.0)   
            ax[0][j].fill_between(d['rPSM'], m-1.96*s, m+1.96*s, alpha=0.2, facecolor=c[j])
            ax[0][j].plot(d['len'].values[:,0], d['len'].values[:,1], 'o', ms=8, c=c[j], alpha=1.0)
            ax[0][j].set_xlim([50, 400])
            ax[0][j].set_ylim([0, 80])
            ax[0][j].set_xlabel('mPSM length (um)')
            ax[0][j].set_ylabel('segment length (um)')

            if outx==None:
                m = d['vel_m'] 
                s = d['vel_s'] 
            else:
                m = np.mean(mv, axis=1)
                s = np.std( mv, axis=1)
            ax[1][j].plot(d['rPSM'], m, lw=2.0,c=c[j], alpha=1.0)   
            ax[1][j].fill_between(d['rPSM'], m-1.96*s, m+1.96*s, alpha=0.2, facecolor=c[j])
            ax[1][j].plot(d['vel'].values[:,0], d['vel'].values[:,1], 'o', ms=8, c=c[j], alpha=1.0)            
            ax[1][j].set_xlim([50, 400])
            ax[1][j].set_ylim([0, 3])
            ax[1][j].set_xlabel('mPSM length (um)')
            ax[1][j].set_ylabel('velocity (um/min)')
    plt.tight_layout()
    if save_fig!=None:
        plt.savefig(save_fig, format='pdf')   
        
        
        
def parameter_values(x1, x2, p, params, p_names, label, c=[cmap(0.8), cmap(0.45)], ylim=[0.2, 1.8], relative_values=True,
                     size=4.5, aspect=1.5, save_fig=None):
    m = []
    for j in range(len(params)):
        for i in range(np.shape(x1)[0]):
            m += [[p_names[j], label[0], x1[i,j]*p[params[j]], (x1/np.mean(x1, axis=0))[i,j]]]
            m += [[p_names[j], label[1], x2[i,j]*p[params[j]], (x2/np.mean(x1, axis=0))[i,j]]]        
    pD = pd.DataFrame(m, columns=['parameter', 'label', 'value', 'relative value'])

    
    if relative_values:
        g = sns.factorplot(x='parameter', y='relative value', hue='label', data=pD, legend_out=False, 
                           palette=c, size=size, aspect=aspect, kind='box')
        g.set_ylabels('Relative parameter values')
        g.set_xlabels('')
        plt.xticks(range(len(params)), p_names, size=20)
        plt.ylim(ylim)
        plt.legend(frameon=False, loc=1)
    else:
        
        g = sns.factorplot(x='label', y='value', hue='label', col='parameter', sharey=False,
                           data=pD, legend_out=False, palette=c, size=size, aspect=aspect, kind='box')
        g.set_ylabels('Parameter values')
        g.set_xlabels('')
    plt.tight_layout()
    if save_fig!=None:
        plt.savefig(save_fig, format='pdf')       

        
def exvivo_segmentation(d, p, save_fig=None):
    xs  = modelf.xsegment(d, p, case='ex-vivo')                            
    ixs = modelf.ixsegment(xs, case='ex-vivo')    
    lxs = xs[ixs,1][1:] - xs[ixs,1][:-1]
    T = xs[ixs,0][1:] - xs[ixs,0][:-1]
    psm = p['L0']-xs[ixs,1]


    fig = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    ax = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
    ax.text(-.15, 0.97, 'A',transform=ax.transAxes, size=20)
    plt.plot(p['tr'], p['tr']*0 + p['L0'], '-', c=cmap(1.0), lw=1.5, label='Posterior end')
    plt.plot(xs[:,0], xs[:,1], 'o', c=cmap(0.5), alpha=0.5, ms=6, label=r'$du/dx=\theta$')
    plt.plot(xs[ixs,0], xs[ixs,1], 'mo', ms=8, label='Segm. point')
    plt.ylim([0, p['L0']*1.05])
    plt.legend(frameon=True, loc=4)
    plt.xlabel('time (min)')
    plt.ylabel('length (um)')

    ax = plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)
    ax.text(-.15, 0.97, 'B',transform=ax.transAxes, size=20)
    plt.plot(xs[ixs,0], psm, 'o-', c=cmap(1.0), alpha=0.8)
    plt.ylabel('PSM length', color=cmap(1.0))
    plt.xlabel('time (min)')
    axt = ax.twinx()
    plt.plot(xs[ixs,0][1:], lxs, 'o-', c=cmap(0.7), alpha=0.8)
    plt.ylabel('Segment length', color=cmap(0.7))

    ax = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
    ax.text(-.15, 0.97, 'C',transform=ax.transAxes, size=20)
    plt.plot(psm[1:], lxs, 'mo-')
    plt.plot(d['len'].values[:,0], d['len'].values[:,1], 'o', c=cmap(0.5), alpha=0.5, ms=12)
    plt.xlabel('PSM length')
    plt.ylabel('Segment length')

    ax = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)
    ax.text(-.15, 0.97, 'D',transform=ax.transAxes, size=20)
    plt.plot(d['vel'].values[:,0], d['vel'].values[:,1], 'o', c=cmap(0.5), alpha=0.5, ms=12)
    plt.plot(psm[1:], psm[1:]/T, 'mo-')
    plt.xlabel('PSM length')
    plt.ylabel('Velocity (um/min)')
    plt.tight_layout()
    if save_fig!=None:
        plt.savefig(save_fig, format='pdf')
        
def modelfit_dynamicGrowth(d, p, g_param=[[3.0, 8.0], [0.5, 1.4]], species=['mouse','chicken','snake'], 
                           psm_lim=[[200, 1200],[0, 1600],[600, 1800]], seg_lim=[[60, 220],[60, 200],[0, 120]], 
                           x_lim=[[0, 60],[0, 60],[100, 300]], bounds=None, c=[cmap(0.8), cmap(0.45)], ax=None, save_fig=None):
    
    xs  = modelf.xsegment(d, p)                            
    ixs = modelf.ixsegment(xs)
    r_somites = np.arange(1, len(ixs)+1, 1)
    
    norm_dist = norm.pdf(xs[ixs,0], loc=np.max(d['t'])/g_param[0][0], scale=np.max(d['t'])/g_param[0][1])
    d['g'] = g_param[1][0] + g_param[1][1]*norm_dist/np.max(norm_dist)

    psm_len = xs[ixs,2]                                 # estimating PSM length
    seg_len = xs[ixs,1][1:] - xs[ixs,1][:-1]            # estimating segmentation length
    
    f = plt.subplots(6, 13, figsize=(12, 9), sharey=False)
    ax = plt.subplot2grid((6, 13), (0, 0), colspan=7, rowspan=3)
    ax.text(-.15, 0.97, 'A', transform=ax.transAxes, size=20)
    plt.plot(r_somites, d['g'], '-', lw=2.5, c=cmap(.9))
    plt.xlim([0, 50])
    plt.ylim([0.4, 2.0])
    plt.xlabel('Somite stage')
    plt.ylabel('Growth rate (um/min)')

    ax = plt.subplot2grid((6, 13), (3, 0), colspan=7, rowspan=3)
    ax.text(-.15, 0.97, 'B', transform=ax.transAxes, size=20)
    plt.plot(r_somites,  psm_len, '-', lw=2.5, c=c[0], label='PSM')
    plt.ylabel('PSM length (um)')
    plt.xlabel('Somite stage')
    plt.ylim([100, 1300])
    plt.legend(frameon=True, loc=0)
    ax = ax.twinx()
    seg_shift = np.float(np.max(psm_len))/seg_len[np.argmax(psm_len)]
    plt.plot(r_somites[1:],             seg_len, '-',  lw=2.0, c=c[1], label='segment')    
    plt.plot(r_somites[1:] - seg_shift, seg_len, '--', lw=2.5, c=c[1], label='segment')    
    plt.ylabel('Segment length (um)')
    plt.ylim([30, 200])
    plt.xlim([0, 50])
    plt.legend(frameon=True, loc=1)

    for i in range(len(species)):
        ax = plt.subplot2grid((6, 13), (2*i, 7), colspan=6, rowspan=2)
        if i==0:
            ax.text(-.25, 0.97, 'C',transform=ax.transAxes, size=20)
        plt.plot(d[species[i]]['Somite number'], d[species[i]]['PSM length'], 's-', ms=10.0, lw=2.0, c=c[0])
        plt.ylabel('PSM length (um)')
        plt.xlabel('Somite stage')
        plt.ylim(psm_lim[i])
        ax = ax.twinx()
        PS_ratio = (d[species[i]]['PSM length']/d[species[i]]['segment size'])[np.argmax(d[species[i]]['PSM length'])]
        plt.plot(d[species[i]]['Somite number'],          d[species[i]]['segment size'], 'o-', ms=11.0, lw=2.0, c=c[1])
        plt.plot(d[species[i]]['Somite number']-PS_ratio, d[species[i]]['segment size'], 'h--',ms=11.0, lw=2.0, c=c[1])
        plt.ylabel('Segment size (um)')
        plt.ylim(seg_lim[i])
        plt.xlim(x_lim[i])
    plt.tight_layout()
    if save_fig!=None:
        plt.savefig(save_fig, format='pdf')

        
        
def phaseDiagram_gxT0(d, p, ri, rj, crange, pi='g', pj='T0', nticks=10, cmap='YlGnBu', inter_species=False, 
                      label=['Tail bud growth rate (um/min)', 'Period at tail bud (min)'], save_fig=None):
    p2 = copy.copy(p)
    d2 = copy.copy(d)
    psm = np.zeros((len(rj), len(ri)))
    xl  = np.zeros((len(rj), len(ri)))
    ts  = np.zeros((len(rj), len(ri)))
    for i in range(len(ri)):
        d2[pi] = ri[i]*np.ones(len(d2['t']))
        d2['L_ef'] = d2['g']*d2['t']
        for j in range(len(rj)):
            p2[pj]  = rj[j]
            if inter_species:
                p2['a'] = p['a']*((d['g'][0]/d2['g'][0])**1.5)*(p2['T0']/p['T0']) 
                p2['b'] = p['b']*((d['g'][0]/d2['g'][0])**1.5)*(p2['T0']/p['T0']) 
            psm[j,i], xl[j,i], ts[j,i] = modelf.model_properties(d2, p2)
        
    X, Y = np.meshgrid(np.arange(0, len(ri)), np.arange(0, len(rj)))
    fig = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    ax = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
    plt.pcolormesh(X, Y, psm, cmap=cmap, vmin=np.min(crange[0]), vmax=np.max(crange[0])-1)
    plt.xticks(np.arange(0, len(ri), nticks))
    plt.yticks(np.arange(0, len(rj), nticks))
    ax.set_xticklabels(ri[::nticks])
    ax.set_yticklabels(rj[::nticks])
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    clb = plt.colorbar()
    clb.set_ticks(crange[0])
    clb.set_ticklabels(crange[0])
    clb.set_label('PSM length (um)', rotation=90)
    
    ax = plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)
    plt.pcolormesh(X, Y, xl, cmap=cmap, vmin=np.min(crange[1]), vmax=np.max(crange[1])-1)
    plt.xticks(np.arange(0, len(ri), nticks))
    plt.yticks(np.arange(0, len(rj), nticks))
    ax.set_xticklabels(ri[::nticks])
    ax.set_yticklabels(rj[::nticks])
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    clb = plt.colorbar()
    clb.set_ticks(crange[1])
    clb.set_ticklabels(crange[1])
    clb.set_label('Somite size (um)', rotation=90)

    ax = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
    plt.pcolormesh(X, Y, ts, cmap=cmap, vmin=np.min(crange[2]), vmax=np.max(crange[2])-1)
    plt.xticks(np.arange(0, len(ri), nticks))
    plt.yticks(np.arange(0, len(rj), nticks))
    ax.set_xticklabels(ri[::nticks])
    ax.set_yticklabels(rj[::nticks])
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    clb = plt.colorbar()
    clb.set_ticks(crange[2])
    clb.set_ticklabels(crange[2])
    clb.set_label('Segmentation period (min)', rotation=90)
    
    ax = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)
    plt.pcolormesh(X, Y, psm/xl, cmap=cmap, vmin=np.min(crange[3]), vmax=np.max(crange[3])-1)
    plt.xticks(np.arange(0, len(ri), nticks))
    plt.yticks(np.arange(0, len(rj), nticks))
    ax.set_xticklabels(ri[::nticks])
    ax.set_yticklabels(rj[::nticks])
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    clb = plt.colorbar()
    clb.set_label('PSM/S1 ratio', rotation=90)
    clb.set_ticks(crange[3])
    clb.set_ticklabels(crange[3])
    plt.tight_layout()
    if save_fig!=None:
        plt.savefig(save_fig, format='pdf')
