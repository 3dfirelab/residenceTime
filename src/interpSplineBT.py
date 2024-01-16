from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np 
from scipy import interpolate
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb 
from scipy.signal import savgol_filter

#warnings.filterwarnings('ignore', '*umber of iterations maxit (set t*',)

###################################
def ApplyUnivariateSpline(time_fq_in, temp_cam_fq_in, dt_average=3):

    duration = (time_fq_in.max() - time_fq_in.min())
    nn = int(duration / dt_average) + 1
    time_all = np.linspace(time_fq_in.min(), time_fq_in.max(), nn)
    
    bt = np.zeros_like(time_all[:-1])
    t = np.zeros_like(time_all[:-1])
    for i, (tb,te) in enumerate(zip(time_all[:-1], time_all[1:])):
        idx = np.where( (time_fq_in>=tb) & (time_fq_in<te) )
        if len(idx[0])>0:
            bt[i] = temp_cam_fq_in[idx].mean()
            t[i] = time_fq_in[idx].mean()
        else:
            bt[i] = np.nan
            t[i] = np.nan

    idx_noNan = ~np.isnan(bt)
    time_fq_     = t[idx_noNan] 
    temp_cam_fq_ = bt[idx_noNan]

    err = []
    ss  = []
    s = 1.e6
    while s > 2.e-3:
        try:
            spl_cam = interpolate.UnivariateSpline(time_fq_, temp_cam_fq_, ext=3, s=s,)
            
            idx_ = np.where( temp_cam_fq_ > 600) 
            if len(idx[0])==0:
                idx = np.where((time_fq_<time_fq_[0]+dt_average) | (time_fq_>time_fq_[-1]-10*dt_average))
            err_ = np.sum(((temp_cam_fq_[idx_]-spl_cam(time_fq_[idx_])))**2, axis=0)
            err.append(err_)
            ss.append(s)
        except UserWarning:
            pass
        s /= 2
    
    idx_ = np.array(err).argmin()
    s = ss[idx_]
    spl_cam = interpolate.UnivariateSpline(time_fq_, temp_cam_fq_, ext=3, s=s,)

    return spl_cam, time_fq_, temp_cam_fq_


############################################
class tempts(object):
    
    def __init__(self,): 
        self.cameras   = 'optrisP400, agema550'

    def init(self,input_):
        self.var    = input_[0]
        self.fitmax = input_[1]
        self.p80    = input_[2]
        self.spl    = input_[3]
        

############################################
def start_process():
    print('Starting', multiprocessing.current_process().name)

##############################################
def star_interpolateTemperatureTimeSeries(args):
    return interpolateTemperatureTimeSeries(*args)

##############################################
def interpolateTemperatureTimeSeries(ij, time_in, dummy, temp_2din, frp_proxy, flag_plot, dir_out, 
                                     minbtTest=550, minbtOK=475, btActive=650):
  
    #340, 290, 550 for LWIR
    #550, 475, 650 for MIR

    temp_1d  = temp_2din[ :,ij[0],ij[1]]

    if temp_1d.max() < minbtTest:
        return 1, None, None, None, None

    if (np.std(np.sort(temp_1d)[-3:]) > 100) : return 2, None, None, None, None

    temp_fq = temp_1d
    time_fq = time_in

    if np.sort(temp_fq)[-3:].min() < minbtTest: 
        return 3, None, None, None , None

    
    idx = np.where(np.array(temp_fq)>=minbtOK)
    temp_fq  = np.array(temp_fq)[idx]
    time_fq  = np.array(time_fq)[idx] 
    
    #return 0, None, None, time_fq, temp_fq

    try:
        spl_bt, tbt, cbt = ApplyUnivariateSpline(time_fq, temp_fq, dt_average=2 )
    except: 
        spl_bt = None 

    if temp_fq.shape[0] > 60:
        temp_fq2  = savgol_filter(temp_fq, 21, 3)
        spl_bt2 = interpolate.UnivariateSpline(time_fq, temp_fq2, ext=3, )
    else:
        spl_bt2 = None


    if spl_bt is None: return 5, None, spl_bt2, time_fq, temp_fq


    idx_varts_ = np.where(temp_fq>btActive)
    if len(idx_varts_[0]) > 0: 
        idx_varts = np.where( (time_fq >= time_fq[idx_varts_].min()) & (time_fq <= time_fq[idx_varts_].max()) )
        spl_bt_max =   spl_bt(time_fq[idx_varts]).max()
        spl_bt_per =   np.percentile(temp_fq[idx_varts],80)
    else: 
        idx_varts = (np.array([], dtype=np.int64),)
        spl_bt_max = -99
        spl_bt_per = -99

    result = tempts()
    result.init([  ((spl_bt(time_fq[idx_varts])-temp_fq[idx_varts])**2).sum(),
                   spl_bt_max,  
                   spl_bt_per,
                   spl_bt,
                ])

    if flag_plot:
        print('*')
        fig = plt.figure()
        
        ax = plt.subplot(121)
        if len(idx_varts[0]) > 0: 
            ax.axvspan(time_fq[idx_varts].min(), time_fq[idx_varts].max(), color='0.5', alpha=0.5)
        
        #ax.scatter(time_fq, temp_lwir_fq, c='b')
        #ax.scatter(tlwir,clwir, c='k',marker='x')
        #ax.plot(time_fq, spl_lwir(time_fq), 'b-', lw=2, alpha=0.7, )
        
        ax.scatter(time_fq, temp_fq, c='orange')
        ax.scatter(tbt,cbt, c='k')
        ax.plot(time_fq, spl_bt(time_fq), c='orange', lw=2, alpha=0.7,)

        
        ax = plt.subplot(122)
        ax.imshow(frp_proxy.T, origin='lower')
        ax.scatter(ij[0],ij[1],c='k')

        fig.savefig('{:s}/{:03d}x{:03d}.png'.format(dir_out,ij[0],ij[1]))
        plt.close(fig)


    return 0, result, spl_bt2, time_fq, temp_fq

