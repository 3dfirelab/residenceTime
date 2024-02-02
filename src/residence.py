import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
import sys
from scipy.signal import savgol_filter, find_peaks 
from scipy import fftpack, interpolate 
import os
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import argparse
import importlib
import socket 
import datetime 
import glob 
import multiprocessing
import warnings 

#homebrewed
sys.path.append('../../GeorefIRCam/src/')
import tools
import spectralTools
import interpSplineBT

'''
search for arrival time (time_b) and end of flaming (time_e)
time_b is either the arrival time from segNablaT or if idx==0, then we use the first inflexion point before the max of the smoothed bt time series
time_e is the first inflexion point after bot max bt and max smooth bt are passed and spline interpolated bt_ is lower than 850
residence time is then time_e-time_b
'''

################################################
def string_2_bool(string):
    if  string in ['true', 'TRUE' , 'True' , '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
        return  True
    else:
        return False

################################################
def getIdxPass2Zeros(time_series):

    # Shift the array by one, padding with the first element
    shifted = np.roll(time_series, -1)

    # Find where the product is negative (crossing zero)
    # We exclude the last element as it's artificially introduced by the roll
    crossings = np.where( (  ( (np.sign(time_series[:-1] * shifted[:-1]) == -1) ) \
                           | ( (time_series[:-1] ==0) ) 
                          )
                          #& 
                          #(spl_(time_[:-1])<850
                          #) 
                        )

    return crossings 

################################################
def getIdxPass2Zeros_up(time_series):

    # Shift the array by one, padding with the first element
    shifted = np.roll(time_series, -1)

    # Find where the product is negative (crossing zero)
    # We exclude the last element as it's artificially introduced by the roll
    crossings = np.where( ( (np.sign(time_series[:-1] * shifted[:-1]) == -1) & (time_series[:-1]<0) ) \
                        | ( (time_series[:-1] ==0) & (np.insert(time_series[:-2],0,999) < 0)) )

    return crossings 

################################################
def getIdxPass2Zeros_down(time_series):

    # Shift the array by one, padding with the first element
    shifted = np.roll(time_series, -1)

    # Find where the product is negative (crossing zero)
    # We exclude the last element as it's artificially introduced by the roll
    crossings = np.where( ( (np.sign(time_series[:-1] * shifted[:-1]) == -1) & (time_series[:-1]>0) ) \
                        | ( (time_series[:-1] ==0) & (np.insert(time_series[:-2],0,999) > 0)) )

    return crossings                

################################################
def getIdxEndFlaming(idxb):
    #get end for residence time from the time where the derivative is getting higher than .95 tmes its min since the time of bt max. 
    #bt also needs to be lower than bt max -20
    idx_zeros = getIdxPass2Zeros(bt_deriv+1.)
    
    idx_ = np.where( (time_[idx_zeros] > time_[idxb] ) )
    if len(idx_[0])>0: 
        time_ctl_cooling = min( [ time_[-1], time_[idx_zeros][idx_].min() ] )
    else:
        time_ctl_cooling = time_[-1]

    bt_deriv_min =       bt_deriv[np.where( (time_>time_[idxb]) & (time_<time_[idxb]+120) & (time_<time_ctl_cooling) & (bt_>minbtTest) & (bt_<maxbtTest) )].min()
    bt_deriv_min_arg =   bt_deriv[np.where( (time_>time_[idxb]) & (time_<time_[idxb]+120) & (time_<time_ctl_cooling) & (bt_>minbtTest) & (bt_<maxbtTest))].argmin()
    bt_deriv_min_time =     time_[np.where( (time_>time_[idxb]) & (time_<time_[idxb]+120) & (time_<time_ctl_cooling) & (bt_>minbtTest) & (bt_<maxbtTest))][bt_deriv_min_arg]
    for ii in range(idx_arrivalT+1,len(bt_deriv),1):
        if (bt_deriv[ii] > 1.1*bt_deriv_min)  & (time_[ii]>bt_deriv_min_time) &  (bt_smooth[ii]< min([bt_smooth.max()-20, 850])): break
        #if bt_deriv[ii] < bt_deriv_prevmin: bt_deriv_prevmin =  bt_deriv[ii]

    return ii


################################################
def star_get_idx_bANDe(arg):
    return get_idx_bANDe(*arg)



################################################
def get_idx_bANDe(ij):
   
    nbre_skip = 0

    if ijc != None:
        if (ij[0] not in ijc[0]) | (ij[1] not in ijc[1]):
            return None,None,None,nbre_skip

    if flag_TestMissingPt_only == True:
        flag_plot = True
    else: 
        flag_plot = False
    
    temp_1d  = temp_data[ :,ij[0],ij[1]]

    if temp_1d.max() < minbtTest:
        return arrivaltime[ij[0],ij[1]],0,0,nbre_skip
    
    if temp_1d.max() < maxbtTest:
        return arrivaltime[ij[0],ij[1]], 0, 0, nbre_skip

    if (np.std(np.sort(temp_1d)[-3:]) > 100) : 
        nbre_skip += 1
        return None,None,None,nbre_skip

    if np.sort(temp_1d)[-3:].min() < minbtTest: 
        nbre_skip += 1
        return None,None,None,nbre_skip
        #continue 
    
    idx = np.where(np.array(temp_1d)>=minbtOK)
    bt_  = np.array(temp_1d)[idx]
    time_  = np.array(time_data)[idx] 

    
    if False: #(ij[0] == 122) & (ij[1]==330): 
        ax = plt.subplot(111)
        ax.plot(time_, bt_)
        ax.scatter(time_, bt_, c='k')
        plt.show()
        pdb.set_trace()

    #remove timeseries with not enough point
    #-------------
    if bt_ is None: 
        if flag_plot: pdb.set_trace()
        nbre_skip += 1
        return None,None,None,nbre_skip
        #continue
    
    while (time_[1]-time_[0]>5)  :  
        time_ = time_[1:]; bt_ = bt_[1:]
    while( time_[-1]-time_[-2]>5) :  
        time_ = time_[:-1]; bt_ = bt_[:-1]
    
    if bt_.shape[0] < 30: 
        #if flag_plot: pdb.set_trace()
        nbre_skip += 1
        return -888,-888,-888,nbre_skip # set for interpolation afterards
        #continue
    
    #check if we have smoldering picking up, in this case we keep only the first chunck above maxbtTest
    idx_tmp = np.where(bt_ > maxbtTest)
    if len(idx_tmp[0])> 1: 
        Diff_Time_hT = np.diff(time_[idx_tmp])
        if Diff_Time_hT.max() > 100: 
            idx_ = idx_tmp[0][np.where(Diff_Time_hT>100)[0].min()+1]
            time_ = time_[:idx_]
            bt_   = bt_[:idx_]
    
    
    #keep the  chunck with the max bt
    if (time_[1:]-time_[:-1]).max() > 10: 
        ii = 0
        difftime = time_[1:]-time_[:-1]
        arr_ = [[]]
        for ii in range(len(time_)-1):
            arr_[-1].append(ii)
            if ((time_[ii+1] - time_[ii]) > 10):
                arr_.append([])
            ii += 1
        btmax = bt_.max()
        flag_max_notFound = True
        for iarr_, arr__  in enumerate(arr_): 
            if len(arr__)==0: break
            if bt_[(arr__,)].max() == bt_.max(): 
                    time_ = time_[(arr__,)]
                    bt_   = bt_[(arr__,)]
                    flag_max_notFound = False
                    break

        if flag_max_notFound: 
            #if flag_plot: pdb.set_trace()
            nbre_skip += 1
            return -888,-888,-888,nbre_skip   

    if bt_.shape[0] < 30: 
        #if flag_plot: pdb.set_trace()
        nbre_skip += 1
        return -888,-888,-888,nbre_skip
        #continue
    

    #remove timeseries with not enough point
    #-------------
    if bt_ is None: 
        if flag_plot: pdb.set_trace()
        nbre_skip += 1
        return None,None,None,nbre_skip
        #continue
    
    
    if (time_[1:]-time_[:-1]).max() > 10:
        if flag_plot: pdb.set_trace()
        nbre_skip += 1
        return None,None,None,nbre_skip
        #continue
    
    if bt_.max() < maxbtTest: 
        #if flag_plot: pdb.set_trace()
        #nbre_skip += 1
        return arrivaltime[ij[0],ij[1]], 0, 0 ,nbre_skip
        #continue
    

    ssarr_test = []
    resiarr = []
    ss_selection = np.logspace(1,5)
    for iss, ss in enumerate(ss_selection): 
    #for ss in [1.e5]:
        # interpolate timeseries
        #--------
        bt_smooth  = savgol_filter(bt_, 21, 3)
        with warnings.catch_warnings(record=True) as w:
            spl_ = interpolate.UnivariateSpline(time_, bt_smooth, ext=3, )
            #spl_.set_smoothing_factor(4.e4)
            spl_.set_smoothing_factor(ss)
            bt_deriv = spl_.derivative()(time_)
            bt_2ndderiv = spl_.derivative(n=2)(time_)
            
            if len(w)>0:
                continue

        idx_test = np.where( (time_>=time_[bt_smooth.argmax()]-10) & (time_<=(time_[bt_smooth.argmax()]+10) ) )
        #if len(idx_test[0])==0: 
        #    pdb.set_trace()
        test = np.sqrt( ((bt_smooth[idx_test]-spl_(time_)[idx_test])**2).mean() ) 
        #test = np.abs(bt_smooth[idx_test]-spl_(time_)[idx_test]).max()
        #if test >10: 
        #    ss = ss_selection[iss-1]
        #    break
        #if np.abs(bt_smooth.max() - spl_(time_)[bt_smooth.argmax()])>10: break
        ssarr_test.append(test)
        #print(test)
    ss = ss_selection[ np.where(np.array(ssarr_test)<10)[0].max() ]

    bt_smooth  = savgol_filter(bt_, 21, 3)
    spl_ = interpolate.UnivariateSpline(time_, bt_smooth, ext=3, )
    spl_.set_smoothing_factor(ss)
    bt_deriv = spl_.derivative()(time_)
    bt_2ndderiv = spl_.derivative(n=2)(time_)

    idx_arrivalT = bt_smooth.argmax() 
    
    #check if max derivative has high enough bt
    if (bt_smooth[idx_arrivalT] < maxbtTest) : 
        return arrivaltime[ij[0],ij[1]], 0, 0 ,nbre_skip
    
    if idx_arrivalT < 3: 
        nbre_skip += 1
        return -888,-888,ss,nbre_skip  # not enough pt in residence time

        #continue

    #max bt is ok
    
    #arrival time from segmentation
   

    '''
    #second arrival time 
    #get it while scaning from max bt backwards where deriv lower than .95*max deriv withing 60s before max bt and bt < bt max -20 
    bt_deriv_max     =   bt_deriv[np.where( (time_<time_[idx_arrivalT]) & (time_>time_[idx_arrivalT]-60) )].max()
    bt_deriv_max_arg =   bt_deriv[np.where( (time_<time_[idx_arrivalT]) & (time_>time_[idx_arrivalT]-60) )].argmax()
    bt_deriv_max_time =     time_[np.where( (time_<time_[idx_arrivalT]) & (time_>time_[idx_arrivalT]-60) )][bt_deriv_max_arg]
    
    bt_deriv_prev = bt_deriv[idx_arrivalT]
    
    for ii in range(idx_arrivalT-1,0,-1):
        if (bt_deriv[ii]<.95*bt_deriv_prev) & (bt_smooth[ii]< min([bt_smooth.max()-20, 850])) & (time_[ii]<bt_deriv_max_time): break
        if bt_deriv[ii]>bt_deriv_prev: bt_deriv_prev=bt_deriv[ii]
        #bt_deriv_prev = bt_deriv[ii]
    idxb2 = ii
    '''
   

    try:
        time_b = time_[getIdxPass2Zeros(bt_2ndderiv)][ np.where( time_[getIdxPass2Zeros(bt_2ndderiv)]<time_[spl_(time_).argmax()]) ].max()
    except: 
        try: 
            time_b = time_[getIdxPass2Zeros(bt_2ndderiv+0.1)][ np.where( time_[getIdxPass2Zeros(bt_2ndderiv+0.1)]<time_[spl_(time_).argmax()]) ].max()
        except: 
            try:
                time_b = time_[np.where(spl_(time_)>maxbtTest)].min()
            except: 
                time_b = None
    
    if time_b == None: # spl_(time_).max() < maxbtTest, no flaming
        return arrivaltime[ij[0],ij[1]], 0, 0 ,nbre_skip
        #continue
  

    idxb1 = np.abs(time_ - arrivaltime[ij[0],ij[1]] ).argmin()
    idxb2 = np.abs(time_- time_b).argmin()
    
    if   (time_[idxb1] >=  time_[idx_arrivalT]-1 ) & (time_[idxb2] < time_[idx_arrivalT]-1): 
        idxb = idxb2
    elif (time_[idxb1] < time_[idx_arrivalT]-1) & (time_[idxb2] >=  time_[idx_arrivalT]-1): 
        idxb = idxb1
    elif (time_[idxb1] >=  time_[idx_arrivalT]-1) & (time_[idxb2] >=  time_[idx_arrivalT]-1):
        nbre_skip += 1
        #if flag_plot: pdb.set_trace()
        return -888,-888,ss,nbre_skip # set for interpolation afterwards, something fishy with idxb1 and idxb2
        #continue
    else:
        if idxb1>1: 
            idxb = idxb1
        else:
            idxb = max([idxb1,idxb2 ])


    try:
        corr_2ndderivmin = 0 
        idx_2ndbtZeros = getIdxPass2Zeros(bt_2ndderiv+corr_2ndderivmin)
        try: 
            idx_2ndbtZeros_ok = np.where( (time_[idx_2ndbtZeros]>time_[spl_(time_).argmax()])     & 
                                          (time_[idx_2ndbtZeros]>time_[np.where(bt_>.9*bt_.max())].max() ) & 
                                          (time_[idx_2ndbtZeros]>time_[bt_smooth.argmax()])    )[0].min()
        except: 
            idx_2ndbtZeros_ok = np.where( (time_[idx_2ndbtZeros]>time_[spl_(time_).argmax()])     & (time_[idx_2ndbtZeros]>time_[bt_smooth.argmax()])    )[0].min() 
        
        time_e = time_[idx_2ndbtZeros][idx_2ndbtZeros_ok]
        
        #time_e = time_[getIdxPass2Zeros(bt_2ndderiv+corr_2ndderivmin)][ np.where( (time_[getIdxPass2Zeros(bt_2ndderiv+corr_2ndderivmin)]>time_[spl_(time_).argmax()])  & 
        #                                                                          (time_[getIdxPass2Zeros(bt_2ndderiv+corr_2ndderivmin)]>time_[bt_smooth.argmax()])   )].min() 


        idxe = np.abs(time_-time_e).argmin()
        if spl_(time_)[idxe] > 850: 
            try:
                idx_2ndbtZeros = getIdxPass2Zeros(bt_2ndderiv+corr_2ndderivmin)
                idx_2ndbtZeros_ok = np.where( (time_[idx_2ndbtZeros]>time_[spl_(time_).argmax()])  & 
                                              (time_[idx_2ndbtZeros]>time_[bt_smooth.argmax()])    &
                                        (spl_(time_)[idx_2ndbtZeros]<850  )                        )[0].min()
                time_e = time_[idx_2ndbtZeros][idx_2ndbtZeros_ok]
                idxe = np.abs(time_-time_e).argmin()
            except: 
                pass

        idx_peak = find_peaks(bt_2ndderiv[idx_arrivalT:idxe])[0]
        if len(idx_peak)>0:
            idxe_ = idxe
            for idx__ in idx_peak:
                idxe__ = idx_arrivalT+idx__
                if  spl_(time_)[idxe__] > 850: 
                    continue
                elif time_[idxe__] < time_[np.where(bt_>.9*bt_.max())].max(): 
                    continue
                else: 
                    idxe_ = idxe__
                    break

            resi_ = time_[idxe_] - time_[idxb]
            if resi_ > 5.:     #only change idxe if residence time is > than 5s
                idxe = idxe_
        

        time_e = time_[idxe]
    except: 
        time_e = None

    if (time_e == None) & (spl_(time_).max()<650):
        return time_b,0,ss,nbre_skip

    if time_e == None:
        nbre_skip += 1
        #if flag_plot: pdb.set_trace()
        return -888,-888,ss,nbre_skip
    
    idxe = np.abs(time_- time_e).argmin()
    
    if bt_.max() > maxbtTest_hightT: 
        ratio_highT_inResidenceTime = np.where((bt_>maxbtTest_hightT) & (time_<time_e) & (time_>time_b))[0].shape[0]/np.where(bt_>maxbtTest_hightT)[0].shape[0]
        if ratio_highT_inResidenceTime < 0.2:
            nbre_skip += 1
            return -888,-888,ss,nbre_skip  # not enough pt in residence time

    if ijc != None:
        if (ij[0] in ijc[0]) & (ij[1] in ijc[1]): 
           flag_plot = True

    if flag_plot:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.scatter(time_, bt_, c='k', label='bt')
        #ax.plot(time_, spl_bt(time_),c='orange', label='spl_bt 1st')
        ax.plot(time_, bt_smooth,c='green', label='bt_smooth')
        ax.plot(time_, spl_(time_),c='red', label='spl_bt')
        ax.legend()
        ax = ax.twinx() 
        ax.plot(time_, spl_.derivative(n=2)(time_), c='k', linestyle=':', label='deriv')
        #ax.axvline(x=arrivalTime[ij[0],ij[1]],c='k')
        ax.axvline(time_[idx_arrivalT],c='k', linestyle=':', label='bt_smooth max')
        ax.axvline(time_[idxb],c='k')
        ax.axvline(time_[idxb1],c='r', linestyle=':', label='idxb1')
        ax.axvline(time_[idxb2],c='r', linestyle='--', label='idxb2')
        ax.axvspan(time_[idxb], time_[idxe], alpha=.5, color='green')
        ax.legend(loc='lower left')
        
        #ax.set_xlim(time_[idx_arrivalT]-60,time_[idxe]+100)
        ax.set_title('{:.2f} {:.3e}'.format(time_[idxe] - time_[idxb], ss))
        #ax = plt.subplot(122)
        #plt.plot(ssarr, resiarr)

        #fig.savefig(dir_out+'png/resi_{:03d}x{:03d}.png'.format(ij[0],ij[1]), dpi=100)
        #plt.close(fig)
        plt.show()
        pdb.set_trace()
        #sys.exit()

    return time_[idxb]  , time_[idxe] - time_[idxb] , ss,nbre_skip


##################################   
if __name__ == '__main__':
##################################   
       
    importlib.reload(interpSplineBT)

    parser = argparse.ArgumentParser(description='get residence time from radiance time series, either mir or lwir')
    parser.add_argument('-i','--input', help='Input run name',required=True)
    parser.add_argument('-mode','--mode', help='either lwir or mwir',required=True)
    parser.add_argument('-c','--check', help='list of idx to check',required=False)
    parser.add_argument('-s','--restart', help='True is we redo the computation',required=False)
    parser.add_argument('-p','--parallel', help='use multiproc',required=False)

    args = parser.parse_args()

    #define Input
    if args.input.isdigit():
        if args.input == '1':
            runName = 'test'
        else:
            print('number not defined')
            sys.exit()
    else:
        runName = args.input
    mode = args.mode
   
    if args.check == None:
        ijc = None
    else: 
        ijc0,ijc1 = args.check.split(';')
        ijc = [[],[]]
        for ii,jj in zip(ijc0[1:-1].split(','),ijc1[1:-1].split(',')):
            ijc[0].append(int(ii))
            ijc[1].append(int(jj))


    if args.restart == None:
        flag_restart = False
    else: 
        flag_restart = string_2_bool( args.restart )
    
    if args.parallel == None:
        flag_parallel = False
    else: 
        flag_parallel = string_2_bool( args.parallel )

    flag_TestMissingPt_only = False

    inputConfig = importlib.machinery.SourceFileLoader('config_'+runName,os.getcwd()+'/../../GeorefIRCam/input_config/config_'+runName+'.py').load_module()
    
    # input parameters
    params_grid           = inputConfig.params_grid
    params_gps            = inputConfig.params_gps 
    params_camera_lwir    = inputConfig.params_lwir_camera
    params_camera_mir     = inputConfig.params_mir_camera
    params_camera_visible = inputConfig.params_vis_camera
    params_rawData        = inputConfig.params_rawData
    params_georef         = inputConfig.params_georef
   
    if mode == 'lwir': 
        params_camera = params_camera_lwir
    elif mode == 'mwir': 
        params_camera = params_camera_mir
    else: 
        print('mode not well setup, mode = ', mode)
        sys.exit()

    if socket.gethostname() == 'coreff':
        path_ = '/media/paugam/goulven/data/'
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/paugam/data/',path_)
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/paugam/data/',path_)
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/paugam/data/',path_)
    elif socket.gethostname() == 'ibo':
        path_ = '/space/paugam/data/'
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/','/space/')
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/','/space/')
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/','/space/')
    
    elif socket.gethostname() == 'moritz':
        path_ = '/media/paugam/toolcoz/data/'
        inputConfig.params_rawData['root'] = inputConfig.params_rawData['root'].\
                                             replace('/scratch/globc/paugam/data/',path_)
        inputConfig.params_rawData['root_data'] = inputConfig.params_rawData['root_data'].\
                                                  replace('/scratch/globc/paugam/data/',path_)
        inputConfig.params_rawData['root_postproc'] = inputConfig.params_rawData['root_postproc'].\
                                                      replace('/scratch/globc/paugam/data/',path_)


    plotname          = params_rawData['plotname']
    root_postproc     = params_rawData['root_postproc']   
    wavelength_resolution = 0.01

    #dirinROS     = root_postproc + 'ROS_Sensitivity_{:s}/output_dx=01.0_dt=000.0_lwir_deepLv15_fsrbf/'.format(params_camera_lwir['dir_input'])
    dirinROS     = root_postproc + 'ROS_Sensitivity_{:s}/output_dx=01.0_dt=000.0__mir_SegNablaT_fsrbf/'.format(params_camera_mir['dir_input'])

    #LWIR set parameter for radiance/temperature conversion 
    if mode == 'lwir': 
        srf_file_lwir = '../../GeorefIRCam/data_static/Camera/'+params_camera_lwir['camera_name'].split('_')[0]\
                        +'/SpectralResponseFunction/'+params_camera_lwir['camera_name'].split('_')[0]+'.txt'
        param_set_radiance = [srf_file_lwir, wavelength_resolution]
        param_set_temperature = spectralTools.get_tabulated_TT_Rad(srf_file_lwir, wavelength_resolution)
    elif mode == 'mwir': 
        srf_file_mir = '../../GeorefIRCam/data_static/Camera/'+params_camera_mir['camera_name'].split('_')[0]\
                        +'/SpectralResponseFunction/'+params_camera_mir['camera_name'].split('_')[0]+'.txt'
        param_set_radiance = [srf_file_mir, wavelength_resolution]
        param_set_temperature = spectralTools.get_tabulated_TT_Rad(srf_file_mir, wavelength_resolution)


    #load ignition time
    ###################
    file_ignition_time = params_rawData['root_data'] + 'ignition_time.dat'
    f = open(file_ignition_time,'r')
    lines = f.readlines()
    ignitionTime = datetime.datetime.strptime(params_rawData['fire_date']+'_'+lines[0].rstrip(), "%Y-%m-%d_%H:%M:%S")
    endTime = datetime.datetime.strptime(params_rawData['fire_date']+'_'+lines[1].rstrip(), "%Y-%m-%d_%H:%M:%S")
    fire_durationTime = (endTime-ignitionTime).total_seconds()


    if mode == 'lwir': 
        dir_out = root_postproc+'ResidenceTime/InterpLWIR/'
    elif mode == 'mwir': 
        dir_out = root_postproc+'ResidenceTime/InterpMWIR/'
    tools.ensure_dir(dir_out)
    tools.ensure_dir(dir_out+'png/')

    
    #transmittance
    ###################
    #if runName == 'knp14sha1_301e': 
    #    trans_mir = 0.939
    #    trans_lwir = 0.607
    #else: 
    if mode == 'lwir': 
        transmittance = 1.
    elif mode == 'mwir': 
        transmittance = 1.


    #load grid
    ###################
    grid = np.load(root_postproc+'grid_'+plotname+'.npy')
    grid = grid.view(np.recarray)
    
    if runName == 'knp14sku4_301_05':
        nx_, ny_ = grid.shape
        grid2 = np.zeros([int(nx_/2),int(ny_/2)], dtype=grid.dtype)
        grid2 = grid2.view(np.recarray)
        grid2.grid_e = tools.downgrade_resolution_4nadir(grid.grid_e, [int(nx_/2),int(ny_/2)] , flag_interpolation='min' ) 
        grid2.grid_n = tools.downgrade_resolution_4nadir(grid.grid_n, [int(nx_/2),int(ny_/2)] , flag_interpolation='min' )
        grid2.mask   = tools.downgrade_resolution_4nadir(grid.mask,   [int(nx_/2),int(ny_/2)] , flag_interpolation='average' ) ; grid2.mask = np.where(grid2.mask>1., 2, 0)

        grid = grid2
        
    grid_e, grid_n, plotMask  = grid.grid_e, grid.grid_n, grid.mask
    dx = grid_e[1,1]-grid_e[0,0]
    dy = grid_n[1,1]-grid_n[0,0]
    nx, ny = grid_e.shape
    
    #load radiance
    ###################
    if not(os.path.isfile(dir_out +'RadData.npy')):
        
        rad_data  = []
        time_data = []
        if mode == 'lwir': 
            filesnpy = sorted(glob.glob(root_postproc+params_camera['dir_input']+'Georef_refined_SH/npy/{:s}_georef2nd*.npy'.format(params_rawData['plotname'])))
        elif mode == 'mwir':
            print('load mir frame and keep good frame ...')
            dir_out_mir            = root_postproc + params_camera_mir['dir_input']
            dir_out_mir_georef_npy = dir_out_mir + 'Georef3_{:s}/npy/'.format('SH')
            dir_out_mir_georef_npy_p2p = dir_out_mir_georef_npy+'/../npy_p2p/'
            mir_id, mir_time, mir_georef = tools.load_good_mir_frame_selection(True, dir_out_mir_georef_npy,
                                                                       plotname, path_, georefMode='SH')
            filesnpy =  ['{:s}{:s}_georef_{:06d}.npy'.format(dir_out_mir_georef_npy_p2p,plotname,iid) for iid in mir_id]
              
        for ifile, file_ in enumerate(filesnpy):
            if mode == 'lwir':
                info_, homogr, maskfull_, temp_, rad_, _ = np.load(file_, allow_pickle=True, encoding='latin1')
                time_data.append(info_[1])
                rad_data.append(rad_)

            elif mode == 'mwir':
                try: 
                    georef_mir_rad, georef_mir_temp, pixelSize = np.load(file_,allow_pickle=True)
                except: 
                    georef_mir_rad, georef_mir_temp, pixelSize = np.load(file_,allow_pickle=True, encoding='latin1')

                time_data.append(mir_time[ifile])
                rad_data.append(georef_mir_rad)

        rad_data = np.array(rad_data)
        time_data = np.array(time_data)
        
        if runName == 'knp14sku4_301_05':
            rad_data2 = []
            nx_, ny_ = rad_data.shape[1:]
            for rad_data_ in rad_data:
                rad_data2.append( tools.downgrade_resolution_4nadir(rad_data_, [int(nx_/2),int(ny_/2)] , flag_interpolation='conservative' ) )
            rad_data = np.array(rad_data2)
        
        np.save(dir_out +'timeData.npy',time_data)
        np.save(dir_out +'RadData.npy',rad_data)
        
        rad_data = None
        time_data     = None

    if socket.gethostname() != 'moritz':
        rad_data   = np.load(dir_out+'RadData.npy',  mmap_mode='r')
        time_data      = np.load(dir_out+'timeData.npy'                   ) 
    else: 
        rad_data  = np.load(dir_out+'RadData.npy')
        time_data      = np.load(dir_out+'timeData.npy')

  
    #conversion to bt with atmospheric transmittance correction
    ###################
    temp_data = spectralTools.conv_Rad2Temp(rad_data/transmittance, param_set_temperature)
    print('data loaded')

    '''
    #spline interpolation
    ###################
    frp_proxy = rad_data.sum(axis=0)
    step = 1
    N = np.where(grid.mask==2)[0].shape[0]/step
    if not(os.path.isfile(dir_out +'splineInterp.npy')):
        print('interpolate bt timeseries ...   ')
        args_here = []
        flag_plot = False
        if mode == 'lwir': 
            minbtTest, minbtOK, btActive = 340, 290, 550 
        elif model == 'mwir':
             minbtTest, minbtOK, btActive = 550, 475, 650 

        for ii, ij in enumerate(list(zip(*np.where(grid.mask==2)))[::step]):
            args_here.append([ij, time_data, None, lwir_temp_data, frp_proxy, flag_plot, dir_out+'png/', minbtTest, minbtOK, btActive])
        
        flag_parallel_ = True
        if flag_parallel_:
            # set up a pool to run the parallel processing
            cpus = tools.cpu_count()
            pool = multiprocessing.Pool(processes=cpus)

            # then the map method of pool actually does the parallelisation  
            results = pool.map(interpSplineBT.star_interpolateTemperatureTimeSeries, args_here)
            
            pool.close()
            pool.join()
           
        else:
            results = []
            for ii, arg in enumerate(args_here):
                print(u'{:4.1f}%  {:d}\r'.format(100.*ii//N, ii), end=' ') 
                sys.stdout.flush()
                results.append(interpSplineBT.star_interpolateTemperatureTimeSeries(arg))

        

        spline_temp = [] 
        flag_return = np.zeros_like(plotMask) - 999
        for ii, (arg,res) in enumerate(zip(args_here,results)):
            ij, time_data, _, lwir_temp_data, frp_proxy, flag_plot, plotDir, minbtTest, minbtOK, btActive = arg
            flag_interp, res_, spl_bt2, time_, bt_ = res 

            flag_return[ij[0],ij[1]] = flag_interp

            #if flag_interp!=0: 
            #    spl_bt = None
            #else: 
            #    spl_bt = res_.spl
            
            spl_bt = None

            spline_temp.append([ij, time_, bt_, spl_bt ]) 


        np.save(dir_out+'splineInterp',np.array(spline_temp, dtype=object)) 
        np.save(dir_out+'splineInterp_flag',flag_return)

    else: 
        spline_temp = np.load(dir_out+'splineInterp.npy', allow_pickle=True)
    
    '''

    arrivaltime = np.load(dirinROS+'/arrivalTime_interp_and_clean.npy')[0]
    #rosoutput = np.load(dirinROS+'/maps_fire_diag_res.npy')
    #rosoutput = rosoutput.view(np.recarray)
    #ros = rosoutput.ros     


    #compute residence time
    ###################
    if (ijc != None) | (flag_restart) | (not(os.path.isfile('{:s}/resi.npy'.format(dir_out)))): 

        print('compute residence time ...   ')
        
        resi = np.zeros([nx,ny],dtype=np.dtype([('resi',float),('resi_arr',float)]))
        resi = resi.view(np.recarray)
        resi.resi = -999
        resi.resi_arr = -999
        nbre_skip = 0
        step = 1
        if mode == 'lwir': 
            minbtTest = 340
            maxbtTest = 400
            maxbtTest_hightT = 500
            minbtOK = 290 
        elif mode == 'mwir': 
            minbtTest = 550
            maxbtTest = 600
            maxbtTest_hightT = 650 
            minbtOK = 475

        #for Testing pt that were not set in previous run 
        if flag_TestMissingPt_only == True:
            resi = np.load('{:s}/resi.npy'.format(dir_out))
            resi = resi.view(np.recarray)

        N = np.where(grid.mask==2)[0].shape[0]/step
        args_here = []
        for iii, ij in enumerate(list(zip(*np.where(grid.mask==2)))[::step]):
            if resi.resi[ij[0],ij[1]]<0: 
                args_here.append([ij])

        #flag_parallel = False
        if flag_parallel: 
            cpus= 22 
            pool = multiprocessing.Pool(processes=cpus)

            # then the map method of pool actually does the parallelisation  
            results = pool.map(star_get_idx_bANDe, args_here) 
            pool.close()
            pool.join()

        else:
            results = []
            for iii, arg in enumerate(args_here):
                results.append(star_get_idx_bANDe(arg))
                
                nbre_skip_ = results[-1][3]
                ss = results[-1][2]
                nbre_skip += nbre_skip_
                if ss == None: continue
                print('{:5.2f} | {:5.2f} | ss={:.3e}'.format(100.*iii/N, 100.*nbre_skip/N, ss),end='\r')
                sys.stdout.flush()
            

        for arg, result in zip(args_here,results): 
            ij = arg[0]
            arriv, resid, ss, nbre_skip_ =  result 
             
            if arriv != None:
                resi.resi_arr[ij[0],ij[1]] = arriv
                resi.resi[ij[0],ij[1]] = resid


        if ijc == None:
            np.save('{:s}/resi'.format(dir_out),resi)
    
    else:
        resi = np.load('{:s}/resi.npy'.format(dir_out))
        resi = resi.view(np.recarray)
