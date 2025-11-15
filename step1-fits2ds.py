#%%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from astropy.io import fits as fits
from pathlib import Path
#%% FUNCTIONS
def convert_fits2ds_exp_y_x(filelist:list[str]|list[Path], attrsnote:str= '') -> xr.Dataset:
    """Converts raw Fits files into xr.Dataset with shape (exp,y,x)
    Args:
        filelist (list[str] | list[Path]): File paths to raw fits files
        attrsnote (str, optional): Note to add as an attribute to the Dataset. Defaults to ''.

    Returns:
        xr.Dataset: Dataset with shape (exp,y,x)
    """
    if isinstance(filelist[0], str):filelist = [Path(f) for f in filelist]
    exps = np.sort(np.unique([get_exp_from_fn_ASIIMG(f) for f in filelist]))
    exposures = [] # in seconds for ds
    imgs = []
    for e in exps:
        fns = [f for f in filelist if f'_{e}s' in str(f.stem)] #get files with exposure e, #type: ignore
        if len(fns)>0:
            im = []
            for fidx,f in enumerate(fns):
                with fits.open(f) as hdul:
                    hidx = int(len(hdul)-1)
                    data = hdul[hidx].data #type: ignore
                    if fidx==0:
                        exposures.append(hdul[hidx].header['EXPOSURE']) #type: ignore
                    im.append(data)
            imgs.append(np.nanmean(np.array(im), axis=0)) #average all images for same exposure
    
    ds = xr.Dataset(
        data_vars={
            'counts': (('exp','y','x'), np.array(imgs),
                        {'units':'ADU','description':'Image counts'}),
        },    coords={ 
            'exp': (('exp',), np.array(exposures), {'units':'s','description':'Exposure time for images'}),
            'y': (('y',), np.arange(np.shape(imgs)[1]), {'units':'pixels','description':'y pixel index'}), #type: ignore
            'x': (('x',), np.arange(np.shape(imgs)[2]), {'units':'pixels','description':'x pixel index'}), #type: ignore
        },    attrs={
            'description':'HiT&MIS claibration images taken with various exposure times',
            'source': sourcename,
            'slit_size_um': slitsize,
            'date_created': np.datetime_as_string(np.datetime64('now'), unit='s', timezone='UTC'),
            'Note': attrsnote}
        ) # type: ignore
    return ds

def fit_linear_counts_vs_exp(ds: xr.Dataset) -> xr.Dataset:
    """Fits a 1st order polynomial to Counts vs Exposure (y = mx + b)
       Where m = countrate (ADU/s) 
             b = bias (ADU)

    Args:
        ds (xr.Dataset): Dataset with 'counts' variable and 'exp' coordinate. 'counts' variable can have any number of additional dimensions (e.g. pos,y,x, etc). 'exp' must be one of the dimensions of 'counts'.

    Returns:
        xr.Dataset: Input Dataset with added variables:
            - countrate (ADU/s): slope of linear fit to counts vs exp for each pixel
            - countrate_err (ADU/s): standard error in countrate from fit
            - bias (ADU): intercept of linear fit to counts vs exp for each pixel
            - bias_err (ADU): standard error in bias from fit  
    """  
    weight = np.full_like(ds.exp.data, 6) # read noise of 6 counts for full exposure  
    fit = ds.counts.polyfit(dim= 'exp', deg = 1, skipna= True, w = 1/weight, cov =True)
    fit.polyfit_coefficients.data = fit.polyfit_coefficients.clip(min=0)
    # assign to ds
    ds = ds.assign(countrate = fit.polyfit_coefficients.sel(degree = 1))
    ds.countrate.attrs['units'] = 'ADU/s'
    ds.countrate.attrs['description'] = 'Count rate per pixel from linear fit to counts vs exposure time for each pixel'
    ds = ds.assign(countrate_err =np.sqrt(fit.polyfit_covariance.sel(cov_i=0, cov_j=0)))
    ds.countrate_err.attrs['units'] = 'ADU/s'
    ds.countrate_err.attrs['description'] = 'Standard Error in count rate per pixel from linear fit to counts vs exposure time for each pixel'
    ds = ds.assign(bias = fit.polyfit_coefficients.sel(degree = 0))
    ds.bias.attrs['units'] = 'ADU'
    ds.bias.attrs['description'] = 'Bias (intercept) per pixel from linear fit to counts vs exposure time for each pixel'
    ds = ds.assign(bias_err = np.sqrt(fit.polyfit_covariance.sel(cov_i=1, cov_j=1)))
    ds.bias_err.attrs['units'] = 'ADU'
    ds.bias_err.attrs['description'] = 'Standard Error in bias (intercept) per pixel from linear fit to counts vs exposure time for each pixel'
    ds = ds.drop_vars(['degree'])
    return ds

def get_exp_from_fn_comic(fn:str|Path)->str: #the file naming convention from COMIC software
    if isinstance(fn, str):
        fn = Path(fn)
    return fn.stem.split('_')[-2].strip('s')

def get_exp_from_fn_ASIIMG(fn:str|Path)->str: # The file naming convention for ASIIMG software
    if isinstance(fn, str):
        fn = Path(fn)
    return fn.stem.split('_')[2].strip('sec')
#%% 
# ##################  USER INPUTS  #########################
datdir = Path('data/raw') #input raw data dir
destdir = Path('data/l0') #input destination dir
slitsize = 100 #um
sourcename = 'Gamma Scientific RS-12D Series Calibration Light Source'
OVERWRITE = True
pos_identifier = f'pos'
############################################################

#%%
destdir.mkdir(parents=True, exist_ok=True)
datadir = list(datdir.glob(f'*{slitsize}um*_2025*'))[0]

dirs = list(datadir.iterdir())
dirs.sort()
allfiles = list(datadir.glob('*/*/*.fit*'))
print(f"Datadir: {datadir} \n # of subdirs: {len(dirs)} \n # of total files: {len(allfiles)}")

#%%
attrsnote= ''
SKIP = False
LINFIT = False
SAVEDS = False
lampdirs = []
bgds = None
for d in dirs:
    id = d.stem
    spath = destdir.joinpath(f'{datadir.stem}_{id}_l0.nc')#save path
    if 'dark' in id.lower():
        #fits -> ds, save to netcdf
        attrsnote= f'{id} dark frames taken at various exposure times'
        LINFIT = True
        SAVEDS = True
    if 'flat' in id.lower():
        #fits -> ds, linear fit, save to netcdf
        attrsnote = f'The {id} are to be used for creating calibration maps for HiT&MIS images.'
        LINFIT = True
        SAVEDS = True
    if 'background' in id.lower():
        #fits -> ds, linear fit, save to netcdf
        attrsnote = f'The {id} is to be used only ask a dark+background correction for the calibration lamp images'
        LINFIT = True
        SAVEDS = False
    if pos_identifier in id.lower():
        #fits -> ds, linear fit, save to netcdf
        lampdirs.append(d)
    if SAVEDS and spath.exists() and not OVERWRITE:
        print(f"{id} dataset already exists at {spath}, skipping processing...")
        SKIP = True
    
    if not SKIP:
        if 'dark' in id.lower() or 'flat' in id.lower() or 'background' in id.lower():
            print(f'Processing {id} data...') #fits -> ds
            all_fns = list(d.glob('*.fit*')) #try getting files in current dir
            if len(all_fns)>0: # found files in current dir
                pass #go onto the next step
            else: 
                all_fns = list(d.glob('*/*.fit*')) #try getting files in subdirs
            if len(all_fns)<1: #if no files found in dir and subdirs, skip this id
                print(f'No fits files found in {d}, skipping...')
                SKIP = True
        
            if not SKIP:
                ds = convert_fits2ds_exp_y_x(all_fns, attrsnote= attrsnote)
                if LINFIT:
                    ds = fit_linear_counts_vs_exp(ds)
                if SAVEDS:
                    encoding = {var: {'zlib': True} for var in (*ds.data_vars.keys(), *ds.coords.keys())}
                    spath = destdir.joinpath(f'{datadir.stem}_{id}_l0.nc')#save path
                    ds.to_netcdf(spath, encoding= encoding)
                    print(f"{id} dataset saved to netcdf with shape: {dict(ds.sizes)}")
                else:
                    bgds = ds
                del ds
        else: ''
#%%
if len(lampdirs)>0:
    LINFIT = True
    SAVEDS = True
    SKIP = False
    print(f'Processing lamp position data...') #fits-> ds, linear fit, save to netcdf
    slits = np.sort(np.unique([d.stem.split('_')[-2] for d in lampdirs]))
    for s in slits:
        dslist = []
        pos = []
        pdirs = list(datadir.glob(f'*{s}_*'))
        for pd in pdirs:
            id = pd.stem
            pos.append(id.split('_')[-1].strip('in'))
            all_fns = list(pd.glob('*.fit*')) #try getting files in current dir
            if len(all_fns)>0: # found files in current dir
                pass #go onto the next step
            else: 
                all_fns = list(pd.glob('*/*.fit*')) #try getting files in subdirs
        
            if len(all_fns)<1: #if no files found in dir and subdirs, skip this id
                print(f'No fits files found in {pd}, skipping...')
                SKIP = True
            bgsub_note = ''
            if not SKIP:
                ds = convert_fits2ds_exp_y_x(all_fns, attrsnote= attrsnote)
                if bgds is not None:
                    ds = ds - bgds
                    bgsub_note = ' Background subtracted before linear fitting.'
                else: 
                    bgsub_note = 'No background corrections applied.'
                dslist.append(ds)
            
            if len(dslist)>0:
                fullds = xr.concat(dslist, dim= xr.DataArray(np.array(pos, dtype= float), dims= 'pos', 
                                                    attrs={'units':'inches','description':'distance between foreoptic and lamp'}))
                fullds.attrs['description'] = f'Calibration lamp images taken at various exposures times using Hit&MIS.'
                fullds.attrs['slit_size_um'] = slitsize
                fullds.attrs['date_created'] = np.datetime_as_string(np.datetime64('now'), unit='s',timezone='UTC')
                fullds.attrs['source'] = sourcename
                fullds.counts.attrs['units'] = 'ADU'
                fullds.counts.attrs['description'] = 'Image counts'
                fullds.attrs['Note'] = bgsub_note
                print(f"Lamp dataset with {s} slit created with shape: {dict(fullds.sizes)}")
                if LINFIT:
                    fullds = fit_linear_counts_vs_exp(fullds)
                if SAVEDS:
                    encoding = {var: {'zlib': True} for var in (*fullds.data_vars.keys(), *fullds.coords.keys())}
                    spath = destdir.joinpath(f'{datadir.stem}_slit_{s}_l0.nc')#save path
                    fullds.to_netcdf(spath, encoding= encoding)
                    print(f"{id} dataset saved to netcdf with shape: {dict(fullds.sizes)}")
                del fullds
else: print('No lamp position directories found, skipping lamp data processing.')


        
#%%\
# tds = xr.open_dataset('data/l0/hmsao_calibration_slit_100um_20251013_slit_br_l0.nc')
