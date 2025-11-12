#%%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os 
from glob import glob
from astropy.io import fits as fits
from pathlib import Path
# %%
destdir = Path('data/l0') #input destination dir
destdir.mkdir(parents=True, exist_ok=True)
# HOMEDIR = os.path.expanduser("~")
# slitsize = 100 #um
# datadir = np.sort(glob(os.path.join(HOMEDIR, 'locsststor','raw',f'*{slitsize}um*_2025*')))[0]
slitsize = 100 #um
datdir = Path('data/raw')
datadir = list(datdir.glob(f'*{slitsize}um*_2025*'))[0]

# datadir = np.sort(glob(os.path.join('data/raw',f'*{slitsize}um*_2025*')))[0]
# %%
dirs = list(datadir.iterdir())
dirs.sort()
allfiles = list(datadir.glob('*/*/*.fit*'))
print(f"Datadir: {datadir} \n # of subdirs: {len(dirs)} \n # of total files: {len(allfiles)}")

# %%
def get_exp_from_fn_comic(fn:str|Path)->str: #the file naming convention from COMIC software
    if isinstance(fn, str):
        fn = Path(fn)
    return fn.stem.split('_')[-2].strip('s')

def get_exp_from_fn_ASIIMG(fn:str|Path)->str: # The file naming convention for ASIIMG software
    if isinstance(fn, str):
        fn = Path(fn)
    return fn.stem.split('_')[2].strip('sec')
    
# %%

pos_identifier = 'pos'  # either 'pos' or 'new'
lampdirs = [d for d in dirs if (pos_identifier  in str(d.stem).lower()) and ('slit' in str(d.stem).lower())]
otherdirs = [d for d in dirs if 'slit' not in str(d.stem).lower()]
# %%
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
            'source':'HiT&MIS calibration data, 2025',
            'slit_size_um': slitsize,
            'date_created': np.datetime_as_string(np.datetime64('now'), unit='s'),
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
    fit = ds.counts.polyfit(dim= 'exp', deg = 1, skipna= True, cov =True)
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

# %% DARK DATA  -
identifier = 'dark'
print(f'Processing {identifier} data...') #fits -> ds, save to netcdf
attrsnote = 'The darks are to be used as dark current correction for all Hit&MIS images.'
if identifier in  [str(d.stem) for d in dirs]:
    all_fns = list(datadir.joinpath(identifier).glob('*.fit*'))
    if len(all_fns)>0:
        pass
    else: 
        all_fns = list(datadir.joinpath(identifier).glob('*/*.fit*'))
        if len(all_fns)>0: raise Warning(f'{identifier} files found in subdirectories, moving them to main dark directory is recommended.')
        else: raise ValueError(f'No {identifier} files found')
    #create dataset of raw images
    ds = convert_fits2ds_exp_y_x(all_fns, attrsnote=attrsnote)
    ds.attrs['description'] = f'({identifier}s) for HiT&MIS taken with various exposure times'
    #fit to get countrate and bias
    ds = fit_linear_counts_vs_exp(ds)
    # save to netcdf
    encoding = {var: {'zlib': True} for var in (*ds.data_vars.keys(), *ds.coords.keys())}
    spath = destdir.joinpath(f'{datadir.stem}_{identifier}.nc')#save path
    ds.to_netcdf(spath, encoding= encoding)
    print(f"{identifier} dataset saved to netcdf with shape: {dict(ds.sizes)}")
    del ds
else: print(f'No {identifier} dir in {datadir}')

# %% FLATFIELD DATA  -
identifier = 'flatfield'
print(f'Processing {identifier} data...') #fits -> ds, linear fit, save to netcdf
attrsnote = f'The {identifier} are to be used for creating calibration maps for Hit&MIS images.'
if identifier in  [str(d.stem) for d in dirs]:
    all_fns = list(datadir.joinpath(identifier).glob('*.fit*'))
    if len(all_fns)>0:
        #create dataset of raw images
        ds = convert_fits2ds_exp_y_x(all_fns, attrsnote=attrsnote)
        ds.attrs['description'] = f'({identifier}s) for HiT&MIS taken with various exposure times'
        #fit to get countrate and bias
        ds = fit_linear_counts_vs_exp(ds)
        # save to netcdf
        encoding = {var: {'zlib': True} for var in (*ds.data_vars.keys(), *ds.coords.keys())}
        spath = destdir.joinpath(f'{datadir.stem}_{identifier}.nc')#save path
        ds.to_netcdf(spath, encoding= encoding)
        print(f"{identifier} dataset saved to netcdf with shape: {dict(ds.sizes)}")
        del ds
else: print(f'No {identifier} files')

# %% BAckground DATA  -
identifier = 'background'
print(f'Processing {identifier} data...') #fits -> ds, linear fit, save to netcdf
attrsnote = f'The {identifier} is to be used only ask a dark+background correction for the calibration lamp images'
if identifier in  [str(d.stem) for d in dirs]:
    all_fns = list(datadir.joinpath(identifier).glob('*/*.fit*'))
    if len(all_fns)>0:
        bgds = convert_fits2ds_exp_y_x(all_fns, attrsnote=attrsnote)
        bgds.attrs['description'] = f'{identifier} (darkroom/no source) for HiT&MIS claibration images taken with various exposure times'
        print(f"{identifier} dataset created with shape: {dict(bgds.sizes)}")
else: print(f'No {identifier} files')

# %% LAMP DATA  -
print('Processing lamp data...')
slits = np.sort(np.unique([d.stem.split('_')[-2] for d in lampdirs]))
for s in slits:
    identifier = f'slit_{s}'
    dslist = []
    pos = []
    pdirs = list(datadir.glob(f'*{s}_*')) # all position dirs for that slit (either 'br' or 'bl')
    for pd in pdirs:
        fns = list(pd.glob('*/*.fit*')) #all files in that postion dir
        pos.append(pd.stem.split('_')[-1].strip('in')) # position from dirname
        if len(fns)>0:
            ds = convert_fits2ds_exp_y_x(fns)
            ds = ds - bgds  # subtract background, #type: ignore
            dslist.append(ds)
        else: print(f'No lamp files for slit size {s} at position')
    if len(dslist)>0:
        fullds = xr.concat(dslist, dim= xr.DataArray(pos, dims= 'pos', 
                                                    attrs={'units':'inches','description':'distance between foreoptic and lamp'}))
        fullds.attrs['description'] = f'Lamp calibration images for HiT&MIS taken with {s} slit size at various positions and exposure times'
        fullds.attrs['Note'] = 'The background has been subtracted from these lamp images.'
        print(f"Lamp dataset with {s} slit created with shape: {dict(fullds.sizes)}")
        # linear fit to get countrate per pixel
        fullds = fit_linear_counts_vs_exp(fullds)
        # save to netcdf
        encoding = {var: {'zlib': True} for var in (*fullds.data_vars.keys(), *fullds.coords.keys())}
        spath = destdir.joinpath(f'{datadir.stem}_{identifier}.nc')#save path
        fullds.to_netcdf(spath, encoding= encoding)
        print(f"Lamp dataset with {s} slit saved to netcdf at {spath.stem}")
        del fullds
    else: print(f'No lamp files for slit size {s}')
           
