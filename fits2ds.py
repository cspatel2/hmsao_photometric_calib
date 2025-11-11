#%%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os 
from glob import glob
from astropy.io import fits as fits
from pathlib import Path
# %%
destdir = 'data/l0' #input destination dir
os.makedirs(destdir, exist_ok=True)
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

##### START HERE TOMORROW###################
# %%
def get_exp_from_fn_comic(fn:str)->str: #the file naming convention from COMIC software
    return os.path.basename(fn).strip('.fits').split('_')[-2].strip('s')

def get_exp_from_fn_ASIIMG(fn:str)->str: # The file naming convention for ASIIMG software
    return os.path.basename(fn).strip('.fits').split('_')[2].strip('sec')
    
# %%
exps = np.sort(np.unique([get_exp_from_fn_ASIIMG(f) for f in allfiles]))

pos_identifier = 'pos'  # either 'pos' or 'new'
lampdirs = [d for d in dirs if (pos_identifier  in d.lower()) and ('slit' in d.lower())]
otherdirs = [d for d in dirs if 'slit' not in d.lower()]



# %%

# # %%
# if 'background' in dirs:
#     print('true')
#     all_fns = glob(os.path.join(datadir,'background','*/*.fit*'))
#     if len(all_fns)>0:
#         exps = np.sort(np.unique([get_exp_from_fn_ASIIMG(f) for f in all_fns]))
#         print(f"Dark frames found with exposures: {exps}")
#         exposures = [] # in seconds for ds
#         bgimgs = []
#         for e in exps:
#             fns = [f for f in all_fns if f'_{e}s' in f]
#             if len(fns)>0:
#                 imgs = []
#                 for fidx,f in enumerate(fns):
#                     with fits.open(f) as hdul:
#                         data = hdul[0].data
#                         if fidx==0: exposures.append(hdul[0].header['EXPOSURE'])
#                         imgs.append(data)
#                 bgimgs.append(np.nanmean(np.array(imgs), axis=0))
#         bgds = xr.Dataset(
#             data_vars={
#                 'counts': (('exp','y','x'), np.array(bgimgs),
#                             {'units':'ADU','description':'Dark frame counts'}),
#             },    coords={ 
#                 'exp': (('exp',), np.array(exposures), {'units':'s','description':'Exposure time for dark frames'}),
#                 'y': (('y',), np.arange(np.shape(bgimgs)[1]), {'units':'pixels','description':'y pixel index'}),
#                 'x': (('x',), np.arange(np.shape(bgimgs)[2]), {'units':'pixels','description':'x pixel index'}),
#             },    attrs={
#                 'description':'DarkRoom (background) for HiT&MIS claibration images taken with various exposure times',
#                 'source':'HiT&MIS calibration data, 2025',
#                 'date_created': np.datetime_as_string(np.datetime64('now'), unit='s'),}
#             )
#     else:
#         print('No Background data files found.')   
# else: 
#     print('No Background data files found')
#     bgds = None

#

# %%
def convert_fits2ds_exp_y_x(filelist:list[str], attrsnote:str= '') -> xr.Dataset:
    exps = np.sort(np.unique([get_exp_from_fn_ASIIMG(f) for f in filelist]))
    exposures = [] # in seconds for ds
    imgs = []
    for e in exps:
        fns = [f for f in filelist if f'_{e}s' in f]
        if len(fns)>0:
            im = []
            for fidx,f in enumerate(fns):
                with fits.open(f) as hdul:
                    hidx = int(len(hdul)-1)
                    data = hdul[hidx].data
                    if fidx==0:
                        exposures.append(hdul[hidx].header['EXPOSURE'])
                    im.append(data)
            imgs.append(np.nanmean(np.array(im), axis=0))
    
    ds = xr.Dataset(
        data_vars={
            'counts': (('exp','y','x'), np.array(imgs),
                        {'units':'ADU','description':'Image counts'}),
        },    coords={ 
            'exp': (('exp',), np.array(exposures), {'units':'s','description':'Exposure time for images'}),
            'y': (('y',), np.arange(np.shape(imgs)[1]), {'units':'pixels','description':'y pixel index'}),
            'x': (('x',), np.arange(np.shape(imgs)[2]), {'units':'pixels','description':'x pixel index'}),
        },    attrs={
            'description':'HiT&MIS claibration images taken with various exposure times',
            'source':'HiT&MIS calibration data, 2025',
            'slit_size_um': slitsize,
            'date_created': np.datetime_as_string(np.datetime64('now'), unit='s'),
            'Note': attrsnote}
        ) # type: ignore
    return ds



# %% DARK DATA  -
# fits -> ds and linear fit to get dark countrate and bias
identifier = 'dark'
attrsnote = 'The darks are to be used as dark current correction for all Hit&MIS images.'
if identifier in dirs:
    all_fns = glob(os.path.join(datadir,identifier,'*.fit*'))
    if len(all_fns)>0:
        ds = convert_fits2ds_exp_y_x(all_fns, attrsnote=attrsnote)
        ds.attrs['description'] = f'({identifier}s) for HiT&MIS taken with various exposure times'
        # print(f"{identifier} dataset created with shape: {dict(ds.sizes)}")
        # linear fit to get countrate and bias per pixel
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
        # save to netcdf
        encoding = {var: {'zlib': True} for var in (*ds.data_vars.keys(), *ds.coords.keys())}
        ds.to_netcdf(os.path.join(destdir,f'hmsao_dark_{slitsize}um.nc'), encoding= encoding)
        print(f"{identifier} dataset saved to netcdf with shape: {dict(ds.sizes)}")
        del ds
else: print(f'No {identifier} files')
# %% BAckkground DATA  -
# fits -> ds
identifier = 'background'
attrsnote = 'The background is to be used only ask a dark+background correction for the calibration lamp images'
if identifier in dirs:
    all_fns = glob(os.path.join(datadir,identifier,'*/*.fit*'))
    if len(all_fns)>0:
        bgds = convert_fits2ds_exp_y_x(all_fns, attrsnote=attrsnote)
        bgds.attrs['description'] = f'{identifier} (darkroom/no source) for HiT&MIS claibration images taken with various exposure times'
        print(f"{identifier} dataset created with shape: {dict(bgds.sizes)}")
else: print(f'No {identifier} files')

# %% LAMP DATA  -
# fits -> ds
# linear fit to get lamp countrate
slits = np.sort(np.unique([d.split('_')[-2] for d in lampdirs]))
positions = np.sort(np.unique([d.split('_')[-1].strip('in') for d in lampdirs])) # currently all the postions are at the same height but for later, there are high and low postions. which means that the foreloop is going to have to change accordingly

for s in slits[:2]:
    dslist = []
    pos = []
    for p in positions:
        fns = [f for f in allfiles if f'_{s}_{p}' in f]
        if len(fns)>0:
            ds = convert_fits2ds_exp_y_x(fns)
            dslist.append(ds)
            pos.append(float(p))
    if len(dslist)>0:
        fullds = xr.concat(dslist, dim= xr.DataArray(pos, dims= 'pos', 
                                                    attrs={'units':'inches','description':'distance between foreoptic and lamp'}))
        fullds.attrs['description'] = f'Lamp calibration images for HiT&MIS taken with {s} slit size at various positions and exposure times'
        print(f"Lamp dataset with {s} slit created with shape: {dict(fullds.sizes)}")
        # linear fit to get countrate per pixel
        fit = fullds.counts.polyfit(dim= 'exp', deg = 1, skipna= True, cov =True)
        fit.polyfit_coefficients.data = fit.polyfit_coefficients.clip(min=0)
        # assign to ds
        fullds = fullds.assign(countrate = fit.polyfit_coefficients.sel(degree = 1))
        fullds.countrate.attrs['units'] = 'ADU/s'
        fullds.countrate.attrs['description'] = 'Count rate per pixel from linear fit to counts vs exposure time for each pixel'
        fullds = fullds.assign(countrate_err =np.sqrt(fit.polyfit_covariance.sel(cov_i=0, cov_j=0)))
        fullds.countrate_err.attrs['units'] = 'ADU/s'
        fullds.countrate_err.attrs['description'] = 'Standard Error in count rate per pixel from linear fit to counts vs exposure time for each pixel'
        fullds = fullds.assign(bias = fit.polyfit_coefficients.sel(degree = 0))
        fullds.bias.attrs['units'] = 'ADU'
        fullds.bias.attrs['description'] = 'Bias (intercept) per pixel from linear fit to counts vs exposure time for each pixel'
        fullds = fullds.assign(bias_err = np.sqrt(fit.polyfit_covariance.sel(cov_i=1, cov_j=1)))
        fullds.bias_err.attrs['units'] = 'ADU'
        fullds.bias_err.attrs['description'] = 'Standard Error in bias (intercept) per pixel from linear fit to counts vs exposure time for each pixel'
        fullds = fullds.drop_vars(['degree'])
        # save to netcdf
        encoding = {var: {'zlib': True} for var in (*fullds.data_vars.keys(), *fullds.coords.keys())}
           
# %%
fullds.countrate.isel(pos=0).plot()
# %%
fns = [f for f in allfiles if f'_{s}_' in f]
pos_ids = np.sort(np.unique([os.path.basename(f).split('_')[0] for f in fns]))
# %%
len(fns)
# %%
pos_ids
# %%
fn = Path(fns[0])
# %%
fn.parts[-3].split('_')[0]
# %%
path = list(Path(datadir).glob('**/*.fit*'))
# %%
len((path))
# %%
Path.joinpath(datadir,'background','*/*.fit*')
# %%
