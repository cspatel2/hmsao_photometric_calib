#%%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os 
from glob import glob
from astropy.io import fits as fits
# %%
HOMEDIR = os.path.expanduser("~")
slitsize = 100 #um
datadir = np.sort(glob(os.path.join(HOMEDIR, 'locsststor','raw',f'*{slitsize}um*_2025*')))[0]
# %%
dirs = np.sort(os.listdir(datadir))
allfiles = glob(os.path.join(datadir,'*/*', '*.fit*'))
print(f"Datadir: {datadir} \n # of subdirs: {len(dirs)} \n # of total files: {len(allfiles)}")
# %%
def get_exp_from_fn_comic(fn:str)->str: #the file naming convention from COMIC software
    return os.path.basename(fn).strip('.fits').split('_')[-2].strip('s')

def get_exp_from_fn_ASIIMG(fn:str)->str: # The file naming convention for ASIIMG software
    return os.path.basename(fn).strip('.fits').split('_')[2].strip('sec')
    
# %%
exps = np.sort(np.unique([get_exp_from_fn_ASIIMG(f) for f in allfiles]))

lampdirs = [d for d in dirs if 'slit' in d.lower()]
otherdirs = [d for d in dirs if 'slit' not in d.lower()]



# %%
slits = np.sort(np.unique([d.split('_')[-2] for d in lampdirs]))
positions = np.sort(np.unique([d.split('_')[-1].strip('in') for d in lampdirs]))
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
        )
    return ds


# %%
identifier = 'background'
attrsnote = 'The background is to be used only ask a dark+background correction for the calibration lamp images'
if identifier in dirs:
    all_fns = glob(os.path.join(datadir,identifier,'*/*.fit*'))
    if len(all_fns)>0:
        bgds = convert_fits2ds_exp_y_x(all_fns, attrsnote=attrsnote)
        bgds.attrs['description'] = f'{identifier} (darkroom/no source) for HiT&MIS claibration images taken with various exposure times'
        print(f"{identifier} dataset created with shape: {dict(bgds.sizes)}")
else: print(f'No {identifier} files')
# %%
identifier = 'dark'
attrsnote = 'The darks are to be used as dark current correction for all Hit&MIS images.'
if identifier in dirs:
    all_fns = glob(os.path.join(datadir,identifier,'*.fit*'))
    if len(all_fns)>0:
        ds = convert_fits2ds_exp_y_x(all_fns, attrsnote=attrsnote)
        ds.attrs['description'] = f'({identifier}s) for HiT&MIS taken with various exposure times'
        print(f"{identifier} dataset created with shape: {dict(ds.sizes)}")
else: print(f'No {identifier} files')


# %%
