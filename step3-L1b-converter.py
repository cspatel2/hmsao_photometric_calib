# perform secondary straightening on a l1a dataset based on a provided line profile
#%%
from datetime import datetime
import numpy as np
import xarray as xr
from skimage import transform
from pathlib import Path
from secondary_straightening.secondary_straightening import secondary_straightening
from matplotlib import pyplot as plt
from tqdm import tqdm
# %%

# %%
######### user inputs #########
win = '5577'
datdir = Path('../data/l1a')
destdir = '' #if empty string, will replace 'l1a' with 'l1b' in datdir path. If None, will use './l1b' as destdir.
line_profile_path = Path(f'secondary_straightening/line_profile_{win}.nc')


##############################
if destdir is None:
    destdir = Path('./l1b')
elif destdir == '' :
    destdir = Path(str(datdir).replace('l1a','l1b'))
destdir.mkdir(exist_ok=True)
print(f"Destination directory: {destdir}")
#%%
fns = list(datdir.glob(f'**/*{win}*.nc'))
print(f"Found {len(fns)} files to process.")
# %%
if line_profile_path.exists():
    lprof = xr.open_dataset(line_profile_path)
else:
    raise FileNotFoundError(f"Line profile file {line_profile_path} not found.")


ds = xr.open_dataset(fns[0])
if 'countrate' in list(ds.data_vars): id = 'countrate'
elif 'intensity' in list(ds.data_vars): id = 'intensity'
else: print("no known data variable found")

for fn in fns:
    print(f"Processing file: {fn.name}...")
    ds = xr.open_dataset(fn)
    ss = secondary_straightening(ds, lprof) 
    ss.tstamp.attrs = ds.tstamp.attrs
    ss.wavelength.attrs = ds.wavelength.attrs
    ss.za.attrs = ds.za.attrs
    ss.attrs['DataProcessingLevel'] = 'L1b - Secondary Straightened'
    ss.attrs['FileCreationDate'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT")
    encoding = {var: {'zlib': True}
                    for var in (*ds.data_vars.keys(), *ds.coords.keys())}
    print('...saving...')
    outfn = destdir.joinpath(fn.stem.replace('l1a','l1b') + fn.suffix)
    ss.to_netcdf(outfn, encoding=encoding)
    print('...Done.')
#
