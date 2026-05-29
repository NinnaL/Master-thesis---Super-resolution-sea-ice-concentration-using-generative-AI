from pathlib import Path
from pyresample import geometry
import warnings

from requests import get
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
import h5py
import logging
import satpy
import numpy as np
import xarray as xr
import numpy as np
from osgeo import ogr, osr
from pyresample import kd_tree, utils
from pyresample.geometry import SwathDefinition
from datetime import datetime, timedelta
import pandas as pd

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
from AMSR2Resampler import AMSR2Resampler
from grid_config import target_area

data_dir = Path('/dmidata/projects/asip-cms/amsr2/2021/02/04/')
output_dir = Path('/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/Data')
output_dir.mkdir(parents=True, exist_ok=True)

amsr2_files = sorted(data_dir.glob('*.h5'))
print(f"Found {len(amsr2_files)} files")

for amsr2_file in amsr2_files:
    print(f"Processing: {amsr2_file.name}")
    t0 = time.time()
    try:
        resampler = AMSR2Resampler(
            amsr2_file=str(amsr2_file),
            output_dir=str(output_dir),
            target_grid=target_area,
            hemisphere='N'
        )
        resampler.save_resampled_ds()
        print(f"  Done: {amsr2_file.name} ({time.time() - t0:.2f}s)")
    except ValueError as e:
        print(f"  Skipped (corrupted): {amsr2_file.name} — {e}")
    except Exception as e:
        logging.error(f"  Failed: {amsr2_file.name} — {e}", exc_info=True)