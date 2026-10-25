#from scipy.io import netcdf
import numpy as np
import sys
import xarray as xr
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import linearmodels as lm
import zarr
import functions

def main():
    era5_temp = xr.open_dataset('era5_2m_temp_2020.nc')

    #Cosine weights
    cos_weights=np.cos(np.deg2rad(era5_temp.latitude))
    cos_weights=np.tile(cos_weights/np.mean(cos_weights), len(era5_temp.time)*len(era5_temp.longitude))

    hres_temp = xr.open_dataset('hres_2m_temp_2020.nc')
    hres_truth_temp = hres_temp.isel(prediction_timedelta=0)
    graph_temp = xr.open_dataset('graphcast_2m_temp_2020.nc')
    pangu_temp = xr.open_dataset('pangu_2m_temp_2020.nc')

    print('prepping data')

    dat_temp_3= functions.prep_all(
        era5_temp['2m_temperature'],
        hres_truth_temp['2m_temperature'],
        hres_temp['2m_temperature'],
        pangu_temp['2m_temperature'],
        graph_temp['2m_temperature'],
        days_ahead=3
    )

    print('computing rmse')

    
    test_cold_99 = functions.RMSE_all(dat_temp_3, operational_comp=True, quant=0.01, greater=False)
    #test_cold_95 = functions.RMSE_all(dat_temp_3, operational_comp=True, quant=0.05, greater=False)
    #test_all = functions.RMSE_all(dat_temp_3, operational_comp=True, greater=False)
    #test_hot_99 = functions.RMSE_all(dat_temp_3, operational_comp=True, quant=0.95, greater=False)
    #test_hot_95 = functions.RMSE_all(dat_temp_3, operational_comp=True, quant=0.99, greater=False)
    print(test_cold_99)

if __name__ == "__main__":
    main()