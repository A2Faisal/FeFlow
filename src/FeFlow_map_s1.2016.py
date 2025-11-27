import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from FlowNetwork import FlowNetwork

raw_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'raw/')
output_path = os.path.join(os.path.dirname(os.getcwd()),'data', 'output/')
plot_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'plot/')
netcdf_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'netcdf/')
atlas_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'atlas/')

verbose = True

# Initialize the flow network with NetCDF data and variable names
grid_data_path = netcdf_path + 'L.T.iron_flows.2008-2016_adj.a.nc'
json_path = output_path + 'grouped_region.json'
bilateral_csv_path = output_path + 'iron_io_stage_1_adj.csv'
trade_tariff_path = raw_path + 'tariffsPairs_88_21_vbeta1-2024-12.csv'

bilateral_dfs = []
bilateral_metrics = []
regional_metrics = []
year_list = list(range(2016, 2017, 1))

for year in year_list:
    fn = FlowNetwork(grid_data_path, 'source_1_adj', 'sink_1_adj', time=year)
    fn.gravity_model(distance='tariff', threshold_percentile=100, trade_tariff_path=trade_tariff_path, year=year, tariff_weight_factor=2, verbose=True)
    fn.ipf_flows(max_iters=100, tol=1e-6, verbose=verbose)
    fn.plot_network_map(title="Fe Extraction to Production", radius=3, color="orangered", bins=[1, 15, 30, 60, 100, np.inf], labels=['<15', '15–30', '30–60', '60–100', '>100'], coverage=0.8, edge_alpha=0.3, edge_cmap="magma", steps=20, output_dir=plot_path, filename=f"network_map_s1_{year}_cp8_ap3_orangered.png", central_longitude=163)