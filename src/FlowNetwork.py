import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from haversine import haversine
import json
import importlib.resources as pkg_resources
import os
import sesame as ssm
import networkx as nx

class FlowNetwork:
	def __init__(self, dataset, outflow_var, inflow_var, node_thresh=0, time=None, verbose=False):
		if isinstance(dataset, str):
			self.ds = xr.open_dataset(dataset)
		elif isinstance(dataset, xr.Dataset):
			self.ds = dataset
		else:
			raise TypeError("Expected a string path or an xarray.Dataset, but got {}".format(type(dataset)))
		if time:
			if isinstance(time, (int, np.integer)):
				time = str(time)
			self.ds = self.ds.sel(time=time).squeeze(drop=True)
		self.inflow_var = inflow_var
		self.outflow_var = outflow_var
		
		country_frac_path = pkg_resources.files(ssm).joinpath("data/country_fraction.1deg.2000-2023.a.nc")
		self.ds_country = xr.open_dataset(country_frac_path)

		try:
			assert(self.ds[self.outflow_var].sum().values == self.ds[self.inflow_var].sum().values)
			self.total_flow = self.ds[self.outflow_var].sum().values
		except:
			if verbose:
				print(f"Warning: Total inflow ({self.ds[self.inflow_var].sum().values}) does not equal total outflow ({self.ds[self.outflow_var].sum().values})")
			self.total_flow = (self.ds[self.outflow_var].sum().values + self.ds[self.inflow_var].sum().values) / 2

		self.node_thresh = node_thresh
		self.selected_time = time

		# Extract valid coordinates
		df = (
			self.ds
			.reset_coords(drop=True)
			.stack(points=("lat", "lon"))
			.to_dataframe()
		)
		self.df = df.dropna(subset=[self.inflow_var, self.outflow_var], how='all')

		self.in_coords = df.index[(df[self.inflow_var].notna()) & (df[self.inflow_var] > node_thresh)].tolist()
		self.out_coords = df.index[(df[self.outflow_var].notna()) & (df[self.outflow_var] > node_thresh)].tolist()

		self.coord_to_in_idx = {coord: i for i, coord in enumerate(self.in_coords)}
		self.coord_to_out_idx = {coord: i for i, coord in enumerate(self.out_coords)}
		self.in_idx_to_coord = {i: coord for coord, i in self.coord_to_in_idx.items()}
		self.out_idx_to_coord = {i: coord for coord, i in self.coord_to_out_idx.items()}

		# Ensemble contains a notion of affinity between nodes
		self.ensemble = np.zeros((len(self.out_coords), len(self.in_coords)))

		# Flow contains an instantiation of flow between nodes based on the ensemble affinities
		self.flow = np.zeros_like(self.ensemble)

		# Predicted marginal df contains the predicted imports/exports distribution of the flows in and out of each country
		self.predicted_marginal_df = None

	def _gridded_inflow_outflow(self):
		# Get coords
		out_coords_array = np.array(self.out_coords)  # (O, 2) - lat, lon
		in_coords_array = np.array(self.in_coords)    # (I, 2)

		origin_lat, origin_lon = out_coords_array[:, 0], out_coords_array[:, 1]
		destination_lat, destination_lon = in_coords_array[:, 0], in_coords_array[:, 1]

		# Flow matrix
		flow_matrix = self.flow  # shape (O, I)

		# Total outflow per origin node (sum across destinations)
		outflow = flow_matrix.sum(axis=1)  # shape (origin,)

		# Total inflow per destination node (sum across origins)
		inflow = flow_matrix.sum(axis=0)   # shape (destination,)

		# Round lat/lon to 1-degree grid centers
		origin_lat_rounded = np.round(origin_lat, 1)
		origin_lon_rounded = np.round(origin_lon, 1)
		destination_lat_rounded = np.round(destination_lat, 1)
		destination_lon_rounded = np.round(destination_lon, 1)

		# Create grid for lat/lon
		lat_vals = np.arange(-89.5, 89.5, 1)
		lon_vals = np.arange(-179.5, 179.5, 1)

		# Initialize empty 2D arrays
		outflow_grid = np.full((len(lat_vals), len(lon_vals)), np.nan)
		inflow_grid = np.full((len(lat_vals), len(lon_vals)), np.nan)

		# Map outflow to grid
		for lat, lon, val in zip(origin_lat_rounded, origin_lon_rounded, outflow):
			lat_idx = np.where(lat_vals == lat)[0]
			lon_idx = np.where(lon_vals == lon)[0]
			if lat_idx.size > 0 and lon_idx.size > 0:
				outflow_grid[lat_idx[0], lon_idx[0]] = val

		# Map inflow to grid
		for lat, lon, val in zip(destination_lat_rounded, destination_lon_rounded, inflow):
			lat_idx = np.where(lat_vals == lat)[0]
			lon_idx = np.where(lon_vals == lon)[0]
			if lat_idx.size > 0 and lon_idx.size > 0:
				inflow_grid[lat_idx[0], lon_idx[0]] = val

		# Create xarray Dataset
		ds = xr.Dataset(
			data_vars={
				"flow": (["origin", "destination"], flow_matrix),
				"outflow": (["origin"], outflow),
				"inflow": (["destination"], inflow),
			},
			coords={
				"origin_lat": ("origin", origin_lat_rounded),
				"origin_lon": ("origin", origin_lon_rounded),
				"destination_lat": ("destination", destination_lat_rounded),
				"destination_lon": ("destination", destination_lon_rounded),
			},
			attrs={
				"description": "Gravity model flow matrix with inflow/outflow totals",
				"flow_units": self.ds[self.inflow_var].attrs.get("units", "tonne per grid"),
			}
		)
		self.ds_flow = ds
		return ds

	def save_to_netcdf(self, output_dir=None, filename=None):
		ds = self._gridded_inflow_outflow()
		# Determine directory: user-provided or script directory
		save_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
		# Determine filename: use provided or default
		filename = filename or "gravity_model_output.nc"
		# Full path
		save_path = os.path.join(save_dir, filename)
		# Save dataset
		ds.to_netcdf(save_path)
		print(f"NetCDF file saved to: {save_path}")

	def mass_displacement(self, output_dir=None, filename=None):
		ds_flow = self._gridded_inflow_outflow()
		flow = ds_flow["flow"].values
		origin_lat = ds_flow["origin_lat"].values
		origin_lon = ds_flow["origin_lon"].values
		destination_lat = ds_flow["destination_lat"].values
		destination_lon = ds_flow["destination_lon"].values

		# Create 1D sorted unique lat/lon values assuming 1-degree regular grid
		lats = np.arange(-89.5, 90, 1)   # 180 rows
		lons = np.arange(-179.5, 180, 1) # 360 columns

		# Build lat-lon to index mapping
		lat_to_idx = {lat: i for i, lat in enumerate(lats)}
		lon_to_idx = {lon: i for i, lon in enumerate(lons)}

		# Initialize output array (lat x lon)
		mass_disp_grid = np.full((len(lats), len(lons)), np.nan)

		# Precompute distance matrix
		n_origin = len(origin_lat)
		n_dest = len(destination_lat)

		distance_matrix = np.zeros((n_origin, n_dest))
		for i in range(n_origin):
			coord_i = (origin_lat[i], origin_lon[i])
			for j in range(n_dest):
				coord_j = (destination_lat[j], destination_lon[j])
				distance_matrix[i, j] = haversine(coord_i, coord_j)  # in km

		# Calculate mass displacement for each origin cell
		mass_displacement = np.nansum(flow * distance_matrix, axis=1)

		# Fill the 2D grid
		for i in range(n_origin):
			lat = origin_lat[i]
			lon = origin_lon[i]
			row = lat_to_idx.get(lat)
			col = lon_to_idx.get(lon)
			if row is not None and col is not None:
				mass_disp_grid[row, col] = mass_displacement[i]

		# Create xarray.Dataset with 2D lat-lon coordinates
		ds_mass_disp = xr.Dataset(
			{
				"mass_displacement": (("lat", "lon"), mass_disp_grid)
			},
			coords={
				"lat": lats,
				"lon": lons
			},
			attrs={
				"units": "tonne-km",
				"description": "Mass displacement = sum(flow × distance) from each origin cell"
			}
		)
		
		# Determine directory: user-provided or script directory
		save_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
		# Determine filename: use provided or default
		filename = filename or "mass_displacement.nc"
		# Full path
		save_path = os.path.join(save_dir, filename)
		# Save dataset
		ds_mass_disp.to_netcdf(save_path)
		self.ds_mass_disp = ds_mass_disp
		return ds_mass_disp

	@classmethod
	def load_data(cls, input_path, verbose=False):
		"""
		Load a FlowNetwork from a compressed file.
		
		Parameters:
			input_path (str): Path to the compressed file
			verbose (bool): Whether to print progress information
			
		Returns:
			FlowNetwork: A new FlowNetwork instance with the loaded data
		"""
		import os
		import shutil
		import tempfile
		
		if verbose:
			print(f"Loading FlowNetwork data from {input_path}...")
			
		# Create temporary directory
		with tempfile.TemporaryDirectory() as temp_dir:
			# Extract zip file
			shutil.unpack_archive(input_path, temp_dir, 'zip')
			
			# Load dataset
			ds = xr.load_dataset(os.path.join(temp_dir, 'dataset.nc'))
			
			# Load metadata
			metadata = np.load(os.path.join(temp_dir, 'metadata.npy'), allow_pickle=True).item()
			
			# Create FlowNetwork instance
			flow_network = cls(
				dataset=ds,
				outflow_var=metadata['outflow_var'],
				inflow_var=metadata['inflow_var'],
				node_thresh=metadata['node_thresh'],
				time=metadata['selected_time'],
				verbose=verbose
			)
			
			# Load numpy arrays
			flow_network.ensemble = np.load(os.path.join(temp_dir, 'ensemble.npy'))
			flow_network.flow = np.load(os.path.join(temp_dir, 'flow.npy'))
			
			# Load predicted marginal dataframe if it exists
			predicted_marginal_path = os.path.join(temp_dir, 'predicted_marginal.csv')
			if os.path.exists(predicted_marginal_path):
				flow_network.predicted_marginal_df = pd.read_csv(predicted_marginal_path)
			
		if verbose:
			print("✓ Data loaded successfully")
			
		return flow_network

	## Helper Functions ##
	def _label_all_nodes(self, year):
		ts = np.datetime64(f"{year}-01-01")
		ds_slice = self.ds_country.sel(time=ts)
		da_iso3 = ds_slice.to_array(dim="ISO3")

		def _pick(lat, lon):
			return da_iso3.sel(lat=lat, lon=lon, method="nearest").idxmax(dim="ISO3").item()

		# Label nodes based on coordinates (efficient, direct)
		self.out_iso3_dict = {coord: _pick(*coord) for coord in self.out_coords}
		self.in_iso3_dict = {coord: _pick(*coord) for coord in self.in_coords}

	def _coord_to_index(self, coord, kind='in'):
		if kind == 'in':
			return self.coord_to_in_idx[coord]
		elif kind == 'out':
			return self.coord_to_out_idx[coord]

	def _index_to_coord(self, idx, kind='in'):
		if kind == 'in':
			return self.in_idx_to_coord[idx]
		elif kind == 'out':
			return self.out_idx_to_coord[idx]
				
	def _pairwise_haversine(self, in_coords, out_coords):
		"""
		Compute the pairwise haversine distances between all in_coords and out_coords.

		Parameters:
			in_coords (np.ndarray): Shape (M, 2) for M inflow coordinates
			out_coords (np.ndarray): Shape (N, 2) for N outflow coordinates

		Returns:
			distances (np.ndarray): Shape (M, N) where dist[i, j] is distance from in_coords[i] to out_coords[j]
		"""
		lat1 = np.radians(in_coords[:, 0])[:, np.newaxis]  # shape (I, 1)
		lon1 = np.radians(in_coords[:, 1])[:, np.newaxis]
		lat2 = np.radians(out_coords[:, 0])[np.newaxis, :]  # shape (1, O)
		lon2 = np.radians(out_coords[:, 1])[np.newaxis, :]

		dlat = np.abs(lat2 - lat1)  # shape (I, O)
		dlon = np.abs(lon2 - lon1)
		dlon = np.where(dlon > np.pi, dlon - 2 * np.pi, dlon)  # Correct for wraparound in longitude
    
		a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
		c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) #TODO: Fix invalid value sometimes

		return 6371 * c  # Earth's radius in km

	def _distance_tariff(self, trade_tariff_path, year, tariff_weight_factor=1.0,
						mode='exponential', alpha=1.0, a=2.0, b=10, c=0.3, verbose=False):
		self._label_all_nodes(year)

		tariff_df = pd.read_csv(trade_tariff_path)
		tariff_df_year = tariff_df[tariff_df["year"] == year].copy()
		max_tariff = tariff_df_year["tariff"].max()
		tariff_df_year["normalized_tariff"] = tariff_df_year["tariff"] / max_tariff
		tariff_dict = tariff_df_year.set_index(["iso2", "iso1"])["normalized_tariff"].to_dict()

		in_coords_array = np.array(self.in_coords)
		out_coords_array = np.array(self.out_coords)
		distances = self._pairwise_haversine(out_coords_array, in_coords_array)
		distances[distances == 0] = np.nan

		tariff_matrix = np.zeros_like(distances)
		for i, out_coord in enumerate(self.out_coords):
			exporter = self.out_iso3_dict[out_coord]
			for j, in_coord in enumerate(self.in_coords):
				importer = self.in_iso3_dict[in_coord]
				normalized_tariff = tariff_dict.get((exporter, importer), 1.0)
				tariff_matrix[i, j] = normalized_tariff

		if mode == 'linear':
			adjusted_distances = distances * (1 + tariff_weight_factor * tariff_matrix)
		elif mode == 'power':
			adjusted_distances = distances * (1 + tariff_matrix) ** alpha
			print('done!')
		elif mode == 'sigmoid':
			scale = 1 + a / (1 + np.exp(-b * (tariff_matrix - c)))
			adjusted_distances = distances * scale
		elif mode == 'piecewise':
			penalty = np.where(tariff_matrix < 0.1, 1.0,
					np.where(tariff_matrix < 0.3, 1.2, 1.6))
			adjusted_distances = distances * penalty
		else:  # default: exponential
			adjusted_distances = distances * np.exp(tariff_weight_factor * tariff_matrix)

		if verbose:
			print(f"[{mode.upper()}] Mean adjusted distance: {np.nanmean(adjusted_distances):.2f}")

		return adjusted_distances

	def _scatterplot(self, df, x_col, y_col, x_log=False, y_log=False, title="Recorded vs Modelled Trade", x_label="Recorded Trade", y_label="Modelled Trade", fontsize=12, xmax=None, ymax=None, rm_outliers=False, output_dir=None, filename=None, ax=None):
		df_copy = df.dropna(subset=[x_col, y_col, "ISO3"]).copy()

		if x_log:
			df_copy = df_copy[df_copy[x_col] > 0]
			df_copy["x_plot"] = np.log10(df_copy[x_col])
		else:
			df_copy["x_plot"] = df_copy[x_col]

		if y_log:
			df_copy = df_copy[df_copy[y_col] > 0]
			df_copy["y_plot"] = np.log10(df_copy[y_col])
		else:
			df_copy["y_plot"] = df_copy[y_col]

		if rm_outliers:
			Q1_x, Q3_x = df_copy["x_plot"].quantile([0.25, 0.75])
			Q1_y, Q3_y = df_copy["y_plot"].quantile([0.25, 0.75])
			IQR_x, IQR_y = Q3_x - Q1_x, Q3_y - Q1_y

			df_copy = df_copy[
				(df_copy["x_plot"] >= Q1_x - 1.5 * IQR_x) & (df_copy["x_plot"] <= Q3_x + 1.5 * IQR_x) &
				(df_copy["y_plot"] >= Q1_y - 1.5 * IQR_y) & (df_copy["y_plot"] <= Q3_y + 1.5 * IQR_y)
			]

		x_vals = df_copy["x_plot"].values
		y_vals = df_copy["y_plot"].values

		if len(x_vals) > 1:
			slope, intercept = np.polyfit(x_vals, y_vals, 1)
			y_pred = slope * x_vals + intercept

			# Metrics
			r2 = 1 - np.sum((y_vals - y_pred)**2) / np.sum((y_vals - np.mean(y_vals))**2)
			rmse = np.sqrt(np.mean((y_vals - x_vals)**2))
			mae = np.mean(np.abs(y_vals - x_vals))
			nrmse = rmse / np.mean(x_vals)
			mape = np.mean(np.abs((y_vals - x_vals) / x_vals)) * 100
			smape = np.mean(2 * np.abs(y_vals - x_vals) / (np.abs(x_vals) + np.abs(y_vals))) * 100
		else:
			r2 = rmse = mae = nrmse = mape = smape = np.nan

		# Compute residuals and CI
		residuals = y_vals - x_vals
		std_residual = np.std(residuals)
		ci = 1.96 * std_residual

		# Define the 1:1 line
		min_val = min(x_vals.min(), y_vals.min())
		max_val = max(x_vals.max(), y_vals.max())
		x_line = np.array([min_val, max_val])
		y_line = x_line

		# --- Plot section ---
		# Create new figure if ax not supplied
		if ax is None:
			fig, ax = plt.subplots(figsize=(7, 7))
		else:
			fig = None
		ax.scatter(x_vals, y_vals, alpha=0.6, s=25)
		ax.plot(x_line, y_line, 'b--', label='1-to-1 line')
		ax.plot(x_line, y_line + ci, 'r--', label='95% Error Band')
		ax.plot(x_line, y_line - ci, 'r--')

		# Annotate points outside the CI
		# for i in range(len(df_copy)):
		# 	if abs(residuals[i]) > ci:
		# 		ax.annotate(df_copy["ISO3"].iloc[i],
		# 					(x_vals[i], y_vals[i]),
		# 					fontsize=10,
		# 					alpha=0.8)

		# Annotate statistics
		ax.text(
			0.05, 0.95,
			f"$R^2$ = {r2:.2f}\n"
			f"RMSE = {rmse:.2f}\n"
			f"NRMSE = {nrmse:.2%}\n"
			f"MAE = {mae:.2f}\n"
			f"MAPE = {mape:.2f}%\n"
			f"sMAPE = {smape:.2f}%",
			transform=ax.transAxes,
			fontsize=fontsize,
			verticalalignment='top',
			bbox=dict(facecolor='white', alpha=0.8)
		)

		ax.legend()
		ax.set_title(title, fontsize=fontsize+2, pad=5)
		ax.set_xlabel(x_label, fontsize=fontsize)
		ax.set_ylabel(y_label, fontsize=fontsize)
		ax.tick_params(axis='x', labelsize=fontsize)
		ax.tick_params(axis='y', labelsize=fontsize)
		if xmax is not None:
			ax.set_xlim(right=xmax)
		if ymax is not None:
			ax.set_ylim(top=ymax)
		ax.set_aspect('equal', adjustable='box')
		plt.tight_layout()

		# --- Save ---
		# Save only if we created the figure here
		if fig is not None and (output_dir or filename):
			filename = filename or "model_validation.png"
			root, ext = os.path.splitext(filename)
			if not ext:
				ext = ".png"
				filename = root + ext
			save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)
			plt.savefig(save_path, dpi=600, bbox_inches='tight', format=ext.lstrip('.'))
			plt.show()
			plt.close(fig)

		# Pack up the stats
		stats = {
			"R2":     r2,
			"RMSE":   rmse,
			"NRMSE":  nrmse,
			"MAE":    mae,
			"MAPE":   mape,
			"sMAPE":  smape,
			"n_points": len(x_vals)
		}
		return stats
		
	## Algorithm Methods ##

	def gravity_model(self, distance='pairwise_haversine', threshold_percentile=100, trade_tariff_path=None, year=None, mode='exponential', tariff_weight_factor=1.0, alpha=1.0, a=2.0, b=10, c=0.3, verbose=False):
		if verbose:
			if threshold_percentile < 100:
				print(f"Running Gravity Model with thresholding at {threshold_percentile}% of possible edges...")
			else:
				print("Running Gravity Model...")
		self.ensemble[:] = 0  # Reset

		inflow_values = self.df.loc[self.in_coords][self.inflow_var].values  # shape (I,)
		outflow_values = self.df.loc[self.out_coords][self.outflow_var].values  # shape (O,)

		in_coords_array = np.array(self.in_coords)  # shape (I, 2)
		out_coords_array = np.array(self.out_coords)  # shape (O, 2)

		if distance == 'pairwise_haversine':
			distances = self._pairwise_haversine(out_coords_array, in_coords_array)  # shape (O, I)
			distances[distances == 0] = np.nan  # prevent divide-by-zero
		elif distance == 'tariff':
			distances = self._distance_tariff(trade_tariff_path, year, tariff_weight_factor, mode, alpha, a, b, c, verbose)

		# Calculate gravity matrix g = G * m_o * m_i / d
		gravity_matrix = np.outer(outflow_values, inflow_values) / distances  # shape (O, I)
		gravity_matrix = np.nan_to_num(gravity_matrix)  # replace nan with 0
		if verbose:
			print(f"Edges after gravity calculation: {np.sum(gravity_matrix > 0)}")

		# Apply threshold if specified
		if threshold_percentile < 100:
			# Step 1: For each node, keep its strongest connections
			thresholded_gravity_matrix = np.zeros_like(gravity_matrix)
			
			# For each outflow node, keep its strongest connections
			for i in range(len(self.out_coords)):
				row = gravity_matrix[i, :]
				if np.any(row > 0):
					threshold = np.percentile(row[row > 0], 100 - threshold_percentile)
					thresholded_gravity_matrix[i, :] = np.where(row >= threshold, row, 0)
			
			# For each inflow node, keep its strongest connections
			for j in range(len(self.in_coords)):
				col = gravity_matrix[:, j]
				if np.any(col > 0):
					threshold = np.percentile(col[col > 0], 100 - threshold_percentile)
					thresholded_gravity_matrix[:, j] = np.maximum(thresholded_gravity_matrix[:, j], 
						np.where(col >= threshold, col, 0))
			
			# Step 2: Edge Data Analysis
			total_flow_mass = np.sum(gravity_matrix)
			nonzero_flow_mass = np.sum(thresholded_gravity_matrix[thresholded_gravity_matrix > 0])
			if verbose:
				print(f"After thresholding there are {np.sum(thresholded_gravity_matrix > 0)} edges containing {nonzero_flow_mass/total_flow_mass:.1%} of total flow mass")

			# Step 3: Node Analysis
			nonzero_rows = np.any(thresholded_gravity_matrix > 0, axis=1)
			nonzero_cols = np.any(thresholded_gravity_matrix > 0, axis=0)
			num_nodes = np.sum(nonzero_rows) + np.sum(nonzero_cols)
			if verbose:
				print(f"Number of nodes after thresholding: {num_nodes}")

			# Calculate flow contained in remaining nodes
			total_inflow = np.sum(inflow_values)
			total_outflow = np.sum(outflow_values)
			inflow_contained = np.sum(inflow_values[nonzero_cols])
			outflow_contained = np.sum(outflow_values[nonzero_rows])
			if verbose:
				print(f"Percent of inflow contained in remaining nodes: {inflow_contained/total_inflow:.1%}")
				print(f"Percent of outflow contained in remaining nodes: {outflow_contained/total_outflow:.1%}")

			gravity_matrix = thresholded_gravity_matrix

		self.ensemble = gravity_matrix
		if verbose:
			print("Gravity Model Complete. Ensemble Generated.")

	def ipf_flows(self, max_iters=100, tol=1e-6, verbose=False):
		"""
		Apply Iterative Proportional Fitting to match row and column sums.
		This method preserves the relative weights of the non-zero entries while
		matching the target row and column sums.
		"""
		if verbose:
			print("Running IPF...")
			print(f"Initial edges in ensemble: {np.sum(self.ensemble > 0)}")
		
		# Get target row and column sums
		target_row_sums = self.df.loc[self.out_coords][self.outflow_var].values
		target_col_sums = self.df.loc[self.in_coords][self.inflow_var].values
		
		# Initialize working matrix
		W = self.ensemble.copy()
		
		# Ensure we have non-zero entries for IPF
		if np.sum(W > 0) == 0:
			raise ValueError("No non-zero entries in ensemble matrix. IPF cannot proceed.")
		
		for iteration in range(max_iters):
			# Row scaling
			row_sums = W.sum(axis=1)
			row_sums[row_sums == 0] = 1  # prevent divide by zero
			W = W * (target_row_sums[:, np.newaxis] / row_sums[:, np.newaxis])
			
			# Column scaling
			col_sums = W.sum(axis=0)
			col_sums[col_sums == 0] = 1  # prevent divide by zero
			W = W * (target_col_sums[np.newaxis, :] / col_sums[np.newaxis, :])
			
			# Check convergence
			row_error = np.max(np.abs(W.sum(axis=1) - target_row_sums))
			col_error = np.max(np.abs(W.sum(axis=0) - target_col_sums))
			
			if max(row_error, col_error) < tol:
				if verbose:
					print(f"IPF converged in {iteration + 1} iterations")
				break
		
		self.flow = W
		if verbose:
			print(f"IPF complete. Flow matrix generated with {np.sum(self.flow > 0)} edges.")

	### Summarize and validate model #####
	def _get_node_iso3s(self, year):
		# slice once
		ts = np.datetime64(f"{year}-01-01")
		ds_slice = self.ds_country.sel(time=ts)
		# stack once
		da_iso3 = ds_slice.to_array(dim="ISO3")

		def _pick(lat, lon):
			return da_iso3.sel(lat=lat, lon=lon, method="nearest")\
						.idxmax(dim="ISO3").item()

		exp_iso3 = [ _pick(lat, lon) for lat, lon in self.out_coords]
		imp_iso3 = [ _pick(lat, lon) for lat, lon in self.in_coords]
		
		# find all non-zero entries in the flow matrix
		out_idx, in_idx = np.nonzero(self.flow)
		# assemble records
		records = []
		for o, i in zip(out_idx, in_idx):
			records.append({
				"exporter": exp_iso3[o],
				"importer": imp_iso3[i],
				"flow":     self.flow[o, i],
				"year":     year
			})

		# 4) return the edge‐list DataFrame
		return pd.DataFrame(records)

	def _group_iso3(self, json_path, trade_df):
		with open(json_path, 'r') as f:
			country_to_grouped_region = json.load(f)

			# Map exporters and importers to their grouped‐region codes
			trade_df['exp_ISO3'] = trade_df['exporter'].map(country_to_grouped_region)
			trade_df['imp_ISO3'] = trade_df['importer'].map(country_to_grouped_region)

			# Now aggregate total_flow by (exp_region, imp_region, year)
			grouped_df = (
				trade_df
				.groupby(['exp_ISO3', 'imp_ISO3'], as_index=False)['total_flow']
				.sum()
				.rename(columns={'total_flow': 'tonnes'})
			)
			return grouped_df

	def bilateral_flow(self, year, json_path=None):
		edge_df = self._get_node_iso3s(year)
		trade_df = (edge_df.groupby(["exporter", "importer", "year"], as_index=False)["flow"].sum().rename(columns={"flow": "total_flow"}))
		if json_path:
			trade_df = self._group_iso3(json_path, trade_df)
			trade_df["year"] = year
		self.bilateral_df = trade_df.copy()
		return self.bilateral_df

	def validate_bilateral(self, bilateral_csv_path, year, column='tonnes', x_log=False, y_log=False, rm_outliers=False, title="Recorded vs Modelled Trade", x_label="Recorded", y_label="Modelled", xmax=None, ymax=None, fontsize=12):
		raw_df = pd.read_csv(bilateral_csv_path)
		raw_df = raw_df[raw_df["year"] == year]
		self.year = year

		pred_df = self.bilateral_df
		# 1) filter to the correct year
		raw_y  = raw_df[raw_df.year == year ].copy().rename(columns={column:'raw_tonnes'})
		pred_y = pred_df[pred_df.year == year].copy().rename(columns={'tonnes':'pred_tonnes'})

		# 2) merge on exp_ISO3, imp_ISO3, year
		val_df = pd.merge(
			raw_y, pred_y,
			on=['exp_ISO3','imp_ISO3','year'],
			how='inner'
		)
		# 3) label for annotation
		val_df['ISO3'] = val_df['exp_ISO3'] + '→' + val_df['imp_ISO3']
		self.val_df = val_df
		# 5) plot
		metrics = self._scatterplot(
			val_df,
			x_col='raw_tonnes',
			y_col='pred_tonnes',
			x_log=x_log,
			y_log=y_log,
			rm_outliers=rm_outliers,
			title=title, 
			x_label=x_label, 
			y_label=y_label,
			xmax=xmax, 
			ymax=ymax,
			fontsize=fontsize
		)
		df_metrics = pd.DataFrame([metrics])
		df_metrics['Year'] = year
		self.df_metrics = df_metrics
		return val_df, df_metrics

	def regional_trade_comparison(self, x_log=False, y_log=False, title="Recorded vs Modelled Trade", x_label="Recorded Trade", y_label="Modelled Trade", fontsize=12, xmax=None, ymax=None, rm_outliers=False, output_dir=None, filename=None):
			df = self.val_df
			# Total exports per country-year (including domestic)
			exports_raw = df.groupby(["exp_ISO3", "year"], as_index=False)["raw_tonnes"].sum().rename(columns={"tonnes": "raw_exports"})

			# Total imports per country-year (including domestic)
			imports_raw = df.groupby(["imp_ISO3", "year"], as_index=False)["raw_tonnes"].sum().rename(columns={"tonnes": "raw_imports"})

			exports_pred = df.groupby(["exp_ISO3", "year"], as_index=False)["pred_tonnes"].sum().rename(columns={"tonnes": "pred_exports"})

			# Total imports per country-year (including domestic)
			imports_pred = df.groupby(["imp_ISO3", "year"], as_index=False)["pred_tonnes"].sum().rename(columns={"tonnes": "pred_imports"})

			# Prepare the data
			df_exports = exports_raw.merge(exports_pred, on=["exp_ISO3", "year"])
			df_exports = df_exports.rename(columns={"exp_ISO3": "ISO3"})

			df_imports = imports_raw.merge(imports_pred, on=["imp_ISO3", "year"])
			df_imports = df_imports.rename(columns={"imp_ISO3": "ISO3"})

			# Create a single figure with two panels
			fig, axes = plt.subplots(1, 2, figsize=(14, 7))

			# Exports plot
			df_metrics_ex = self._scatterplot(
					df=df_exports,
					x_col="raw_tonnes", y_col="pred_tonnes",
					x_log=x_log, y_log=y_log,
					title=f"Exports: {title}",
					x_label=x_label, y_label=y_label,
					fontsize=fontsize, xmax=xmax, ymax=ymax,
					rm_outliers=rm_outliers,
					ax=axes[0]  # 1st panel
			)
			df_metrics_ex = pd.DataFrame([df_metrics_ex])
			df_metrics_ex['Trade'] = "exports"
			# Imports plot
			df_metrics_im = self._scatterplot(
					df=df_imports,
					x_col="raw_tonnes", y_col="pred_tonnes",
					x_log=x_log, y_log=y_log,
					title=f"Imports: {title}",
					x_label=x_label, y_label=y_label,
					fontsize=fontsize, xmax=xmax, ymax=xmax,
					rm_outliers=rm_outliers,
					ax=axes[1]  # 2nd panel
			)
			df_metrics_im = pd.DataFrame([df_metrics_im])
			df_metrics_im['Trade'] = "imports"
			df_comb = pd.concat([df_metrics_ex, df_metrics_im])
			df_comb["year"] = self.year
			plt.tight_layout()

			# Save combined figure
			if output_dir or filename:
					# Default filename if none provided
					filename = filename or "combined_trade_validation.png"
					
					# Check if the filename includes an extension
					root, ext = os.path.splitext(filename)
					if not ext:
							ext = ".png"  # default to PNG if no extension provided
							filename = root + ext

					# Construct full save path
					save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)

					# Save the plot with appropriate format
					plt.savefig(save_path, dpi=600, bbox_inches='tight', format=ext.lstrip('.'))
			plt.show()
			plt.close(fig)
			return df_comb

	### Extra
	def plot_isci(self,
				isci_col='ISCI', 
				country_col='ISO3',
				title="Iron System Centrality Index",
				xlabel='Eigenvector Centrality',
				figsize=(8, 10),
				dpi=300,
				font_size=10,
				bar_color='Orange'):

		# sort so largest is at top
		df = self.isci_df
		df_sorted = df.sort_values(isci_col, ascending=False)
		
		fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
		ax.barh(df_sorted[country_col], df_sorted[isci_col], color=bar_color, edgecolor='none')
		ax.invert_yaxis()
		
		# labels & title
		ax.set_xlabel(xlabel, fontsize=font_size+2)
		ax.set_ylabel('')
		ax.set_title(title, fontsize=font_size+4, pad=15)
		
		# grid styling
		ax.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
		ax.set_axisbelow(True)
		
		# remove default margins
		ax.margins(y=0.01)
		
		# tick styling
		ax.tick_params(axis='x', labelsize=font_size)
		ax.tick_params(axis='y', labelsize=font_size)
		
		fig.tight_layout()
		return fig, ax

	def compute_ISCI(self, method='eigenvector', plot=False):
		"""
		Build a country-level directed graph from self.bilateral_df (for the given year)
		and compute the Iron System Centrality Index (ISCI) via the specified
		network-centrality method.
		
		method: one of 'eigenvector', 'pagerank', 'betweenness', 'closeness'
		"""
		df = self.bilateral_df

		# build graph
		G = nx.DiGraph()
		for _, row in df.iterrows():
			G.add_edge(row['exp_ISO3'], row['imp_ISO3'],
						weight=row['tonnes'])

		# choose centrality
		if method == 'eigenvector':
			C = nx.eigenvector_centrality_numpy(G, weight='weight')
		elif method == 'pagerank':
			C = nx.pagerank(G, weight='weight')
		elif method == 'betweenness':
			C = nx.betweenness_centrality(G, weight='weight')
		elif method == 'closeness':
			C = nx.closeness_centrality(G, distance=lambda u, v, d: 1.0/d.get('weight',1.0))
		else:
			raise ValueError(f"Unknown centrality method {method}")

		# record as DataFrame on the instance
		self.isci_df = (
			pd.DataFrame.from_dict(C, orient='index', columns=['ISCI'])
				.reset_index()
				.rename(columns={'index':'ISO3'})
		)
		if plot:
			# no extra `self` positional argument!
			self.plot_isci(
				isci_col='ISCI',
				country_col='ISO3',
				title="Iron System Centrality Index",
				xlabel=f"{method.capitalize()} Centrality",
				figsize=(8, 10),
				dpi=300,
				font_size=12,
				bar_color='orange'
			)

		return self.isci_df.sort_values('ISCI', ascending=False)

	def compute_cell_centrality(self,
								method='eigenvector',
								as_xarray=False,
								max_iter=500,
								tol=1e-6,
								fill_value=np.nan):
		"""
		Compute grid‐cell centrality (ISCI) on the directed cell‐graph.
		
		Parameters
		----------
		method : str
			One of 'eigenvector','pagerank','betweenness','closeness'.
		as_xarray : bool
			If False (default), returns a DataFrame with columns
			['lat','lon','centrality'] for each cell that has flow.
			If True, returns an xarray.Dataset with dims (lat, lon)
			matching self.ds, filling non‐active cells with fill_value.
		max_iter, tol : for the power‐method eigenvector solver.
		fill_value : value for inactive cells in the xarray output.
		
		Returns
		-------
		pandas.DataFrame or xarray.Dataset
		"""
		# 1) Build directed graph of nonzero flows between cells
		G = nx.DiGraph()
		O, I = self.flow.shape
		for oi, ji in zip(*np.nonzero(self.flow)):
			u = tuple(self.out_coords[oi])
			v = tuple(self.in_coords[ji])
			G.add_edge(u, v, weight=self.flow[oi, ji])

		# 2) Compute centrality
		if method == 'eigenvector':
			C = nx.eigenvector_centrality(
				G, weight='weight', max_iter=max_iter, tol=tol
			)
		elif method == 'pagerank':
			C = nx.pagerank(G, weight='weight')
		elif method == 'betweenness':
			C = nx.betweenness_centrality(G, weight='weight')
		elif method == 'closeness':
			C = nx.closeness_centrality(
				G, distance=lambda u, v, d: 1.0 / d.get('weight', 1.0)
			)
		else:
			raise ValueError(f"Unknown centrality method {method!r}")

		# 3) Unpack to DataFrame
		df = pd.DataFrame([
			{'lat': coord[0], 'lon': coord[1], 'centrality': score}
			for coord, score in C.items()
		]).sort_values('centrality', ascending=False).reset_index(drop=True)

		if not as_xarray:
			return df

		# 4) Paint to full grid
		lats = self.ds['lat'].values
		lons = self.ds['lon'].values
		arr  = np.full((lats.size, lons.size), fill_value, dtype=float)

		# nearest‐neighbor mapping from (lat,lon)→index
		for _, row in df.iterrows():
			i = np.abs(lats - row.lat).argmin()
			j = np.abs(lons - row.lon).argmin()
			arr[i, j] = row.centrality

		da = xr.DataArray(
			arr,
			coords={'lat': lats, 'lon': lons},
			dims=['lat', 'lon'],
			name=f'cell_centrality_{method}'
		)
		return xr.Dataset({da.name: da})

	### Plot map
	# Mass Displacement Plot Function
	def _mass_displacement_plot(self, title="", radius=3, color="darkblue", bins=[0.1, 15, 30, 60, 100, np.inf], labels=['0–15', '15–30', '30–60', '60–100', '>100']):
		# Prepare data
		ds = self.mass_displacement()
		lat_vals = ds['lat'].values
		lon_vals = ds['lon'].values
		mass_vals = ds['mass_displacement'].values * 1e-9  # convert to Gt-km

		lat_flat = lat_vals.repeat(len(lon_vals))
		lon_flat = np.tile(lon_vals, len(lat_vals))
		mass_flat = mass_vals.flatten()

		mask = np.isfinite(mass_flat)
		lat_flat = lat_flat[mask]
		lon_flat = lon_flat[mask]
		mass_flat = mass_flat[mask]

		# Define bins and styles
		bins = bins
		labels = labels
		sizes = [radius**(2*i) for i in range(len(labels))]
		colors = [color] * len(labels)  # same border color

		# Assign classes
		class_indices = np.digitize(mass_flat, bins) - 1

		# Plot
		fig = plt.figure(figsize=(12, 8))
		ax = plt.axes(projection=ccrs.Robinson())
		ax.set_global()
		ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
		# ax.add_feature(cfeature.BORDERS, linewidth=0.5)
		ax.spines['geo'].set_visible(False)

		# Plot each class
		for i in range(len(labels)):
			idx = class_indices == i
			if np.any(idx):
				ax.scatter(
					lon_flat[idx], lat_flat[idx],
					s=sizes[i],  # area
					facecolors='none',
					edgecolors=colors[i],
					label=labels[i],
					alpha=0.2 if i == 0 else 1,
					transform=ccrs.PlateCarree()
				)

		# Legend and title
		plt.legend(title="Mass Displacement\n(Gt-km)", title_fontsize=14, loc='center left', frameon=False, fontsize=14)
		plt.title(title, fontsize=18)
		
		return fig, ax

	# Flow Overlay Function
	def _plot_flow_network_gradient(self, ax, num_edges=1000, edge_alpha=0.6, edge_cmap="magma", steps=20):
		ds_flow = self._gridded_inflow_outflow()
		flow = ds_flow["flow"].values
		origin_coords = list(zip(ds_flow["origin_lat"].values, ds_flow["origin_lon"].values))
		destination_coords = list(zip(ds_flow["destination_lat"].values, ds_flow["destination_lon"].values))

		flow_clean = np.where(np.isfinite(flow), flow, 0)

		flat_indices = np.argpartition(flow_clean.ravel(), -num_edges)[-num_edges:]
		top_indices = flat_indices[np.argsort(flow_clean.ravel()[flat_indices])][::-1]
		out_idxs, in_idxs = np.unravel_index(top_indices, flow.shape)

		flow_vals = flow_clean[out_idxs, in_idxs]
		norm = mcolors.Normalize(vmin=flow_vals.min(), vmax=flow_vals.max())
		cmap = plt.get_cmap(edge_cmap)

		# ---- Calculate coverage of top edges ----
		total_flow = flow_clean.sum()
		top_flow_sum = flow_vals.sum()
		coverage_percent = (top_flow_sum / total_flow) * 100
		print(f"Top {num_edges:,} edges cover {coverage_percent:.2f}% of total flow.")

		min_width, max_width = 0.2, 3

		for out_i, in_j, val in zip(out_idxs, in_idxs, flow_vals):
			if val <= 0:
				continue
			lat1, lon1 = origin_coords[out_i]
			lat2, lon2 = destination_coords[in_j]
			width = min_width + (max_width - min_width) * norm(val)

			lats = np.linspace(lat1, lat2, steps)
			lons = np.linspace(lon1, lon2, steps)

			for i in range(steps - 1):
				color = cmap(i / (steps - 1))
				ax.plot(
					[lons[i], lons[i+1]], [lats[i], lats[i+1]],
					color=color,
					linewidth=width,
					alpha=edge_alpha,
					transform=ccrs.Geodetic()
				)

	# Main plotting function
	def plot_network_map(self, title="", radius=3, color="darkblue", bins=[0.1, 15, 30, 60, 100, np.inf], labels=['0–15', '15–30', '30–60', '60–100', '>100'], num_edges=1000, edge_alpha=0.6, edge_cmap="magma", steps=20, output_dir=None, filename=None):
		# Plot both on same figure
		fig, ax = self._mass_displacement_plot(
			title=title,
			radius=radius,
			color=color,
			bins=bins,
			labels = labels
		)

		self._plot_flow_network_gradient(
			ax=ax,
			num_edges=num_edges,
			edge_alpha=edge_alpha,
			edge_cmap=edge_cmap,
			steps=steps
		)

		# Save and show
		plt.tight_layout()
		if output_dir or filename:
			# Default filename if none provided
			filename = filename or "network_plot.png"
			
			# Check for file extension
			root, ext = os.path.splitext(filename)
			if not ext:
				ext = ".png"  # default to PNG if no extension provided
				filename = root + ext

			# Construct full save path
			save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)

			# Save figure using detected or default format
			plt.savefig(save_path, dpi=600, bbox_inches='tight', format=ext.lstrip('.'))
		plt.show()

	def compute_marginal_country_trade(self, out_path=None, verbose=False):
		"""
		Add estimated imports and exports to self.ds by comparing edge flows between different countries.
		Requires static_ctry_frac_ds with shape (lat, lon), one data_var per country.
		"""
		if verbose:
			print("Computing country trade from network flows...")

		df = ssm.grid_2_table(dataset=self.ds, variables=[self.inflow_var, self.outflow_var], agg_function='sum', verbose=verbose)
		#df = ssm.grid_2_table(dataset=self.ds, variables=self.outflow_var, agg_function='sum', verbose=verbose)
		#df = df.merge(df, on='ISO3')
		print(df)


		df['estimated_imports'] = np.maximum(0, df[self.inflow_var] - df[self.outflow_var])
		df['estimated_exports'] = np.maximum(0, df[self.outflow_var] - df[self.inflow_var])

		self.predicted_marginal_df = df
		if out_path:
			self.predicted_marginal_df.to_csv(out_path)
			if verbose:
				print("✓ Estimated imports and exports and saved.")
	
	def check_mass_conservation(self, tol=1e-6, verbose=False):
		"""
		Verify that the flow matrix conserves mass at each node, by comparing
		inflow and outflow values in the dataset to the sums of incoming and
		outgoing edges in the flow matrix.
		"""

		if verbose:
			print("Checking mass conservation...")

		inflow_pass, outflow_pass = True, True

		# Check inflow nodes: sum over columns (axis=1)
		for j, in_coord in enumerate(self.in_coords):
			flow_in = self.flow[:, j].sum()
			ds_in = self.df.loc[in_coord][self.inflow_var]
			if not np.isclose(flow_in, ds_in, atol=tol):
				inflow_pass = False
				print(f"[!] Inflow mismatch at {in_coord}: graph={flow_in:.3f}, dataset={ds_in:.3f}")

		# Check outflow nodes: sum over rows (axis=0)
		for i, out_coord in enumerate(self.out_coords):
			flow_out = self.flow[i, :].sum()
			ds_out = self.df.loc[out_coord][self.outflow_var]
			if not np.isclose(flow_out, ds_out, atol=tol):
				outflow_pass = False
				print(f"[!] Outflow mismatch at {out_coord}: graph={flow_out:.3f}, dataset={ds_out:.3f}")

		if inflow_pass and outflow_pass:
			if verbose:
				print("✅ All nodes pass mass conservation check.")
		else:
			print("❌ Some nodes failed the check. See above for details.")

	def analyze_degree_distribution(self, caption=None, verbose=False):
		"""
		Analyze the degree distribution of the flow network by fitting various probability distributions.
		Returns a dictionary containing the fitted parameters and statistics.
		"""
		import numpy as np
		import scipy.stats as stats
		from collections import OrderedDict
		import matplotlib.pyplot as plt

		# Calculate degree sequence from flow matrix
		degree_sequence = np.sum(self.flow > 0, axis=1)  # Out-degree for each node
		degree_sequence = degree_sequence[degree_sequence > 0]  # Remove zero-degree nodes

		# Distributions to fit
		distributions = OrderedDict({
			'normal': stats.norm,
			'lognorm': stats.lognorm,
			'expon': stats.expon,
			'powerlaw': stats.powerlaw
		})

		fitted_results = {}

		# Fit the different distributions
		for name, dist in distributions.items():
			try:
				params = dist.fit(degree_sequence)
				log_likelihood = np.sum(dist.logpdf(degree_sequence, *params))
				num_params = len(params)
				aic = 2 * num_params - 2 * log_likelihood
				fitted_results[name] = {
					'params': params,
					'log_likelihood': log_likelihood,
					'aic': aic,
					'num_params': num_params
				}
			except Exception as e:
				print(f"Could not fit {name}: {e}")

		# Plot the results
		plt.figure(figsize=(10, 6))
		
		# Create 15 equal-sized bins between min and max degree
		min_degree = min(degree_sequence)
		max_degree = max(degree_sequence)
		if verbose:
			print('Min degree:', min_degree, 'Max degree:', max_degree)
		bins = np.linspace(min_degree, max_degree, 16)  # 16 edges for 15 bins
		
		# Plot histogram with equal-sized bins
		hist, bins, _ = plt.hist(degree_sequence, bins=bins, density=False,
								alpha=0.6, label='Data', color='gray')
		
		x = np.linspace(min(degree_sequence), max(degree_sequence), 100)
		for name, res in fitted_results.items():
			if 'params' in res:
				# Scale the PDF to match the histogram counts
				y = distributions[name].pdf(x, *res['params']) * len(degree_sequence) * (bins[1] - bins[0])
				plt.plot(x, y, label=f"{name} (AIC={res['aic']:.0f})")

		if caption:
			plt.figtext(0.5, 0.01, caption, ha='center', fontsize=10)

		plt.yscale('log')
		plt.xlabel('Degree')
		plt.ylabel('Number of Nodes')
		plt.title('Degree Distribution with Fitted Models')
		plt.legend()
		plt.grid(True, which="both", ls="--", lw=0.5)
		plt.tight_layout()
		plt.show()

		# Print summary
		if verbose:
			print("\n=== Distribution Fit Summary ===")
			for name, res in fitted_results.items():
				if 'aic' in res:
					print(f"{name:16} | AIC: {res['aic']:.2f} | Params: {res['num_params']} | LogL: {res['log_likelihood']:.2f}")

		return fitted_results

	## Overrides ##

	def __str__(self):
		lines = []
		lines.append("🌍 Grid Dataset Summary")
		lines.append(f"  NetCDF Source: {getattr(self, 'source_file', 'unknown')}")
		lines.append(f"  Time Selected: {getattr(self, 'selected_time', 'N/A')}")
		lines.append(f"  Inflow Var: {self.inflow_var} ({self.ds[self.inflow_var].attrs['units']})")
		lines.append(f"  Outflow Var: {self.outflow_var} ({self.ds[self.inflow_var].attrs['units']})")
		lines.append(f"  Total Flow (IO): {self.ds[self.outflow_var].sum().item():,.2f} {self.ds[self.outflow_var].sum().item():,.2f}")

		if hasattr(self, "ensemble") and isinstance(self.ensemble, np.ndarray):
			lines.append("\n📊 Ensemble Graph")
			lines.append(f"  Shape: {self.ensemble.shape}")
			lines.append(f"  Total Weight: {self.ensemble.sum():,.2f}")
			lines.append(f"  Nonzero Edges: {(self.ensemble > 0).sum()}")

		if hasattr(self, "flow") and isinstance(self.flow, np.ndarray):
			lines.append("\n📦 Flow Graph")
			lines.append(f"  Shape: {self.flow.shape}")
			lines.append(f"  Total Flow: {self.flow.sum():,.2f}")
			lines.append(f"  Nonzero Edges: {(self.flow > 0).sum()}")

		return "\n".join(lines)