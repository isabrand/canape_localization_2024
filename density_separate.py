import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import geopy.distance
from scipy.ndimage import gaussian_filter
import matplotlib.colors as colors
import os
import csv

data_path = os.path.dirname(__file__)
data_path = os.path.join(data_path, 'sep_dens_plots/')


def plot_calls_over_time(data, path):
    date_format = '%Y%m%d-%H%M%S'
    print("\n\nRAW DATA: ", data.head())
    for index, row in data.iterrows():
        dt = row['DateTime']
        dt_fixed = datetime.strptime(dt[0:14], date_format)
        row['DateTime'] = dt_fixed
    data_sorted = data.sort_values(by='DateTime')
    data_sorted.reset_index(drop=True, inplace=True)
    print("\n\nSORTED DATA: ", data_sorted.head())

    bins = set(data_sorted['DateTime'])
    print(bins)

def Density(pos, data_path):
    # THIS IS THE BATHYMETRY SECTION FROM GEBCO GRIDDED BATHYMETRY 2023 DATASET ONLINE
    with open('bathymetry/beaufort_expanded.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        dataset = np.array(data, dtype=float)
    bath_y, bath_x, bath_z = dataset[1:, 0], dataset[0, 1:], dataset[1:, 1:]

    CATEGORIES = ["Bowhead whale", "Bearded seal", "Ringed seal", "Ribbon seal", "Unknown seal"]

    SHRU = [[-159.01806666666667, 72.90687166666666],
            [-158.27207166666668, 72.75391166666667],
            [-157.53745833333335, 72.61097000000000],
            [-157.48740333333333, 72.75763333333333]]
    SHRU_LOC = np.asarray(SHRU)

    min_xy = [-160.6, 71.1]
    max_xy = [-156, 74.4]
    lon_range = np.linspace(min_xy[0], max_xy[0], 1000)
    lat_range = np.linspace(min_xy[1], max_xy[1], 1000)
    nx, ny = 90, 70
    lon_bins = np.linspace(min(lon_range), max(lon_range), nx + 1)
    lat_bins = np.linspace(min(lat_range), max(lat_range), ny + 1)

    # divide into species specific groups
    inbounds = pos[(pos['Latitude'].between(min_xy[1], max_xy[1]))
                   & (pos['Longitude'].between(min_xy[0], max_xy[0]))
                   & (pos['Latitude'] != float('inf'))]
    bh = inbounds[inbounds['Species'].isin([51, 52])]
    bs = inbounds[inbounds['Species'].isin([11, 12, 13, 14, 15])]
    rs = inbounds[inbounds['Species'].isin([21, 22, 23])]
    rbs = inbounds[inbounds['Species'] == 31]
    u = inbounds[inbounds['Species'].isin([0, 41])]
    exists = ['Bowhead Whale', len(bh), 'Bearded Seal', len(bs), 'Ringed Seal', len(rs), 'Ribbon Seal', len(rbs)]
    xy = [bh, bs, rs, rbs]

    count = 0

    for i in xy:
        fig = plt.figure(figsize=(9, 5), clear=True, layout="tight", dpi=200)
        ax = plt.axes()

        # Add bathymetric map
        bath_lines = ax.contour(bath_x, bath_y, bath_z, cmap='Greys')
        bath_cb = fig.colorbar(bath_lines, shrink=0.8)
        bath_cb.lines[0].set_linewidth(3)
        bath_cb.ax.set_ylabel('Depth', fontsize=18)
        bath_cb.ax.tick_params(labelsize=14)

        xx = i['Longitude'].values
        yy = i['Latitude'].values
        zz, xedges, yedges = np.histogram2d(xx, yy, bins=[lon_bins, lat_bins])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        ax.scatter(SHRU_LOC[:, 0], SHRU_LOC[:, 1], s=20, color="k", edgecolors="k", zorder=2)
        # h = ax.imshow(zz.T, extent=extent, cmap='summer', origin='lower')
        # plt.colorbar(mappable=h)

        plt.pcolormesh(xedges, yedges, zz.T, cmap="summer", norm=LogNorm(), alpha=0.5, zorder=1)
        plt.xlim([min_xy[0], max_xy[0]])
        plt.ylim([min_xy[1], max_xy[1]])
        plt.title(exists[count * 2])
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar()

        file_path = ('species_' + (exists[count * 2]) + '_density.png')
        plt.savefig(data_path + file_path)
        plt.close(fig)

        count += 1


list_files = []
for i, filename in enumerate(pathlib.Path(data_path).glob('**/position.csv')):
    if str(filename)[-14] != "E" and i <= 5500:
        list_files.append(filename)
pos = pd.read_csv("sep_dens_plots/position_jul2024.csv")
Density(pos, data_path)
