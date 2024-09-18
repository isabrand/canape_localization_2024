from progress.bar import Bar
import matplotlib.pyplot as plt
import math
import geopy.distance
import numpy as np
import numpy.ma as ma
import csv
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import NonUniformImage
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.colors as colors
from datetime import datetime
import matplotlib.patches as mpatches


class VIS:
    def __init__(self, wav_dir, v, shru, categories):
        self.wav_dir = wav_dir
        self.shru = shru
        self.categories = categories
        self.v = v

    # PLOT time series of call detections over time
    def presence(self, SHRU_LOC, area_x, area_y, all_data, path, all_locs):
        df = all_data
        species_list_by_phone = []
        species_names = ['BowheadWhale', 'BeardedSeal', 'RingedSeal', 'RibbonSeal', 'UnknownSeal']

        # SET UP FIGURE
        fig = plt.figure(figsize=(25, 25))
        ax = fig.add_subplot(111)
        ax.scatter(SHRU_LOC[:, 0], SHRU_LOC[:, 1], s=100, marker='^', color="k", zorder=2)

        # THIS IS THE BATHYMETRY SECTION FROM GEBCO GRIDDED BATHYMETRY 2023 DATASET ONLINE
        baths = ['bathymetry/bath1.csv', 'bathymetry/bath2.csv', 'bathymetry/bath3.csv', 'bathymetry/bath4.csv']
        bath = [[], [], [], []]
        for b in range(len(baths)):
            with open(baths[b], 'r') as f:
                reader = csv.reader(f)
                data = list(reader)
                bath[b] = np.array(data, dtype=float)
        bath_y = bath[0][1:, 0]
        bath_x = np.hstack((bath[0][0, 1:], bath[1][0, 1:], bath[2][0, 1:], bath[3][0, 1:]))
        bath_z = np.concatenate((bath[0][1:, 1:], bath[1][1:, 1:], bath[2][1:, 1:], bath[3][1:, 1:]), axis=1)

        # Add bathymetric map
        bath_lines = ax.contour(bath_x, bath_y, bath_z, cmap='Greys')
        bath_cb = fig.colorbar(bath_lines, shrink=0.8)
        bath_cb.lines[0].set_linewidth(3)
        bath_cb.ax.set_ylabel('Depth', fontsize=22)
        bath_cb.ax.tick_params(labelsize=18)

        # SEPARATE INTO CHANNEL
        one = df[df['Channel'] == 1]
        two = df[df['Channel'] == 2]
        three = df[df['Channel'] == 3]
        four = df[df['Channel'] == 4]
        ch_list = [one, two, three, four]
        ch_names = ['1', '2', '3', '4']
        ch_colors = ['seagreen', 'slateblue', 'palevioletred', 'orange']
        ch_markers = ["v", "s", "X", "o"]

        cols_species = ["dodgerblue", 'slategray', 'mediumseagreen', 'mediumpurple', 'gold', 'tomato']
        names_species = ["Bowhead Whale", "Bearded Seal", "Ringed Seal", "Ribbon Seal", "Unknown Seal", "Unspecified"]

        # GET ELLIPSES BOUNDS FOR EACH HYDROPHONE
        ch1_x, ch1_y = (np.asarray([-160.548, -159.0186, -157.488, -159.0186]),
                        np.asarray([72.90687, 73.357, 72.90687, 72.4567]))
        ch2_x, ch2_y = (np.asarray([-159.79, -158.27207, -156.755, -158.27207]),
                        np.asarray([72.7539, 73.2028, 72.7539, 72.304]))
        ch3_x, ch3_y = (np.asarray([-159.005, -157.4874, -155.97, -157.4874]),
                        np.asarray([72.7576, 73.208, 72.7576, 72.3072]))
        ch4_x, ch4_y = (np.asarray([-159.045, -157.5374, -156.03, -157.5374]),
                        np.asarray([72.61097, 73.061, 72.61097, 72.16094]))

        ch_x = [ch1_x, ch2_x, ch3_x, ch4_x]
        ch_y = [ch1_y, ch2_y, ch3_y, ch4_y]
        all_ellipse, all_ellipse_a, all_ellipse_b = [], [], []

        for x in range(4):
            chx, chy = ch_x[x], ch_y[x]
            xmean, ymean = chx.mean(), chy.mean()
            chx -= xmean
            chy -= ymean
            U, S, V = np.linalg.svd(np.stack((chx, chy)))

            tt = np.linspace(0, 2 * np.pi, 1000)
            circle = np.stack((np.cos(tt), np.sin(tt)))  # unit circle
            transform = np.sqrt(2 / 4) * U.dot(np.diag(S))  # transformation matrix
            fit = transform.dot(circle) + np.array([[xmean], [ymean]])
            all_ellipse.append(fit)

            all_ellipse_a.append((max(fit[0, :]) - min(fit[0, :])) / 2)
            all_ellipse_b.append((max(fit[1, :]) - min(fit[1, :])) / 2)

        x_in, y_in = [], []
        species_codes = [[51, 52], [11, 12, 13, 14, 15], [21, 22, 23], [31], [0, 41]]

        # GET RID OF POINTS OUTSIDE ELLIPSES
        for h in range(4):
            pos = ch_list[h]
            for p in range(5):
                spec = pos[pos['call_type'].isin(species_codes[p])]
                for l in range(len(spec['Longitude'])):
                    x = spec['Longitude'].values[l]
                    y = spec['Latitude'].values[l]
                    for r in range(4):
                        point = SHRU_LOC[r]
                        x_translated, y_translated = x - point[0], y - point[1]
                        a, b = all_ellipse_a[r], all_ellipse_b[r]
                        if (x_translated ** 2 / a ** 2) + (y_translated ** 2 / b ** 2) <= 1:
                            plt.scatter(x, y, col=cols_species[p], label=names_species[p], marker=ch_markers[r])

        ax.set_ylabel('Latitude', fontsize=40, labelpad=10)
        ax.set_xlabel('Longitude', fontsize=40, labelpad=10)
        ax.legend(fontsize=20)
        plt.title("All calls annotated per channel")
        plt.xlim(-162, -155)
        plt.ylim(71.5, 74)

        fig.savefig(path.joinpath('CallsByChannel.png'))

        return
