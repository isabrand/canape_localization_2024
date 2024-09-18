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


class DENS:
    def __init__(self, wav_dir, v, shru, categories):
        self.wav_dir = wav_dir
        self.shru = shru
        self.categories = categories
        self.v = v

    # PLOT time series of call detections over time
    def plot_calls_over_time(self, all_data, gen_pos, path):
        df = pd.merge(all_data, gen_pos[['WAV', 'Localization', 'Channel', 'Longitude', 'Latitude']],
                      on=['WAV', 'Localization', 'Channel'], how='left')

        # SET UP FIGURE
        fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(25, 30))  # 5 rows, 1 column

        # GET BINS BASED ON DATE CHUNKS
        total_bins = df['DateTime'].dt.to_period('M').nunique()
        months = ['201611', '201612', '201701', '201702', '201703', '201704', '201705', '201706', '201707']
        month_colors = ['r', 'm', 'b', 'dodgerblue', 'c', 'g', 'lime', 'y', 'orange']
        leg_labels = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

        # SEPARATE INTO SPECIES
        bh = df[df['Call Type'].isin([51, 52])]
        bs = df[df['Call Type'].isin([11, 12, 13, 14, 15])]
        rs = df[df['Call Type'].isin([21, 22, 23])]
        rbs = df[df['Call Type'] == 31]
        u = df[df['Call Type'].isin([0, 41])]
        species_list = [bh, bs, rs, rbs, u]
        species_names = ['BowheadWhale', 'BeardedSeal', 'RingedSeal', 'RibbonSeal', 'UnknownSeal']

        print('here, species list is ', species_list)
        for i in range(len(species_list)):
            date_str, reps = [], []
            using = species_list[i]
            print('here in the for loop with df as ', df)
            using['DateTime'] = pd.to_datetime(using['DateTime'], errors='coerce')
            spec_bins = using['DateTime'].dt.to_period('M').unique()
            dates = spec_bins(str).tolist()
            print('spec bins is ', spec_bins)
            print('dates is ', dates)

            for x in spec_bins:
                reps.append(df['DateTime'].dt.to_period('M').value_counts().sort_index())
                date_str.append(x)

            # ADD IN ZEROES FOR REST OF DATES
            all_reps = []
            for d in range(len(total_bins)):
                if total_bins[d] in dates:
                    where = dates.index(total_bins[d])
                    all_reps.append(reps[where])
                else:
                    all_reps.append(0)

            # PLOT BAR PLOT
            colors = [month_colors[months.index(date.strftime('%Y%m%d-%H%M%S')[0:6])] for date in total_bins]
            log_reps = np.log10(all_reps)
            axs[i].bar(total_bins, all_reps, color=colors)
          #  axs[i].set_yscale('log')
            axs[i].set_ylabel(species_names[i] + '\nNum Calls', fontsize=40, labelpad=10)

        for ax in axs:
            ax.tick_params(axis='x', labelsize=30)
            ax.tick_params(axis='y', labelsize=30)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="left")

        legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(month_colors, leg_labels)]
        fig.legend(handles=legend_patches, loc='lower center', ncol=len(leg_labels), fontsize=30)

        axs[i].set_xlabel('Date', fontsize=40)
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('Species Presence over Time', fontsize=50)
        fig.savefig(path.joinpath('Species_Presence_Over_Time.png'))

        return

    def map(self, SHRU_LOC, area_x, area_y, position, dens_path):
        position['Species'] = position['Species'].astype(int)

        # SETS UP CIRCLES FOR EACH CHANNEL RANGE
        ch1_x, ch1_y = np.asarray([-160.548, -159.0186, -157.488, -159.0186]), np.asarray(
            [72.90687, 73.357, 72.90687, 72.4567])
        ch2_x, ch2_y = np.asarray([-159.79, -158.27207, -156.755, -158.27207]), np.asarray(
            [72.7539, 73.2028, 72.7539, 72.304])
        ch3_x, ch3_y = np.asarray([-159.005, -157.4874, -155.97, -157.4874]), np.asarray(
            [72.7576, 73.208, 72.7576, 72.3072])
        ch4_x, ch4_y = np.asarray([-159.045, -157.5374, -156.03, -157.5374]), np.asarray(
            [72.61097, 73.061, 72.61097, 72.16094])

        ch_x = [ch1_x, ch2_x, ch3_x, ch4_x]
        ch_y = [ch1_y, ch2_y, ch3_y, ch4_y]
        circles, circlesx, circlesy = [], [], []

        for c in range(4):
            chx = ch_x[c]
            chy = ch_y[c]
            xmean, ymean = chx.mean(), chy.mean()
            chx -= xmean
            chy -= ymean
            U, S, V = np.linalg.svd(np.stack((chx, chy)))
            tt = np.linspace(0, 2 * np.pi, 1000)
            circle = np.stack((np.cos(tt), np.sin(tt)))  # unit circle
            transform = np.sqrt(2 / 4) * U.dot(np.diag(S))  # transformation matrix
            fit = transform.dot(circle) + np.array([[xmean], [ymean]])
            circles.append(fit)
            circlesx.append((max(fit[0, :]) - min(fit[0, :])) / 2)
            circlesy.append((max(fit[1, :]) - min(fit[1, :])) / 2)

        xy_inells = [pd.DataFrame() for _ in range(6)]
        spec_codes = [[51, 52], [11, 12, 13, 14, 15], [21, 22, 23], [31], [0, 41]]

        # GET RID OF POINTS OUTSIDE ELLIPSES
        for p in range(5):
            spec = position[position['Species'].isin(spec_codes[p])]
            for row in spec.itertuples(index=False):
                for r in range(4):
                    point = SHRU_LOC[r]
                    x_translated = row.Longitude - point[0]
                    y_translated = row.Latitude - point[1]
                    a, b = circlesx[r], circlesy[r]
                    if (x_translated ** 2 / a ** 2) + (y_translated ** 2 / b ** 2) <= 1:
                        rw = pd.DataFrame([row._asdict()])
                        xy_inells[p] = pd.concat([xy_inells[p], rw], ignore_index=True)
                        xy_inells[5] = pd.concat([xy_inells[5], rw], ignore_index=True)
                        break

        exists = ['BowheadWhale', len(xy_inells[0]), 'BeardedSeal', len(xy_inells[1]), 'RingedSeal', len(xy_inells[2]),
                  'RibbonSeal', len(xy_inells[3]), 'Unknown/Unspecified', len(xy_inells[4]), 'All', len(xy_inells[5])]

        # make modified colormap
        color_array = plt.get_cmap('GnBu')(range(256))
        color_array[:, -1] = np.linspace(0.0, 1.0, 256)
        map_object = LinearSegmentedColormap.from_list(name='trans_gnbu', colors=color_array)
        plt.register_cmap(cmap=map_object)

        # get the distance of lat and long covered in kilometers
        x_km = geopy.distance.geodesic((area_y[0], area_x[0]), (area_y[0], area_x[1])).km
        y_km = geopy.distance.geodesic((area_y[0], area_x[0]), (area_y[1], area_x[0])).km

        print("Plotted latitude ranges from", area_y[0], 'to', area_y[1], '.')
        print("Plotted longitude ranges from", area_x[0], 'to', area_x[1], '.')
        print('This gives a covered area of', x_km, "km of longitude and", y_km, 'km of latitude, so', (x_km * y_km),
              'km^2.')

        x_y_diff = [area_x[1] - area_x[0], area_y[1] - area_y[0]]
        bins_calc = [x_y_diff[0] / (x_km / (x_km / 38)), x_y_diff[1] / (y_km / (y_km / 40))]
        bins1 = np.arange(area_x[0], area_x[1] + bins_calc[0], bins_calc[0])
        bins2 = np.arange(area_y[0], area_y[1] + bins_calc[1], bins_calc[1])

        x_km_bin_div, y_km_bin_div = round(int(x_km) / 5), round(int(y_km) / 5)
        binw_x, binw_y = int(x_km) / x_km_bin_div, int(y_km) / y_km_bin_div

        print('binw_x and binw_y are', binw_x, binw_y, ', making the bin area', binw_y * binw_x, 'km^2')

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

        with Bar('Density Plots...', max=6) as bar:
            for i in range(6):
                if exists[(i * 2) + 1] > 0:
                    print('\nPlotting', exists[(i * 2) + 1], 'localizations of', exists[i * 2])

                    # Set up plot
                    fig = plt.figure(figsize=(25, 25))
                    frame = xy_inells[i]

                    ax = fig.add_subplot(211)
                    ax.set_xlabel("Longitude", fontsize=30)
                    ax.set_ylabel("Latitude", fontsize=30)
                    ax.tick_params(axis='both', which='major', labelsize=20, direction='inout', length=6, width=2)
                    ax.set_xlim(-161, -155.5)
                    ax.set_ylim(72.0, 73.5)
                    ax.scatter(SHRU_LOC[:, 0], SHRU_LOC[:, 1], s=100, marker='^', color="k", zorder=2)
                    for x in range(len(circles)):
                        ell = circles[x]
                        plt.plot(ell[0, :], ell[1, :], 'r', linestyle='--', linewidth=.5)

                    # Add bathymetric map
                    bath_lines = ax.contour(bath_x, bath_y, bath_z, cmap='Greys')
                    bath_cb = fig.colorbar(bath_lines, shrink=0.8)
                    bath_cb.lines[0].set_linewidth(3)
                    bath_cb.ax.set_ylabel('Depth', fontsize=22)
                    bath_cb.ax.tick_params(labelsize=20)

                    count = 0
                    month_colors = ['b', 'dodgerblue', 'c', 'g', 'lime', 'y', 'orange', 'r', 'm']
                    months = ['Jan 2017', 'Feb 2017', 'Mar 2017', 'Apr 2017', 'May 2017', 'June 2017', 'July 2017',
                              'Nov 2016', 'Dec 2016']
                    all_months = [1, 2, 3, 4, 5, 6, 7, 11, 12]

                    for n in all_months:
                        if n in [11, 12]:
                            month = '2016' + str(n)
                        else:
                            month = '20170' + str(n)
                        monthly_data = frame[frame['DateTime'].str.startswith(month)]
                        ax.scatter(monthly_data['Longitude'], monthly_data['Latitude'], label=months[count],
                                   color=month_colors[count])
                        count += 1

                    ax.legend(fontsize=20)

                    ax = fig.add_subplot(212)
                    ax.set_xlabel("Longitude", fontsize=30)
                    ax.set_ylabel("Latitude", fontsize=30)
                    ax.tick_params(axis='both', which='major', labelsize=20, direction='inout', length=6, width=2)
                    ax.set_xlim(-161, -155.5)
                    ax.set_ylim(72.0, 73.5)
                    ax.scatter(SHRU_LOC[:, 0], SHRU_LOC[:, 1], s=100, marker='^', color="k", zorder=2)
                    for x in range(len(circles)):
                        ell = circles[x]
                        plt.plot(ell[0, :], ell[1, :], 'r', linestyle='--', linewidth=.5)

                    # Add bathymetric map
                    bath_lines = ax.contour(bath_x, bath_y, bath_z, cmap='Greys')
                    bath_cb = fig.colorbar(bath_lines, shrink=0.8, pad=0.01)
                    bath_cb.lines[0].set_linewidth(3)
                    bath_cb.ax.set_ylabel('Depth', fontsize=22)
                    bath_cb.ax.tick_params(labelsize=20)

                    D, xlim, ylim = np.histogram2d(frame['Longitude'], frame['Latitude'], bins=(bins1, bins2))
                    X, Y = np.meshgrid(xlim, ylim)
                    norm = colors.PowerNorm(gamma=0.5)
                    D_masked = ma.masked_where(D.T == 0, D.T)
                    plt.pcolormesh(xlim, ylim, D_masked, cmap="summer", norm=norm, alpha=0.5, zorder=1)
                    spec_cb = plt.colorbar(shrink=0.8, pad=0.01)
                    spec_cb.ax.set_ylabel('Species Density', fontsize=22)
                    spec_cb.ax.tick_params(labelsize=20)

                    fig.suptitle('Density Map : ' + exists[i * 2] + '\nTotal Count = ' + str(exists[(i * 2) + 1]),
                                 fontsize=26)
                    fig.savefig(dens_path.joinpath('Species_' + exists[i * 2] + '.png'))

                bar.next()

        return exists
