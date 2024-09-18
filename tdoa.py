import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
import numpy as np
import csv
from pyproj import Proj, transform


class TDOA:
    def __init__(self, wav_dir, v, shru, categories):
        self.wav_dir = wav_dir
        self.shru = shru
        self.categories = categories
        self.v = v

    def hyperbola(self, P, SHRU_XY, data):
        x, y = [], []
        for _, row in data.iterrows():
            x0, y0 = SHRU_XY[int(row["Channel#2"] - 1)][0], SHRU_XY[int(row["Channel#2"] - 1)][1]
            x1, y1 = SHRU_XY[int(row["Channel#1"] - 1)][0], SHRU_XY[int(row["Channel#1"] - 1)][1]
            dist = row["Resulting lag"] * self.v  # Calculate the distance btwn receiver and source
            t0a = math.pi
            t1a = -math.pi
            t = np.linspace(t0a, t1a, 100)
            c = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2
            a = dist / 2
            b = np.sqrt(c ** 2 - a ** 2)
            X = [a * math.cosh(tt) for tt in t]
            Y = [b * math.sinh(tt) for tt in t]
            ca = (x1 - x0) / (2 * c)
            sa = (y1 - y0) / (2 * c)
            x01 = [(x0 + x1) / 2 + i * ca - j * sa for i, j in zip(X, Y)]
            y01 = [(y0 + y1) / 2 + i * sa + j * ca for i, j in zip(X, Y)]
            x.append(x01)
            y.append(y01)
        x = np.asarray(x)
        y = np.asarray(y)

        #### Isabelle's added part to fix parabolas
        ch0, ch1, res_lag = data['Channel#1'].to_numpy(), data['Channel#2'].to_numpy(), data['Resulting lag'].to_numpy()
        ch = np.unique(np.concatenate((ch0, ch1)))
        locs = []
        resLag = np.zeros(len(ch))

        # get specific locations just for the channels used each run
        for i in range(len(ch)):
            if ch[i] == 3:
                locs.append(SHRU_XY[3])
            elif ch[i] == 4:
                locs.append(SHRU_XY[2])
            else:
                locs.append(SHRU_XY[ch[i] - 1])

        # get specific lag for each channel in relation to the first recorded channel
        for n, item in enumerate(ch0):
            if item == min(ch): resLag[np.where(ch == ch1[n])[0][0]] = res_lag[n]
        for n, item in enumerate(ch1):
            if item == min(ch): resLag[np.where(ch == ch0[n])[0][0]] = res_lag[n]
        to_add = np.vstack(resLag)
        locs_w_time = np.hstack((locs, to_add))
        return x, y, locs_w_time

    def bounds(self, locWtime, x, y):
        x_all, y_all, t_all = [0] * len(locWtime), [0] * len(locWtime), [0] * len(locWtime)
        if len(locWtime) > 1:
            x_all[0], y_all[0], t_all[0] = locWtime[0]
            x_all[1], y_all[1], t_all[1] = locWtime[1]

        if len(locWtime) > 2:
            x_all[2], y_all[2], t_all[2] = locWtime[2]

        if len(locWtime) > 3:
            x_all[3], y_all[3], t_all[3] = locWtime[3]

        max_x = max(max(x_all), x)
        min_x = min(min(x_all), x)
        max_y = max(max(y_all), y)
        min_y = min(min(y_all), y)
        range_x = max_x - min_x
        min_x -= range_x * .2
        max_x += range_x * .2
        range_y = max_y - min_y
        min_y -= range_y * .2
        max_y += range_y * .2

        # Create a grid of input coordinates
        xs = np.linspace(min_x, max_x, 100)
        ys = np.linspace(min_y, max_y, 100)

        ys = np.linspace(8000371.60, 8146904.42, 100)
        xs = np.linspace(465696.18, 582885.35, 100)

        xs, ys = np.meshgrid(xs, ys)

        return xs, ys

    def functions(self, SHRU_XY, call):
        def fn(args):
            x, y = args
            F = []
            prev = [0, 0]
            for _, row in call.iterrows():
                x0, y0 = SHRU_XY[int(row["Channel#2"] - 1)][0], SHRU_XY[int(row["Channel#2"] - 1)][1]
                x1, y1 = SHRU_XY[int(row["Channel#1"] - 1)][0], SHRU_XY[int(row["Channel#1"] - 1)][1]
                # Perform calculation to get hyperbola for each case. X0, y0, x1, y1, x2, y2, d01, d02, d12 as args
                time = row['Resulting lag'] * self.v  # multiply lag between channels by the speed of sound, 1520 m/s
                a = np.sqrt(np.power(x - x1, 2.) + np.power(y - y1, 2.)) - np.sqrt(
                    np.power(x - x0, 2.) + np.power(y - y0, 2.)) - time
                F.append(a)
            return F

        return fn

    def jacobian(self, SHRU_XY, call):
        # Jacobian matrix collects part derivs of each function w respect to independent variable. Helps the solver.
        def fn(args):
            x, y = args
            J = []
            for row in call.itertuples(index=False):
                j = []
                x0, y0 = SHRU_XY[int(row["Channel#2"] - 1)][0], SHRU_XY[int(row["Channel#2"] - 1)][1]
                x1, y1 = SHRU_XY[int(row["Channel#1"] - 1)][0], SHRU_XY[int(row["Channel#1"] - 1)][1]
                adx = (x - x1) / np.sqrt(np.power(x - x1, 2.) + np.power(y - y1, 2.)) - (x - x0) / np.sqrt(
                    np.power(x - x0, 2.) + np.power(y - y0, 2.))
                ady = (y - y1) / np.sqrt(np.power(x - x1, 2.) + np.power(y - y1, 2.)) - (y - y0) / np.sqrt(
                    np.power(x - x0, 2.) + np.power(y - y0, 2.))
                j.append(adx)
                j.append(ady)
                J.append(j)
            return J
        return fn

    def plot(self, SHRU_LOC, P, lonlatpath, index, s, dt, loc, ll_x, ll_y, xs, ys, hyperbs, num_recs):
        # Set up the plot
        fig = plt.figure(figsize=(6, 5), dpi=200)
        ax2 = plt.axes()
        ax2.grid()
        ax2.set_xlim(497117.285991, 552328.955078)
        ax2.set_ylim(8055297.40851, 8092371.20121)

        # Convert UTM coordinates to latitude and longitude for axis names
        utm_x, utm_y, latitudes, longitudes = ax2.get_xticks(), ax2.get_yticks(), [], []
        for i in range(len(utm_x)):
            e, g = P(utm_x[i], utm_y[i], inverse=True)
            latitudes.append(float(f"{e:.2f}"))
            longitudes.append(float(f"{g:.2f}"))

        for x in range(len(SHRU_LOC[:, 0])):
            coord = P(SHRU_LOC[:, 0][x], SHRU_LOC[:, 1][x])
            ax2.scatter(coord[0], coord[1], s=20, marker='^', color="k", zorder=2)

        ax2.contour(xs, ys, hyperbs[0], [0], colors='darkorange')
        ax2.contour(xs, ys, hyperbs[1], [0], colors='olivedrab')
        ax2.contour(xs, ys, hyperbs[2], [0], colors='lightseagreen')

        if num_recs > 3:
            ax2.contour(xs, ys, hyperbs[3], [0], colors='mediumpurple')
            ax2.contour(xs, ys, hyperbs[4], [0], colors='orchid')
            ax2.contour(xs, ys, hyperbs[5], [0], colors='steelblue')

        x, y = P(ll_x, ll_y)
        cdict_species = {0: "dodgerblue", 1: 'slategray', 2: 'mediumseagreen', 3: 'mediumpurple', 4: 'gold', 5: 'tomato'}
        n_species = {0: "Bowhead Whale", 1: "Bearded Seal", 2: "Ringed Seal", 3: "Ribbon Seal", 4: "Unknown Seal",
             5: "Unspecified"}
        ax2.scatter(x, y, color=cdict_species[self.categories[str(int(s))][0]], label=n_species[self.categories[str(int(s))][0]])

        custom1 = [Line2D([], [], marker='.', color=cdict_species[0], linestyle='None', label=n_species[0]),
                  Line2D([], [], marker='.', color=cdict_species[1], linestyle='None', label=n_species[1]),
                  Line2D([], [], marker='.', color=cdict_species[2], linestyle='None', label=n_species[2]),
                  Line2D([], [], marker='.', color=cdict_species[3], linestyle='None', label=n_species[3]),
                  Line2D([], [], marker='.', color=cdict_species[4], linestyle='None', label=n_species[4]),
                  Line2D([], [], marker='.', color=cdict_species[5], linestyle='None', label=n_species[5])]

        legend1 = fig.legend(handles=custom1, loc='upper right', ncol=1, fontsize=8)

        ax2.set_xticklabels(latitudes)
        ax2.set_yticklabels(longitudes)
        ax2.set_title('TDOA Localization '+str(int(loc))+'\n'+dt[0:31], fontsize=12)
        ax2.set_xlabel('Longitude', fontsize=10)
        ax2.set_ylabel('Latitude', fontsize=10)

        fig.savefig(lonlatpath.joinpath('group_' + str(index + 1) + 'lonlat.png'))
