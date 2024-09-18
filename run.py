import datetime
import json, os, time, sys
import pathlib
import shutil
import warnings
from itertools import groupby, combinations
import csv

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from progress.bar import Bar
import scipy.optimize as opt
from celluloid import Camera
from inputimeout import inputimeout, TimeoutOccurred
from matplotlib.animation import FFMpegWriter
from pyproj import Proj

import density_map
import frequency
import general
import spectrogram
import tdoa
import timelag
import all_data_plotting

warnings.filterwarnings("error")
warnings.filterwarnings('ignore')


def step_vis_all(config, path, presence_data, all_locs):
    SHRU_LOC = np.asarray([config["SHRU"][key][0:3] for key in config["SHRU"]])
    RANGELONG = np.asarray([config["RANGELONG_LOW"], config["RANGELONG_HIGH"]])
    RANGELAT = np.asarray([config["RANGELAT_LOW"], config["RANGELAT_HIGH"]])
    vis = all_data_plotting.VIS(config["AUDIO_DIR"], config["SPEED_OF_SOUND"], config["SHRU"], config["CATEGORIES"])
    #   vis.presence(SHRU_LOC, RANGELONG, RANGELAT, presence_data, path, all_locs)
    return


def step_frequency(config, counts, general_data, fp_path):
    srt_list = [0]
    for i in range(len(counts)):
        srt_list.append(srt_list[i] + counts[i])

    gen_w_freq = pd.DataFrame(columns=['DateTime', 'WAV', 'Channel', 'Begin Time (s)', 'High Freq', 'Low Freq',
                                       'Call Type', 'Repetition', 'Localization'])
    freq_range = pd.DataFrame(columns=['DateTime', 'High', 'Low', 'Localization', 'Repetition', 'Call Type'])
    f = frequency.Frequency(wav_dir=config["AUDIO_DIR"])

    with Bar('Frequency...', max=len(counts)) as bar:
        for i, c in enumerate(counts):
            try:
                pix_num, freq, dt, wav, ch, t, sp, reps, locs = f.Pixel(general_data, srt_list[i], srt_list[i + 1], c)
                df_f, y_rubberband = f.Baseline(pix_num, freq)
                if len(df_f[(df_f['frequency'] >= 800) & (df_f['frequency'] <= 950) & (
                        df_f["y_rubberband"] >= np.percentile(df_f["y_rubberband"].values, 95)[0])]) != 0 and len(
                    df_f[(df_f['frequency'] >= 950)]) >= 5:
                    low, high, df_f2 = f.NoiseMode(df_f)
                    df_f = df_f2
                else:
                    low, high = f.GaussianModel(df_f)

                # APPENDING COMPLETE INFO FOR EACH CALL TO GEN_W_FREQ
                for l in range(len(dt)):
                    to_add = [dt[l], wav[l], ch[l], t[l], high, low, sp[l], reps[l], locs[l]]
                    gen_w_freq.loc[len(gen_w_freq)] = to_add
                # APPENDING ONLY NECESSARY INFO TO FREQ_RANGE
                add = [dt[0], high, low, locs[0], reps[0], sp[0]]
                freq_range.loc[len(freq_range)] = add

                if list(config["PLOT"].values())[0]:
                    f.Plot(freq, y_rubberband, pix_num, df_f, low, high, srt_list[i], i, c, fp_path)

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print('step_frequency error:\n', e.__class__.__name__, ' ON LINE', exc_tb.tb_lineno, 'OF', fname, '\n')
                pass
            bar.next()
    return freq_range, gen_w_freq


def step_spectrogram(config, counts, df, freq_range, band_path, long_path, short_path):
    srt = [0]
    for i in range(len(counts)):
        srt.append(srt[i] + counts[i])

    s = spectrogram.Spectrogram(wav_dir=config["AUDIO_DIR"])

    if list(config["PLOT"].values())[1]:
        with Bar('Spectrogram...', max=len(counts)) as bar:
            for ii, c in enumerate(counts):
                try:
                    call_group = df.iloc[srt[ii]:srt[ii+1], :].sort_values(['Begin Time (s)'], ascending=True)
                    call_group = call_group.reset_index(drop=True)
                    start = call_group['Begin Time (s)'].values[0]
                    s.Plot(call_group, start, freq_range, c, ii, srt[ii], band_path, long_path, short_path)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print('step_spectrogram error:\nERROR TYPE:' + e.__class__.__name__ + ' ON LINE',
                          exc_tb.tb_lineno, 'OF', fname, '\n')
                    pass
                bar.next()


def step_lag(config, counts, gen_w_freq, freq_range, lp_path):
    time_lag, cc_lag, dt, locs, sp = pd.DataFrame(), [], [], [], []
    l = timelag.TimeLag(wav_dir=config["AUDIO_DIR"])

    srt_list = [0]
    for i in range(len(counts)):
        srt_list.append(srt_list[i] + counts[i])

    with Bar('Time Lag...', max=len(counts)) as bar:
        for i, c in enumerate(counts):
            try:
                c = int(c)
                lag = pd.DataFrame(columns=['DateTime', 'WAV', 'Call Type', 'Group Num', 'Localization', 'Channel#1',
                                            'Channel#2', 'Time#1', 'Time#2', 'Spectrogram lag'])
                call_group = gen_w_freq.iloc[srt_list[i]:srt_list[i+1], :].sort_values(['Begin Time (s)'], ascending=False)
                group = np.ndarray.tolist(call_group['Channel'].astype(int).values)
                pair_tuple = list(combinations(group, 2))
                pair = list(map(list, zip(*pair_tuple)))

                gn = i + 1
                ch1, ch2 = pair[0], pair[1]
                t1 = [call_group[call_group['Channel'] == n]['Begin Time (s)'].values[0] for n in pair[0]]
                t2 = [call_group[call_group['Channel'] == n]['Begin Time (s)'].values[0] for n in pair[1]]

                for k in range(len(ch1)):
                    spec_lag = t1[k]-t2[k]
                    to_add = [call_group['DateTime'].values[0], call_group['WAV'].values[0],
                              call_group['Call Type'].values[0], gn, call_group['Localization'].values[0], ch1[k],
                              ch2[k], t1[k], t2[k], spec_lag]
                    lag.loc[len(lag)] = to_add

                if i == 0:
                    time_lag = lag
                else:
                    time_lag = pd.concat([time_lag, lag], ignore_index=True)

                k = 0
                fig, ax = plt.subplots(int((c * (c - 1)) / 2), 2, figsize=(20, 12), sharex=False, dpi=300, clear=True,
                                       layout='constrained')

                for p1, p2 in zip(pair[0], pair[1]):
                    t1 = call_group[call_group['Channel'] == p1]['Begin Time (s)'].values[0] - 1
                    t2 = call_group[call_group['Channel'] == p2]['Begin Time (s)'].values[0] - 1
                    wav1 = call_group[call_group['Channel'] == p1]["WAV"].values[0]
                    wav2 = call_group[call_group['Channel'] == p2]["WAV"].values[0]
                    audio, sr1 = librosa.load(config["AUDIO_DIR"] + wav1, sr=None, offset=t1, duration=2, mono=False)
                    audio_p1 = audio[p1 - 1] - np.mean(audio[p1 - 1])
                    audio, sr2 = librosa.load(config["AUDIO_DIR"] + wav2, sr=None, offset=t2, duration=2, mono=False)
                    audio_p2 = audio[p2 - 1] - np.mean(audio[p2 - 1])
                    audio_p1 = librosa.util.normalize(
                        l.butter_bandpass_filter(audio_p1, freq_range["Low"].values[i], freq_range["High"].values[i],
                                                 sr1, order=6))
                    audio_p2 = librosa.util.normalize(
                        l.butter_bandpass_filter(audio_p2, freq_range["Low"].values[i], freq_range["High"].values[i],
                                                 sr2, order=6))
                    d1, d2 = pd.Series(audio_p1), pd.Series(audio_p2)
                    lgs = np.arange(-(1000), (1000 + 1))
                    rs = np.nan_to_num([l.crosscorr(d1, d2, lg) for lg in lgs])

                    if lgs[np.argmax(rs)] < 0:
                        j = -1
                    else:
                        j = 1

                    lag_s = j * abs(lgs[np.argmax(rs)]) * (4 / len(audio_p1))
                    cc_lag.append(lag_s)

                    if list(config["PLOT"].values())[2]:
                        l.Plot(audio_p1, audio_p2, lag_s, c, p1, p2, i, lp_path, k, ax, fig)
                        k = k + 1

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print('step_lag error:\nERROR TYPE:' + e.__class__.__name__ + ' ON LINE',
                      exc_tb.tb_lineno, 'OF', fname, '\n')
                for i in range(len(pair[0])):
                    cc_lag.append(0.0)
                pass

            bar.next()
    time_lag['Cross-correlation lag'] = cc_lag
    time_lag['Resulting lag'] = time_lag['Spectrogram lag'] + time_lag['Cross-correlation lag']
    return time_lag


def step_tdoa(time_lag, gen_w_freq, counts, config, lonlat_path, anim_path):
    gen_pos = gen_w_freq.copy()
    gen_lat, gen_lon, gen_x, gen_y = [], [], [], []
    position = pd.DataFrame(columns=['DateTime', 'Species', 'Latitude', 'Longitude', 'x', 'y', 'localization'])
    t = tdoa.TDOA(wav_dir=config["AUDIO_DIR"], v=config["SPEED_OF_SOUND"], shru=config["SHRU"],
                  categories=config["CATEGORIES"])
    P = Proj(config["PROJ"])

    RANGELONG = np.asarray([config["RANGELONG_LOW"], config["RANGELONG_HIGH"]])
    RANGELAT = np.asarray([config["RANGELAT_LOW"], config["RANGELAT_HIGH"]])
    SHRU_LOC = np.asarray([config["SHRU"][key][0:3] for key in config["SHRU"]])
    SHRU_XY = np.stack(P(SHRU_LOC[:, 0], SHRU_LOC[:, 1]), axis=1)

    srt_list = [0]
    for i in range(len(counts)):
        srt_list.append(srt_list[i] + counts[i])
    group_nums = [len(list(c)) for i, c in groupby(np.ndarray.tolist(time_lag["Group Num"].values))]
    srt_list = [0]
    for i in range(len(group_nums)):
        srt_list.append(srt_list[i] + group_nums[i])

    if list(config["PLOT"].values())[3]:
        with Bar('TDOA...', max=len(group_nums)) as bar:
            for index, num in enumerate(group_nums):
                try:
                    call = time_lag[srt_list[index]:srt_list[index+1]]
                    s = call['Call Type'].values[0]
                    loc = call['Localization'].values[0]
                    dt = call['DateTime'].values[0]
                    d = call['WAV'].values[0]
                    d = d[0:15]

                    # initial guess for the location of the event using the average of the receivers
                    mx = np.mean([SHRU_XY[m - 1][0] for m in np.unique(call.iloc[:, 5:7].values)])
                    my = np.mean([SHRU_XY[m - 1][1] for m in np.unique(call.iloc[:, 5:7].values)])

                    # TDOA calculation that results in a location, global optimization, picks best option
                    F = t.functions(SHRU_XY, call)
                    J = t.jacobian(SHRU_XY, call)
                    x, y = opt.leastsq(F, x0=[mx, my], Dfun=J, maxfev=1200)[0]
                    lon, lat = P(x, y, inverse=True)

                    # Plot the calculated X and Y with their hyperbolas
                    if list(config["PLOT"].values())[3]:
                        hx, hy, locWtime = t.hyperbola(P, SHRU_XY, call)
                        hlon, hlat = P(hx, hy, inverse=True)
                        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
                        xs, ys = t.bounds(locWtime, x, y)
                        num_recs = len(locWtime)

                        # Evaluate the system across the grid, plot P(cover_x[0], cover_y[1]) with my plotting function
                        hyperbs = [F((xs, ys))[0], F((xs, ys))[1], F((xs, ys))[2], 0, 0, 0]
                        if num_recs > 3:
                            hyperbs = [F((xs, ys))[0], F((xs, ys))[1], F((xs, ys))[2], F((xs, ys))[3], F((xs, ys))[
                                4], F((xs, ys))[5]]
                        t.plot(SHRU_LOC, P, lonlat_path, index, s, d, loc, lon, lat, xs, ys, RANGELONG, RANGELAT, hyperbs, num_recs)

                    to_add = [d, s, lat, lon, x, y, loc]
                    position.loc[len(position)] = to_add

                    for x in range(counts[index]):
                        gen_lat.append(lat)
                        gen_lon.append(lon)
                        gen_x.append(x)
                        gen_y.append(y)

                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print('step_tdoa error:\nERROR TYPE:' + e.__class__.__name__ + ' ON LINE',
                          exc_tb.tb_lineno, 'OF', fname, '\n')
                    pass

                bar.next()

    gen_pos['Longitude'] = gen_lon
    gen_pos['Latitude'] = gen_lat
    gen_pos['X'] = gen_x
    gen_pos['Y'] = gen_y

    #  if config["ANIMATION"] & list(config["PLOT"].values())[3]:
    #      animation = camera.animate()
    #      FFwriter = FFMpegWriter(fps=4, extra_args=['-vcodec', 'libx264'])
    #      animation.save(anim_path.joinpath('TDOAanimation.mp4'), writer=FFwriter)
    #      plt.close(fig2)

    return position, gen_pos


def step_dens(config, dens_path, position, presence_data, gen_pos):
    ds = density_map.DENS(wav_dir=config["AUDIO_DIR"], v=config["SPEED_OF_SOUND"], shru=config["SHRU"],
                          categories=config["CATEGORIES"])
    RANGELONG = np.asarray([config["RANGELONG_LOW"], config["RANGELONG_HIGH"]])
    RANGELAT = np.asarray([config["RANGELAT_LOW"], config["RANGELAT_HIGH"]])
    SHRU_LOC = np.asarray([config["SHRU"][key][0:3] for key in config["SHRU"]])
    if list(config["PLOT"].values())[4]:
        exists = ds.map(SHRU_LOC, RANGELONG, RANGELAT, position, dens_path)
        #ds.plot_calls_over_time(presence_data, gen_pos, dens_path)
    return exists


def run_from_config(config_path, log_path=None):
    file = open(config_path)
    config = json.load(file)
    run_cuts, run_cutoff = True, 30

    # Setting up the Results directory
    now_time = datetime.datetime.now(pytz.timezone('America/New_York'))
    if len(config["OUT_DIR"]) == 0:
        log_path = pathlib.Path(os.path.normpath(os.getcwd() + os.sep + os.pardir)).joinpath("Results")
    else:
        log_path = pathlib.Path(config["OUT_DIR"]).joinpath("Results")
    date_path = pathlib.Path(log_path).joinpath(now_time.strftime('%m-%d-%y'))
    data_path = pathlib.Path(date_path).joinpath(now_time.strftime('%H:%M:%S'))

    if not log_path.exists():
        os.mkdir(str(log_path))
    if not date_path.exists():
        os.mkdir(str(date_path))
    if not data_path.exists():
        os.mkdir(str(data_path))

    if any([value == True for value in config["LOAD"].values()]):
        all_sub = [log_path.joinpath(sub) for sub in os.listdir(log_path)]
        latest_date_subdir = max(all_sub, key=os.path.getmtime)
        all_sub = [latest_date_subdir.joinpath(sub) for sub in os.listdir(latest_date_subdir) if str(sub)[-1] == "S"]
        latest_data_subdir = max(all_sub, key=os.path.getmtime)

    if len(os.listdir(log_path)) > 0:
        for date_folder in os.listdir(log_path):
            old_date_path = os.path.join(log_path, date_folder)
            for data_folder in os.listdir(old_date_path):
                old_data_path = os.path.join(old_date_path, data_folder)
                if (os.stat(old_data_path).st_mtime < (time.time() - 3600 / 6)) & (str(old_data_path)[-1] != "S"):
                    shutil.rmtree(old_data_path)
            if (len(os.listdir(old_date_path)) == 0) & (os.stat(old_date_path).st_mtime <= (time.time() - 24 * 3600)):
                shutil.rmtree(old_date_path)

    json.dump(config, open(data_path.joinpath(config_path.name), mode='a'))
    gen_path = pathlib.Path(data_path).joinpath("general")
    os.mkdir(str(gen_path))

    # Starting processing the actual data
    if not list(config["LOAD"].values())[0]:  # "general" = false
        d = general.General(txt_dir=config["ANNOTATION_DIR"], wav_dir=config["AUDIO_DIR"])

        # OPTION TO PULL ONLY 1 MONTH OF DATA
        divide_month = True
        month = '201612'

        # GENERATE NEW DATA FILES TO PROCESS
        general_data, counts, species, presence_data = d.data(divide_month, month, run_cuts, run_cutoff)
    elif list(config["LOAD"].values())[0]:  # "general" = true
        general_data = pd.read_csv('use_gen/general/general_output.csv')
        presence_data = pd.read_csv('use_gen/general/presence_output.csv')
        counts = np.loadtxt('use_gen/general/counts.txt')
        species = np.loadtxt('use_gen/general/species.txt')
    if config["INTERVAL"]:
        general_data = general_data[
                       int(sum(counts[0:config["LOWER_LIMIT"] - 1])):int(sum(counts[0:config["UPPER_LIMIT"]]))]
        counts = counts[config["LOWER_LIMIT"] - 1:config["UPPER_LIMIT"]]
        species = species[config["LOWER_LIMIT"] - 1:config["UPPER_LIMIT"]]

    general_data.to_csv(gen_path.joinpath('general_output.csv'), index=False)
    presence_data.to_csv(gen_path.joinpath('presence_output.csv'), index=False)
    all_loc_data = pd.read_csv('use_gen/general/all_localizable_pos.csv')

    step_vis_all(config, gen_path, presence_data, all_loc_data)

    # I changed counts so that experimental runs are shorter. Delete section to run max as listed in config file.
    counts = counts[0:run_cutoff]

    try:
        with open(gen_path.joinpath('counts.txt'), 'w') as f:
            for line in counts:
                f.write(f"{line}\n")
        with open(gen_path.joinpath('species.txt'), 'w') as f:
            for line in species:
                f.write(f"{line}\n")
        if not list(config["STOP"].values())[0]:
            freq_path = pathlib.Path(data_path).joinpath("frequency")
            os.mkdir(str(freq_path))
            if list(config["PLOT"].values())[0]:
                fp_path = pathlib.Path(freq_path).joinpath("plots")
                os.mkdir(str(fp_path))
            elif not list(config["PLOT"].values())[0]:
                fp_path = None
            if not list(config["LOAD"].values())[1]:
                freq_range, gen_w_freq = step_frequency(config, counts, general_data, fp_path=fp_path)
            elif list(config["LOAD"].values())[1]:
                freq_range = pd.read_csv(latest_data_subdir.joinpath('frequency/freq_range.csv'))
                gen_w_freq = pd.read_csv(latest_data_subdir.joinpath('frequency/gen_w_freq.csv'))
                if config["INTERVAL"]:
                    freq_range = freq_range[int(sum(counts[0:config["LOWER_LIMIT"] - 1])):int(
                        sum(counts[0:config["UPPER_LIMIT"]]))]
                    gen_w_freq = gen_w_freq[int(sum(counts[0:config["LOWER_LIMIT"] - 1])):int(
                        sum(counts[0:config["UPPER_LIMIT"]]))]
            freq_range.to_csv(freq_path.joinpath('freq_range.csv'), index=False)
            gen_w_freq.to_csv(freq_path.joinpath('gen_w_freq.csv'), index=False)
            if not list(config["STOP"].values())[1]:
                if list(config["PLOT"].values())[1]:
                    spec_path = pathlib.Path(data_path).joinpath("spectrogram")
                    sp_path = pathlib.Path(spec_path).joinpath("plots")
                    band_path = pathlib.Path(sp_path).joinpath("bandpass")
                    long_path = pathlib.Path(sp_path).joinpath("long_time_interval")
                    short_path = pathlib.Path(sp_path).joinpath("short_time_interval")
                    os.mkdir(str(spec_path))
                    os.mkdir(str(sp_path))
                    os.mkdir(str(band_path))
                    os.mkdir(str(long_path))
                    os.mkdir(str(short_path))
                    step_spectrogram(config, counts, gen_w_freq, freq_range, band_path=band_path, long_path=long_path,
                                     short_path=short_path)
                if not list(config["STOP"].values())[2]:
                    lag_path = pathlib.Path(data_path).joinpath("time_lag")
                    os.mkdir(str(lag_path))
                    if list(config["PLOT"].values())[2]:
                        lp_path = pathlib.Path(lag_path).joinpath("plots")
                        os.mkdir(str(lp_path))
                    elif not list(config["PLOT"].values())[2]:
                        lp_path = None
                    if not list(config["LOAD"].values())[2]:
                        time_lag = step_lag(config, counts, gen_w_freq, freq_range, lp_path)
                    elif list(config["LOAD"].values())[2]:
                        time_lag = pd.read_csv(latest_data_subdir.joinpath('time_lag/time_lag.csv'))
                        if config["INTERVAL"]:
                            time_lag = time_lag[int(sum(counts[0:config["LOWER_LIMIT"] - 1])):int(
                                sum(counts[0:config["UPPER_LIMIT"]]))]
                    time_lag.to_csv(lag_path.joinpath('time_lag.csv'), index=False)
                    if not list(config["STOP"].values())[3]:
                        tdoa_path = pathlib.Path(data_path).joinpath("TDOA")
                        os.mkdir(str(tdoa_path))
                        if list(config["PLOT"].values())[3]:
                            tp_path = pathlib.Path(tdoa_path).joinpath("plots")
                            lonlat_path = pathlib.Path(tp_path).joinpath("lonlat")
                            os.mkdir(str(tp_path))
                            os.mkdir(str(lonlat_path))
                        elif not list(config["PLOT"].values())[3]:
                            tp_path = None
                            lonlat_path = None
                        if list(config["PLOT"].values())[4]:
                            dens_path = pathlib.Path(data_path).joinpath("density_map")
                            os.mkdir(str(dens_path))
                        elif not list(config["PLOT"].values())[4]:
                            dens_path = None
                        if config["ANIMATION"]:
                            anim_path = pathlib.Path(tdoa_path).joinpath("anim")
                            os.mkdir(str(anim_path))
                        position, gen_pos = step_tdoa(time_lag, gen_w_freq, counts, config, lonlat_path=lonlat_path, anim_path=anim_path)

                        exists = step_dens(config, dens_path, position, presence_data, gen_pos)
                        position.to_csv(tdoa_path.joinpath('position.csv'), index=False)
                        gen_pos.to_csv(tdoa_path.joinpath('gen_w_pos.csv'), index=False)
                    else:
                        pass
                else:
                    pass
            else:
                pass
        else:
            pass
        if config["SAVE"]:
            os.rename(os.path.join(date_path, now_time.strftime('%H:%M:%S')),
                      os.path.join(date_path, now_time.strftime('%H:%M:%S') + "_S"))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('run_from_config error:\nERROR TYPE:' + e.__class__.__name__ + ' ON LINE',
              exc_tb.tb_lineno, 'OF', fname, '\n')
        pass
    return


if __name__ == '__main__':
    try:
        print('\nWelcome to SHRU Localization!\nThis code runs using the config file found in', os.getcwd(), '.\n\n')
        config_file = pathlib.Path(os.getcwd() + '/config.json')
        run_from_config(config_file)
    except Exception as e:
        print('\nSORRY, there is a problem!\n', sys.exc_info())
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('\nERROR TYPE:' + e.__class__.__name__ + ' ON LINE', exc_tb.tb_lineno, 'OF', fname, '\n')
        if e.__class__ == FileNotFoundError:
            print('\nThe config file is not in this directory.\nPlease confirm config file call!\n')
        else:
            print('\nPlease check input and code files for errors\n')
            print('If you think the problem is not user error, contact isabelle.brandicourt@whoi.edu\nGood luck!\n')
        # sys.exit()
