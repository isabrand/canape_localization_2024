import numpy as np
import pandas as pd
import librosa
from scipy import signal
from sklearn.mixture import GaussianMixture
import rampy
import matplotlib.pyplot as plt


class Frequency:
    def __init__(self, wav_dir):
        self.wav_dir = wav_dir

    def Pixel(self, df, srt, end, c):
        srt, end, c = int(srt), int(end), int(c)
        call_group = df.iloc[srt:end, :].sort_values(['Begin Time (s)'], ascending=True)
        call_group = call_group.reset_index(drop=True)
        start = call_group['Begin Time (s)'].values[0]

        dt = call_group['DateTime'].values
        wav = call_group['WAV'].values
        ch = call_group['Channel'].values
        t = call_group['Begin Time (s)'].values
        sp = call_group['Call Type'].values
        reps = call_group['Repetition'].values
        locs = call_group['Localization'].values

        for index, row in call_group.iterrows():
            audio_multi, sr = librosa.load(self.wav_dir + call_group['WAV'].values[0], sr=None, offset=start - 5,
                                           duration=90, mono=False)
            audio = audio_multi[int(row["Channel"]) - 1]
            freqs, _, spec = signal.spectrogram(audio, fs=sr, window=signal.windows.gaussian(sr // 8, sr // 48),
                                                nperseg=sr // 8, noverlap=int(sr // 8 * 0.97), nfft=sr // 3)
            spec[spec < 0] = 0
            spec[spec == 0] = np.finfo(float).eps
            spec = 10 * np.log10(spec) + np.abs(np.min(10 * np.log10(spec)))
            spectro = spec[:, int((row['Begin Time (s)'] - (start - 5) - 2) * np.shape(spec)[1] / 90):int(
                (row['Begin Time (s)'] - (start - 5) + 2) * np.shape(spec)[1] / 90)]
            if len(spectro[0]) < 1:
                print('\nskipped', call_group['WAV'].values[0], '\nrow[Begin Time] :', (row['Begin Time (s)']),
                      '\nmaking the selected range of spec',
                      int((row['Begin Time (s)'] - (start - 5) - 2) * np.shape(spec)[1] / 90), 'to',
                      int((row['Begin Time (s)'] - (start - 5) + 2) * np.shape(spec)[1] / 90), 'and spec shape is',
                      np.shape(spec))
            else:
                percen = np.percentile(spectro, 92)
                pix = []
                for row_s in spectro:
                    pix.append(len(row_s[row_s >= percen]))
                pix = np.asarray(pix)
                if index == 0:
                    pix_num = pix
                else:
                    pix_num = pix_num + pix
        pix_num = pix_num[30:]
        freqs = freqs[30:]
        freq = np.linspace(int(np.min(freqs)), int(np.max(freqs)), len(pix_num))

        return pix_num, freq, dt, wav, ch, t, sp, reps, locs

    def Baseline(self, pix_num, freq):
        roi = np.array(
            [[min(freq), 100], [110, 190], [200, 260], [270, 350], [360, 450], [460, 550], [560, 650], [660, 790],
             [800, 1000], [1100, 1500], [1500, max(freq)]])
        y_rubberband, base_rubberband = rampy.baseline(freq, pix_num, roi, 'rubberband')
        y_rubberband = y_rubberband / np.amax(y_rubberband)
        ind = [i for i, e in enumerate(y_rubberband) if e >= 0.03]
        df_f = pd.DataFrame()
        df_f['frequency'] = [freq[e] for i, e in enumerate(ind)]
        df_f["y_rubberband"] = [y_rubberband[i] for i in ind]
        df_f["base_rubberband"] = [base_rubberband[i] for i in ind]
        return df_f, y_rubberband

    def GaussianModel(self, df_f):
        gmm_model = GaussianMixture(n_components=3, random_state=0)
        gmm_model.fit(df_f)
        cluster_labels = gmm_model.predict(df_f)
        df_f2 = df_f.copy()
        df_f2['cluster'] = cluster_labels
        num = df_f2['cluster'].iloc[-1]
        df_f2['cluster'] = df_f2["cluster"].replace(num, "last")
        high = df_f2.loc[df_f2['cluster'] == "last"]['frequency'].values[-1]
        low = df_f2.loc[df_f2['cluster'] == "last"]['frequency'].values[0]
        return low, high

    def NoiseMode(self, df_f):
        df_f2 = pd.DataFrame()
        ind = [i for i, e in enumerate(df_f["y_rubberband"].values) if e >= 0.05]
        df_f2['frequency'] = [df_f["frequency"].values[e] for i, e in enumerate(ind)]
        df_f2["y_rubberband"] = [df_f["y_rubberband"].values[i] for i in ind]
        df_f2["base_rubberband"] = [df_f["base_rubberband"].values[i] for i in ind]
        if (df_f2[(df_f2['frequency'] >= 950)]['frequency']).size > 0:
            high = df_f2[(df_f2['frequency'] >= 950)]['frequency'].values[-1]
            low = df_f2[(df_f2['frequency'] >= 950)]['frequency'].values[0]
        else:
            high, low = self.GaussianModel(df_f2)
        return low, high, df_f2

    def Plot(self, freq, y_rubberband, pix_num, df_f, low, high, srt, i, c, fp_path):
        fig, ax = plt.subplots(2, 1, figsize=(14, 12), clear=True, sharex=True, layout='constrained', dpi=200)
        ax[0].plot(freq, pix_num, "k-", label="Raw")
        ax[0].plot(df_f["frequency"].values, df_f["base_rubberband"].values, "--", color="red",
                   label="Rubberband baseline")
        ax[0].legend()
        ax[0].set_ylabel('Pixel Count', fontsize=12)
        ax[1].plot(freq, y_rubberband, "k")
        ax[1].plot([low] * len(df_f["y_rubberband"].values), df_f["y_rubberband"].values, "b--", label="Low frequency")
        ax[1].plot([high] * len(df_f["y_rubberband"].values), df_f["y_rubberband"].values, "b--",
                   label="High frequency")
        ax[1].set_ylabel('Normalized Pixel Count', fontsize=12)
        fig.supxlabel('Frequency (Hz)', fontsize=12)
        fig.savefig(fp_path.joinpath(
            'group_' + str(i + 1) + '_range_' + str(srt + 1) + '-' + str(srt + c) + '_frequency_range' + '.png'))
        plt.close(fig)
