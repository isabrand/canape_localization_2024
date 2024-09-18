import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class TimeLag:
    def __init__(self, wav_dir):
        self.wav_dir = wav_dir

    def crosscorr(self, datax, datay, lag=0):
        return datax.corr(datay.shift(lag))

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        if highcut > 1950:
            highcut = 1950
        if lowcut > highcut:
            temp = lowcut
            lowcut = highcut
            highcut = temp
        b, a = signal.butter(order, [lowcut, highcut], fs=fs, btype='band')
        y = signal.lfilter(b, a, data)
        return y

    def Plot(self, audio_p1, audio_p2, lag_s, c, p1, p2, i, lp_path, k, ax, fig):
        x = np.linspace(0, 4, len(audio_p1))
        ax[k, 0].plot(x, audio_p1, c='b', lw=0.5, label='Channel ' + str(p1), alpha=0.6)
        ax[k, 0].plot(x, audio_p2, c='r', lw=0.5, label='Channel ' + str(p2), alpha=0.6)
        ax[k, 0].legend(fontsize=8)
        ax[k, 0].set_xlabel("Time (s)", fontsize=10)
        ax[k, 0].set_ylabel("Amplitude", fontsize=10)
        ax[k, 1].plot(x, audio_p1, c='b', lw=0.5, label='Channel ' + str(p1), alpha=0.6)
        x_shift = x + lag_s
        ax[k, 1].plot(x_shift, audio_p2, c='r', lw=0.5, label='Channel ' + str(p2), alpha=0.6)
        ax[k, 1].plot([], [], linestyle='None', label='Lag: ' + str(round(lag_s, 3)) + " s")
        ax[k, 1].legend(fontsize=8)
        ax[k, 1].set_xlabel("Time (s)", fontsize=10)
        ax[k, 1].set_ylabel("Amplitude", fontsize=10)
        if (k + 1) * 2 == c * (c - 1):
            fig.savefig(lp_path.joinpath('group_' + str(i + 1) + '.png'))
            plt.close(fig)
