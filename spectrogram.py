import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy import signal

class Spectrogram:
    def __init__(self,wav_dir):
        self.wav_dir = wav_dir

    def Spectro(self,row,start):
        audio_multi, sr = librosa.load(self.wav_dir+row['WAV'],sr = None,offset=start-5,duration=60,mono=False)
        audio = audio_multi[int(row["Channel"])-1]
        freqs, _, spec = signal.spectrogram(audio, fs = sr, window = signal.windows.gaussian(sr//8,sr//48), nperseg= sr//8, noverlap=int(sr//8*0.97),nfft=sr//3)
        spec[spec == 0] = np.finfo(float).eps
        spec = 10*np.log10(spec) + np.abs(np.min(10*np.log10(spec)))
        return spec, freqs

    def Plot(self,call_group, start,freq_range,c,ii,srt,band_path,long_path,short_path):
        # long path plots
        fig_main_lp, ax_main_lp = plt.subplots(c,1,figsize=(27, 18),clear=True,sharex=True,sharey=True,layout='constrained',dpi = 200)
        # short path plots
        fig_main_sp, ax_main_sp = plt.subplots(1,c,figsize=(14, 12),clear=True,sharey=True,layout='constrained',dpi = 200)
        # bandpass plots
        fig_main_bp, ax_main_bp = plt.subplots(1,c,figsize=(14, 12),clear=True,sharey=True,layout='constrained',dpi = 200)
        for index, row in call_group.iterrows():
            spec, freqs = Spectrogram.Spectro(self,row,start)
            fig_lp = plt.figure(figsize =(12,5), frameon=False, dpi=200)
            canvas = fig_lp.canvas
            # get and work with the current axis of this subplot
            ax_lp = fig_lp.gca()
            ax_lp.imshow(spec, aspect='auto',origin='lower',cmap= "binary",vmin = 70, vmax = 120, extent = [start-5,start+55,np.min(freqs),np.max(freqs)])
            ax_lp.axis('off')
            # print("\n The vertical red line in the long_path plot is row[Begin Time] * len(freqs) on x axis which is: ", [row['Begin Time(s)']]*len(freqs), "\n")
            ax_lp.plot([row['Begin Time (s)']] * len(freqs), freqs, 'r--',linewidth=2)  
            fig_lp.tight_layout(pad=0)
            ax_lp.axis('off')
            ax_lp.margins(0)
            canvas.draw() 
            image_from_plot = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            image = image_from_plot.reshape(canvas.get_width_height()[::-1] + (3,))
            plt.close(fig_lp)
            ax_main_lp[index].imshow(image[::-1],aspect='auto',origin='lower',extent = [0,60,np.min(freqs),np.max(freqs)])
            ax_main_lp[index].set_title("Channel " + str(int(row["Channel"])),fontsize = 30)
            ax_main_lp[index].set_xticks(np.linspace(0, 60,31))
            ax_main_lp[index].set_xticklabels(np.linspace(0, 60,31),fontsize=16)
            ax_main_lp[index].set_yticks([*range(int(np.min(freqs)), int(np.max(freqs)),250)])
            ax_main_lp[index].set_yticklabels([*range(int(np.min(freqs)), int(np.max(freqs)),250)],fontsize=16)
            fig_sp = plt.figure(figsize =(1,4), frameon=False, dpi=200)
            canvas_sec = fig_sp.canvas
            ax_sp = fig_sp.gca()
            ax_sp.imshow(spec[:,int((row['Begin Time (s)']-(start-5)-2)*np.shape(spec)[1]/60):int((row['Begin Time (s)']-(start-5)+2)*np.shape(spec)[1]/60)], aspect='auto',origin='lower',cmap= "binary",vmin = 70, vmax = 120, extent = [row['Begin Time (s)']-1,row['Begin Time (s)']+1,np.min(freqs),np.max(freqs)])
            ax_sp.axis('off')
            ax_sp.plot([row['Begin Time (s)']] * len(freqs), freqs, 'r--',linewidth=0.8)
            fig_sp.tight_layout(pad=0)
            ax_sp.axis('off')
            ax_sp.margins(0)
            canvas_sec.draw() 
            image_from_plot_sec = np.frombuffer(canvas_sec.tostring_rgb(), dtype=np.uint8)
            image_sec = image_from_plot_sec.reshape(canvas_sec.get_width_height()[::-1] + (3,))
            plt.close(fig_sp)
            ax_main_sp[index].imshow(image_sec[::-1],aspect='auto',origin='lower',extent = [np.round(row['Begin Time (s)']-(start-5)-2,2),np.round(row['Begin Time (s)']-(start-5)+2,2),np.min(freqs),np.max(freqs)])   
            ax_main_sp[index].set_title("Channel " + str(int(row["Channel"])),fontsize = 24)
            ax_main_sp[index].set_xticks(np.linspace(row['Begin Time (s)']-(start-5)-2, row['Begin Time (s)']-(start-5)+2,5))
            ax_main_sp[index].set_xticklabels(['{:.1f}'.format(a) for a in np.linspace(row['Begin Time (s)']-(start-5)-2, row['Begin Time (s)']-(start-5)+2,5)],fontsize=12)
            ax_main_sp[index].set_yticks([*range(int(np.min(freqs)), int(np.max(freqs)),250)])
            ax_main_sp[index].set_yticklabels([*range(int(np.min(freqs)), int(np.max(freqs)),250)],fontsize=12)
            fig_bp = plt.figure(figsize =(1,4), frameon=False, dpi=200)
            canvas_tri = fig_bp.canvas
            ax_bp = fig_bp.gca()
            ax_bp.imshow(spec[:,int((row['Begin Time (s)']-(start-5)-2)*np.shape(spec)[1]/60):int((row['Begin Time (s)']-(start-5)+2)*np.shape(spec)[1]/60)], aspect='auto',origin='lower',cmap= "binary",vmin = 70, vmax = 120, extent = [row['Begin Time (s)']-1,row['Begin Time (s)']+1,np.min(freqs),np.max(freqs)])
            ax_bp.axis('off')
            ax_bp.plot([row['Begin Time (s)']] * len(freqs), freqs, 'r--',linewidth=0.8)
            ax_bp.plot([row['Begin Time (s)']-1,row['Begin Time (s)']+1], [freq_range["High"].values[ii]] * len([row['Begin Time (s)']-1,row['Begin Time (s)']+1]), 'b--',linewidth=0.8)
            ax_bp.plot([row['Begin Time (s)']-1,row['Begin Time (s)']+1], [freq_range["Low"].values[ii]] * len([row['Begin Time (s)']-1,row['Begin Time (s)']+1]), 'b--',linewidth=0.8)        
            fig_bp.tight_layout(pad=0)
            ax_bp.axis('off')
            ax_bp.margins(0)
            canvas_tri.draw() 
            image_from_plot_tri = np.frombuffer(canvas_tri.tostring_rgb(), dtype=np.uint8)
            image_tri = image_from_plot_tri.reshape(canvas_tri.get_width_height()[::-1] + (3,))  
            plt.close(fig_bp)
            ax_main_bp[index].imshow(image_tri[::-1],aspect='auto',origin='lower',extent = [np.round(row['Begin Time (s)']-(start-5)-2,2),np.round(row['Begin Time (s)']-(start-5)+2,2),np.min(freqs),np.max(freqs)])   
            ax_main_bp[index].set_title("Channel " + str(int(row["Channel"])),fontsize = 24)
            ax_main_bp[index].set_xticks(np.linspace(row['Begin Time (s)']-(start-5)-2, row['Begin Time (s)']-(start-5)+2,5))
            ax_main_bp[index].set_xticklabels(['{:.1f}'.format(a) for a in np.linspace(row['Begin Time (s)']-(start-5)-2, row['Begin Time (s)']-(start-5)+2,5)],fontsize=12)
            ax_main_bp[index].set_yticks([*range(int(np.min(freqs)), int(np.max(freqs)),250)])
            ax_main_bp[index].set_yticklabels([*range(int(np.min(freqs)), int(np.max(freqs)),250)],fontsize=12)
            fig_main_lp.supxlabel('Time (s)',fontsize = 24)
            fig_main_lp.supylabel('Frequency (Hz)',fontsize = 24)
            fig_main_lp.savefig(long_path.joinpath('spec_' +str(ii+1) +'_range_'+str(srt+1)+'-'+str(srt+c)+'_long.png'))
            plt.close(fig_main_lp)
            fig_main_sp.supxlabel('Time (s)',fontsize = 18)
            fig_main_sp.supylabel('Frequency (Hz)',fontsize = 18)
            fig_main_sp.savefig(short_path.joinpath('spec_' +str(ii+1) +'_range_'+str(srt+1)+'-'+str(srt+c)+'_short.png'))
            plt.close(fig_main_sp)
            fig_main_bp.supxlabel('Time (s)',fontsize = 18)
            fig_main_bp.supylabel('Frequency (Hz)',fontsize = 18)
            fig_main_bp.savefig(band_path.joinpath('spec_' +str(ii+1) +'_range_'+str(srt+1)+'-'+str(srt+c)+'_bandpass.png'))
            plt.close(fig_main_bp)
