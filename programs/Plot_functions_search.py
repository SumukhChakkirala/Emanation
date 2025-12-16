import matplotlib.pyplot as plt
import os

# Use env var EMANATION_RESULTS_DIR if set; otherwise default to project Results folder
NEW_SAVE_FOLDER = os.environ.get('EMANATION_RESULTS_DIR', os.path.join(os.getcwd(), 'Results'))
os.makedirs(NEW_SAVE_FOLDER, exist_ok=True)
import os
import numpy as np
import matplotlib
import numpy as np
# import matplotlib.pyplot as plt
from scipy.signal import square, cheby1
from scipy.signal import resample
# from EstimatePeaks import *
# We plot PSD, all identified peaks in red, positive peaks for specific harmonics series in green
import os
import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import kaiser
from EstimatePeaks_search import *
# from EstimateHarmonic import *
# from EstimateHarmonic_LowFreq_ParametricSearch import EstimateHarmonic_LowFreq

from scipy import signal
import pickle
import yaml
from scipy.signal.windows import kaiser

# from Plot_functions import *


# import matplotlib
# import sys
# sys.path.insert(1,'/home/gelu/Desktop/Documents/Emanations/Emanations/PythonScript/')


# ###################################################################################################################
# ###################################################################################################################

# Read from YAML config file


# def load_config(
#         yaml_filename="/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Emanations/synapse_emanation.yaml"):
#     with open(yaml_filename, 'r') as config_file:
#         config = yaml.safe_load(config_file)
#     #     global SF_eman, EF_eman, f_step, dur_ensemble, perc_overlap, min_duration, min_samprate, min_peaks_detect, \
#     #         numpeaks_crossthresh, kaiser_beta, numtaps, fs_lh, wt_meas_pred_hh, wt_meas_pred_lh, p_hh, p_lh, \
#     #         gb_thresh_hh, gb_thresh_lh
#
#     # ###################################################################################################################
#     # ###################################################################################################################
#     # # Candidates for config files
#     SF_eman = config['EmanationDetection']['SF_eman']
#     EF_eman = config['EmanationDetection']['EF_eman']
#     f_step = config['EmanationDetection']['f_step']
#
#     dur_ensemble = config['EmanationDetection'][
#         'dur_ensemble']  # duration of each ensemble ( time slices of a long sequence) over whose FFT,
#     # we average over.
#     perc_overlap = config['EmanationDetection']['perc_overlap']  # percentage overlap between successive time slices
#     # kaiser_beta = config['EmanationDetection']['kaiser_beta']
#     min_duration = config['EmanationDetection']['min_duration']
#
#     min_samprate = config['EmanationDetection']['min_samprate']
#     min_peaks_detect = config['EmanationDetection']['min_peaks_detect']
#     numpeaks_crossthresh = config['EmanationDetection'][
#         'numpeaks_crossthresh']  # we detect atleast these many peaks as part of harmonic series above an SNR threshold.
#
#     # tran_width = 0.001 # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiserord.html
#     # ripple = 65  # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiserord.html
#     # # numtaps, beta = signal.kaiserord(ripple, tran_width)
#     kaiser_beta_hh = config['EmanationDetection']['kaiser_beta_hh']
#     kaiser_beta_lh = config['EmanationDetection']['kaiser_beta_lh']
#     numtaps = config['EmanationDetection']['numtaps']
#     # small frequency harmonics parameter. We take this smaller bandwidth of the large spectrum, and process that to calculate lower
#     # valued harmonics. Taking K peaks from alrge spectrum will pick all harmonics including higher valued.
#     # This approach helps us focus on lower valued harmonics.
#     fs_lh = config['EmanationDetection']['fs_lh']
#
#     # Threshold used to set SNR_thresh for checking if peaks are atleast meeting this SNR.
#     # There is a gb_thresh inside of EstimatePeaks. We need to combine those likely.
#     gb_thresh_lh = config['EmanationDetection']['gb_thresh_lh']
#     gb_thresh_hh = config['EmanationDetection']['gb_thresh_hh']
#     log10_nan_check_limit = config['EmanationDetection']['log10_nan_check_limit']
#     num_runs_fundharmonic_estimate_perfreqslot = config['EmanationDetection'][
#         'num_runs_fundharmonic_estimate_perfreqslot']
#     dur = config['EmanationDetection']['dur']
#     lowerHarmonic_window = config['EmanationDetection']['lowerHarmonic_window']
#     ntimes_ns = config['EmanationDetection']['ntimes_ns']
#     Maxpeaks_hh = config['EmanationDetection']['Maxpeaks_hh']
#     Maxpeaks_lh = config['EmanationDetection']['Maxpeaks_lh']
#     return SF_eman, EF_eman, f_step, dur_ensemble, perc_overlap, min_duration, min_samprate, min_peaks_detect, \
#            numpeaks_crossthresh, kaiser_beta_hh, kaiser_beta_lh, numtaps, fs_lh, \
#            gb_thresh_hh, gb_thresh_lh, log10_nan_check_limit, num_runs_fundharmonic_estimate_perfreqslot, dur, \
#            lowerHarmonic_window, ntimes_ns, Maxpeaks_hh, Maxpeaks_lh
#     # ###################################################################################################################
#     # ###################################################################################################################

def plotpeaks(f_peaks, harmonic_list, f_range, fft_iq_dB, locpeaks, high_ff, zoom_perc_list, \
    plot_diffcolor_eachharmonic_flag,resultssavelocation, save_filename, setaxislim_flag = True):
      # Set this flag to true so taht we manually specify the y limits.
    # psd_val = WelchPSDEstimate(iq_feature, fs, dur_ensemble, perc_overlap, kaiser_beta)
    # psd_val_dB = 10 * np.log10(psd_val)
    # win_len = np.floor(dur_ensemble * fs).astype(int)
    # f_range_iq_plotval = np.arange(-f_step / 2, f_step / 2, fs / len(psd_val_dB))
    for zoom_perc in zoom_perc_list:
        x_lim_max = (zoom_perc / 100) * np.max(f_range)
        x_lim_min = -x_lim_max
        max_idx = np.argmin(np.abs(f_range - x_lim_max))
        min_idx = np.argmin(np.abs(f_range - x_lim_min))
        if max_idx > len(f_range):
            max_idx = len(f_range) - 1
        if min_idx < 0:
            min_idx = 0
        f_range_zoom = f_range[min_idx:max_idx]  # np.linspace(x_lim_min, x_lim_max, num=len(psd_val))
        psd_val_dB_zoom = fft_iq_dB[min_idx:max_idx]
        if high_ff:
            # f_range_updated = np.divide(f_range_zoom, 1e6)
            f_range_zoom_scaled = np.divide(f_range_zoom, 1e6)
            f_range_scaled = np.divide(f_range, 1e6)
            x_label = 'Frequency (MHz)'
        else:
            f_range_zoom_scaled = np.divide(f_range_zoom, 1e3)
            f_range_scaled = np.divide(f_range, 1e3)
            x_label = 'Frequency (kHz)'
        plt.figure()
        # Find peaks and its indices within f_range which within specified frequency range - for zooming.
        peak_sort_idx = np.argsort(f_range[locpeaks])
        peaks = f_range[locpeaks][peak_sort_idx]
        fft_iq_dB_peaks = fft_iq_dB[locpeaks][peak_sort_idx]
        min_closest_idx = np.argmin(np.abs(peaks - x_lim_min))
        max_closest_idx = np.argmin(np.abs(peaks - x_lim_max))
        if peaks[min_closest_idx] <= x_lim_min:
            min_closest_idx = min_closest_idx + 1
        if peaks[max_closest_idx] >= x_lim_max:
            max_closest_idx = max_closest_idx - 1
        if high_ff:
            peaks_scaled = np.divide(peaks, 1e6)
        else:
            peaks_scaled = np.divide(peaks, 1e3)
        marker_idx = 0
        plt.plot(f_range_zoom_scaled, psd_val_dB_zoom, label='_nolegend_')
        if plot_diffcolor_eachharmonic_flag == True:
            marker_list = ['s', '^', 'o', '8', '+', 'D']
            marker_color = ['g', 'r', 'c', 'm', 'g', 'k', 'w']
            filename_peaks = '_peaksOverlayDiffHarmonic'

            legend_text = [' ']
            for pitch_freq in list(harmonic_list.keys()):
                if (pitch_freq < 1e3) and (not high_ff):
                    legend_label = 'Pitch: ' + str(int(pitch_freq)) + ' Hz.'
                elif (pitch_freq < 1e6) and (pitch_freq >= 1e3) and high_ff:
                    legend_label = 'Pitch: ' + str(int(pitch_freq / 1e3)) + ' kHz.'
                elif (pitch_freq >= 1e6) and high_ff:
                    legend_label = 'Pitch: ' + str(int(pitch_freq / 1e6)) + ' MHz.'
                if high_ff and pitch_freq < 2000:
                    continue
                elif (not high_ff) and pitch_freq > 2000:
                    continue
                else:
                    harmonic_freq = harmonic_list[pitch_freq]['components_relativefreq']
                    for harmonic_freq_val in harmonic_freq:
                        harmonic_freq = np.append(harmonic_freq, -harmonic_freq_val)
                        # print(harmonic_freq)
                    harmonic_freq = np.sort(harmonic_freq)
                    min_idx_hf = np.argmin(np.abs(harmonic_freq - x_lim_min))
                    max_idx_hf = np.argmin(np.abs(harmonic_freq - x_lim_max))
                    if harmonic_freq[max_idx_hf] > x_lim_max:
                        max_idx_hf = max_idx_hf - 1
                    if harmonic_freq[min_idx_hf] < x_lim_min:
                        min_idx_hf = min_idx_hf + 1
                    pitch_freq_zoom = harmonic_freq[min_idx_hf:max_idx_hf + 1]
                    idx_closest = []
                    for pitch_freq_zoom_comp in pitch_freq_zoom:
                        idx_closest.append(np.argmin(np.abs(peaks - pitch_freq_zoom_comp)))
                    plt.scatter(peaks_scaled[idx_closest], fft_iq_dB_peaks[idx_closest], marker=marker_list[marker_idx],
                                color=marker_color[marker_idx], label=legend_label)
                    marker_idx = marker_idx + 1
            plt.xlabel(x_label, fontsize=40)

        else:
            marker_list = ['o']
            marker_color = ['r']
            filename_peaks = '_peaksOverlaySameHarmonic'
            plt.scatter(peaks_scaled[min_closest_idx: max_closest_idx + 1],
                        fft_iq_dB_peaks[min_closest_idx: max_closest_idx + 1], marker=marker_list[marker_idx],
                        color=marker_color[marker_idx])

            # plt.xlabel(x_label, fontsize=40)
        # plt.title("Power spectral density", fontsize=40)

        if zoom_perc == 100:
            plt.ylabel("Power (dB)", fontsize=40)
        plt.xticks(fontsize=40)
        # plt.yticks(fontsize=40)
        plt.legend(fontsize=40)
        # fig.tight_layout()

        if setaxislim_flag:
            if plot_diffcolor_eachharmonic_flag == True:
                if high_ff:
                    plt.yticks(np.arange(-185, -140, 10), fontsize=40)
                else:
                    plt.yticks(np.arange(-210, -140, 20), fontsize=40)
            else:
                if high_ff:
                    plt.yticks(np.arange(-185, -140, 10),fontsize=40)
                else:
                    plt.yticks(np.arange(-210, -140, 20),fontsize=40)


        plt.gcf().set_size_inches(15, 10)
        plt.subplots_adjust(left=0.12,
                            bottom=0.1,
                            right=0.9,
                            top=0.95,
                            wspace=0.25,
                            hspace=0.25)
        plt.tight_layout()
    save_path = os.path.join(NEW_SAVE_FOLDER, 'PSD_' + save_filename + '_zp_' + str(zoom_perc) + filename_peaks + '_len_breadth_0.67ratio.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=.01)
        # plt.show()
    plt.close()

def spectrogram_emanations(iq, fs, CF_snapshot, resultssavelocation, save_filename, cmap, scenario):

    # iq3 = np.multiply(np.conjugate(iq), iq)
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(10, 5)
    cmap = plt.get_cmap(cmap)
    # vmin = 20*np.log10(np.max(x)) - 40  # hide anything below -40 dBc
    cmap.set_under(color='k', alpha=None)

    NFFT = 4096
    NFFT = int(NFFT)
    noverlap = int(NFFT // 2)
    pxx, freq, t, cax = ax.specgram(iq, Fs=fs,
                                    NFFT=NFFT, noverlap=noverlap, cmap=cmap, vmin=-180)
    # fig.colorbar(cax)
    # plt.title("Spectrogram", fontsize=40)
    fig.colorbar(cax).set_label('Power spectral density (dB/Hz)')
    # scale = 1e3                     # KHz
    ticks = matplotlib.ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(int(CF_snapshot / 1e6) + y / 1e6))
    ax.yaxis.set_major_formatter(ticks)

    ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e3))
    ax.xaxis.set_major_formatter(ticks)

    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True,
                    length=8, width=3, direction='out')
    # plt.legend(legend_text,fontsize=36)
    plt.xlabel("Time (ms) ", fontsize=14)
    plt.ylabel("IQ capture Frequency (MHz)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # # plt.title("Accuracy versus SNR plot. Effect of bandwidth on performance.")
    plt.tight_layout()
    save_path = os.path.join(NEW_SAVE_FOLDER, 'Spectrogram_' + 'Scenario_' + scenario + save_filename + '.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=.01)
    plt.close()




def fft_iq_iqpower_plot(iq, f_step, iq_feature, fs):
    iq3 = np.multiply(np.conjugate(iq), iq)

    f_range_iq_plotval = np.arange(-f_step / 2, f_step / 2, fs / len(iq))
    fft_iq_plotval = 20 * np.log10(np.fft.fftshift(np.abs(np.fft.fft(iq))))

    f_range_iqfeature_plotval = np.arange(-f_step / 2, f_step / 2, fs / len(iq_feature))
    fft_iqfeature_plotval = 10 * np.log10(np.fft.fftshift(np.abs(np.fft.fft(iq3))))

    plt.figure(1)


    save_plot_location = '/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Emanations/Plots/April12th/'

    zoom_perc = 10
    zoom_len = int(len(iq) / zoom_perc)
    zoom_limit_idx = np.arange(int(len(iq) / 2) - zoom_len, int(len(iq) / 2) + zoom_len)
    x_axis_freqval_MHz = f_range_iq_plotval[zoom_limit_idx] / 1e6
    plt.subplot(211)
    plt.plot(x_axis_freqval_MHz, fft_iq_plotval[zoom_limit_idx])
    plt.title("FFT over IQ", fontsize=40)
    # plt.tick_params(top=False,bottom=True,left=True,right=False,labelleft=True,labelbottom=True,length=8,width=3, direction='out')
    # plt.legend(legend_text,fontsize=36)
    # plt.xlabel("Frequency (MHz)" ,fontsize=20)
    plt.ylabel("Power (dB)", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(212)
    zoom_len_2 = int(len(iq_feature) / zoom_perc)
    zoom_limit_idx2 = np.arange(int(len(iq_feature) / 2) - zoom_len_2, int(len(iq_feature) / 2) + zoom_len_2)
    x_axis_freqval_MHz = f_range_iqfeature_plotval[zoom_limit_idx2] / 1e6
    plt.plot(x_axis_freqval_MHz, fft_iqfeature_plotval[zoom_limit_idx2])
    plt.title("FFT over IQ power", fontsize=40)
    # plt.tick_params(top=False,bottom=True,left=True,right=False,labelleft=True,labelbottom=True,length=8,width=3, direction='out')
    # plt.legend(legend_text,fontsize=36)
    plt.xlabel("Frequency (MHz)", fontsize=20)
    plt.ylabel("Power (dB)", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(left=0.12,
                        bottom=0.1,
                        right=0.9,
                        top=0.95,
                        wspace=0.25,
                        hspace=0.25)
    # plt.savefig(save_plot_location + 'EmanationsFromDesktop_CineBench_FFTIQ_IQpower.png', format='png',
    #             bbox_inches='tight', pad_inches=.01)
    plt.show()

def plot_PSD(config_dict, iq_feature, fs, dur_ensemble, perc_overlap, kaiser_beta, f_range, resultssavelocation, save_filename,\
             high_ff, zoom_perc_list, option_processing, f_step, setaxislim_flag = True):
    #setaxislim_flag = True # Set this flag to true so taht we manually specify the y limits.


    if dur_ensemble == 0.1:
        w = kaiser(len(iq_feature), kaiser_beta)
        w /= np.sum(w)
        w_energy = (np.real(np.vdot(w, w))) / len(w)
        iq_w = np.multiply(iq_feature, w)
        fft_iq = np.fft.fftshift(np.abs(np.fft.fft(iq_w)))
        # fft_iq_slice = np.fft.fftshift(np.abs(np.fft.fft(iq_feature)))
        psd_val = np.multiply(fft_iq, fft_iq) / (w_energy * len(w))
    else:
    # fft_power_dB = 10 * np.log10(fft_power)
        psd_val = WelchPSDEstimate(iq_feature, fs, dur_ensemble, perc_overlap, kaiser_beta, config_dict)

    # psd_val = WelchPSDEstimate(iq_feature, fs, dur_ensemble, perc_overlap, kaiser_beta, config_dict)
    psd_val_dB = 10 * np.log10(psd_val)
    # f_step = int(25e6)
    win_len = len(psd_val_dB)
    f_range = np.arange(min(f_range), max(f_range), (max(f_range)-min(f_range))/win_len)
    # win_len = np.floor(dur_ensemble * fs).astype(int)
    # f_range_iq_plotval = np.arange(-f_step / 2, f_step / 2, fs / len(psd_val_dB))
    for zoom_perc in zoom_perc_list:
        x_lim_max = (zoom_perc/100)*np.max(f_range)
        x_lim_min = -x_lim_max
        max_idx = np.argmin(np.abs(f_range - x_lim_max))
        min_idx = np.argmin(np.abs(f_range - x_lim_min))
        if max_idx >len(f_range):
            max_idx = len(f_range) - 1
        if min_idx < 0:
            min_idx = 0


        f_range_zoom = f_range[min_idx:max_idx]#np.linspace(x_lim_min, x_lim_max, num=len(psd_val))
        psd_val_dB_zoom = psd_val_dB[min_idx:max_idx]
        if high_ff:
            f_range_updated = np.divide(f_range_zoom,1e6)
            x_label = 'Frequency (MHz)'

        else:
            f_range_updated = np.divide(f_range_zoom, 1e3)
            x_label = 'Frequency (kHz)'
        plt.plot(f_range_updated, psd_val_dB_zoom)
        # plt.title("Power/ spectral density", fontsize=40)
        # plt.tick_params(top=False,bottom=True,left=True,right=False,labelleft=True,labelbottom=True,length=8,width=3, direction='out')
        # plt.legend(legend_text,fontsize=36)
        if  option_processing in ['_withPreprocessing', 'withPreprocessing']:
            plt.xlabel(x_label, fontsize=40)
        if zoom_perc == 100:
            plt.ylabel("Power (dB)", fontsize=40)
        plt.xticks(fontsize=40)
        # plt.yticks(fontsize=40)

        # ticks = matplotlib.ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(int(CF_snapshot / 1e6) + y / 1e6))
        # ax.yaxis.set_major_formatter(ticks)
        #
        # ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e3))
        # ax.xaxis.set_major_formatter(ticks)
        if setaxislim_flag:
            # if zoom_perc == 100:
            if high_ff:
                if option_processing in ['_withPreprocessing', 'withPreprocessing']:
                    # plt.ylim([-140,-180])
                    plt.yticks(np.arange(-185, -145, 10), fontsize=40)
                else:
                    # plt.ylim([-130, -190])
                    plt.yticks(np.arange(-195, -50, 20), fontsize=40)
            else:
                if option_processing in ['_withPreprocessing', 'withPreprocessing']:
                    # plt.ylim([-150,-230])
                    plt.yticks(np.arange(-230,-145,20), fontsize=40)
                else:
                    # plt.ylim([-70, -170])
                    plt.yticks(np.arange(-170, -70, 20), fontsize=40)
        else:
            plt.yticks(fontsize=40)


        plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True,
                        length=8, width=3, direction='out')

        # fig.tight_layout()
        plt.gcf().set_size_inches(15, 10)
        plt.subplots_adjust(left=0.12,
                            bottom=0.1,
                            right=0.9,
                            top=0.95,
                            wspace=0.25,
                            hspace=0.25)
        # plt.savefig(save_plot_location + 'EmanationsFromDesktop_CineBench_FFTIQ_IQpower.png', format='png',
        #             bbox_inches='tight', pad_inches=.01)
        plt.tight_layout()
    save_path = os.path.join(NEW_SAVE_FOLDER, 'PSD_' + save_filename + '_zoomperc_' + str(zoom_perc)+ '_len_breadth_0.67ratio.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=.01)

    plt.close()

def Objfunc_ErrvsFreq_plot(harmonic_list, CF_p1_p2, resultssavelocation):
    harmonic_freq_list = list(harmonic_list.keys())
    for harmonic_freq in harmonic_freq_list:
        plt.figure()
        obj_func_val = harmonic_list[harmonic_freq]['obj_func_val']
        # plt.subplot(2,1,1)
        # {'Err_total': Err_total, 'Err_total_fine': Err_total_fine, 'freq_search': freq_search, \
        #  freq_search_fine: freq_search_fine}
        if len(obj_func_val['freq_search_fine']) > 0:
            min_idx = np.argmin(obj_func_val['Err_total_fine'])
            min_freq = obj_func_val['freq_search_fine'][min_idx]
            min_objfunc_val = obj_func_val['Err_total_fine'][min_idx]
            plt.plot(obj_func_val['freq_search'], obj_func_val['Err_total'])
            plt.plot(obj_func_val['freq_search_fine'], obj_func_val['Err_total_fine'], '--r')
            plt.scatter(min_freq, min_objfunc_val, marker='o',color='y')
            plt.legend(['Coarse', 'Fine'],fontsize=14)
        else:
            min_idx = np.argmin(obj_func_val['Err_total'])
            min_freq = obj_func_val['freq_search'][min_idx]
            min_objfunc_val = obj_func_val['Err_total'][min_idx]
            plt.scatter(min_freq, min_objfunc_val, marker='o', color='y')
            plt.plot(obj_func_val['freq_search'], obj_func_val['Err_total'])
            # plt.plot(harmonic_list[harmonic_freq]['freq_search_fine'], harmonic_list[harmonic_freq]['Err_total_fine'])
            # plt.legend(['Coarse', 'Fine'])
        # plt.title("Objective function.", fontsize=40)
        # plt.tick_params(top=False,bottom=True,left=True,right=False,labelleft=True,labelbottom=True,length=8,width=3, direction='out')
        # plt.legend(legend_text,fontsize=36)
        plt.xlabel('Candidate pitch frequency in Hz', fontsize=20)
        plt.ylabel("Objective function error.", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # fig.tight_layout()
        plt.gcf().set_size_inches(15, 10)
        plt.subplots_adjust(left=0.12,
                            bottom=0.1,
                            right=0.9,
                            top=0.95,
                            wspace=0.25,
                            hspace=0.25)
        # plt.savefig(save_plot_location + 'EmanationsFromDesktop_CineBench_FFTIQ_IQpower.png', format='png',
        #             bbox_inches='tight', pad_inches=.01)
        plt.tight_layout()
    save_path = os.path.join(NEW_SAVE_FOLDER, 'Errorfunction_' + CF_p1_p2 + '_pitchFreq_' + str(int(harmonic_freq))+ '.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=.01)
    plt.close()
        # plt.subplot(2,1,2)

class EmanationInput:
    def __init__(self, input_info):
        # meta_data is a dictionary with details of the IQ snapshot such as sample rate, center frequency, timestamp, hardware ID.
        self.iq = input_info["iq"]
        self.fs = input_info["sample_rate"]
        self.CF = input_info["center_freq"]
        self.TS = input_info["time_duration"]
        self.HW_ID = input_info["hw_id"]
        self.phase = input_info["test_phase"]
        self.folder = input_info["path"]

# def plotMainfunction(input_info):
#     (SF_eman, EF_eman, f_step, dur_ensemble, perc_overlap, min_duration, min_samprate, min_peaks_detect, \
#      numpeaks_crossthresh, kaiser_beta_hh, kaiser_beta_lh, numtaps, fs_lh, \
#      gb_thresh_hh, gb_thresh_lh, log10_nan_check_limit, num_runs_fundharmonic_estimate_perfreqslot, dur, \
#      lowerHarmonic_window, ntimes_ns, Maxpeaks_hh, Maxpeaks_lh) = load_config()
#     emanationInputObj = EmanationInput(input_info)
#     save_plot_location = '/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Emanations/Plots/'
#     fs = emanationInputObj.fs  # emanationInputDict['SampleRate']
#     harmonic_freq_list = []
#     harmonic_SNR_list = []
#     iq = emanationInputObj.iq  # emanationInputDict['FileName']
#     if len(iq) / fs > 0.1:
#         iq = iq[0:int(dur * fs)]
#     d = (len(iq) / fs)
#     cutoff = f_step / 2
#     CF_snapshot = emanationInputObj.CF
#     win_len = np.floor(dur_ensemble * f_step).astype(int)
#     N = len(iq)
#     Ts = 1 / fs


    # spectrogram_emanations(iq, fs, CF_snapshot)
    # fft_iq_iqpower_plot(iq, f_step, iq_feature, fs)
    # plot_PSD(iq_feature, fs, dur_ensemble, perc_overlap, kaiser_beta_hh)

def OOK_plot():
    # code is from CHATGPT that converted my MATLAB code.
    # see for the matlab code from this locationk: Emanations_Journal_code/Emanations/Plots/Code
    np.random.seed(42)  # For reproducibility

    # Parameters
    randomized_data = 0
    noiseBwArr = [500e3]
    NRZ = 0
    if NRZ:
        dutyCyclePercentArr = [100]
    else:
        dutyCyclePercentArr = [70]

    fundamentalHarmonicArr = [500]
    f_IF_arr = [4e6]
    snr_db_arr = [20]

    on_level = 1
    off_level = 0

    totalSignaltime = 0.1
    plotFlag = 0
    Fs = 10e6
    sigma = 1
    sampleSignal = sigma * np.random.randn(int(totalSignaltime * Fs))

    for bwIdx in range(len(noiseBwArr)):
        noise_bw = noiseBwArr[bwIdx]

        # Carrier Gen
        duty = dutyCyclePercentArr[bwIdx]
        fundamentalharmonic = fundamentalHarmonicArr[bwIdx]
        sampleRate = noise_bw
        t = np.arange(0, totalSignaltime, 1 / sampleRate)
        carrier = square(2 * np.pi * fundamentalharmonic * t, duty)
        num_inforbits_totaltime = int(fundamentalharmonic * totalSignaltime)
        info = np.random.randint(2, size=num_inforbits_totaltime)
        num_samples_per_cycle = int(sampleRate / fundamentalharmonic)
        num_samples_per_cycle_riseTime = int((duty / 100) * num_samples_per_cycle)

        if randomized_data:
            for idx_info in range(len(info)):
                start_idx = (idx_info - 1) * num_samples_per_cycle
                end_idx = (idx_info - 1) * num_samples_per_cycle + num_samples_per_cycle_riseTime
                carrier[start_idx:end_idx] = info[idx_info]
        else:
            carrier[carrier > 0] = on_level
            carrier[carrier < 0] = off_level

        carrier = carrier / np.sqrt(np.mean(carrier ** 2))

        if plotFlag:
            f_axis = np.fft.fftshift(np.fft.fftfreq(len(carrier), d=1 / sampleRate))
            plt.figure()
            plt.plot(f_axis, np.abs(np.fft.fftshift(np.fft.fft(carrier))))
            plt.title("carrier fft")
            plt.xlabel('f axis')

        # Noise Gen
        numSamples = int(totalSignaltime * sampleRate)
        y = 10 ** (snr_db_arr[bwIdx] / 20) * np.random.randn(numSamples)

        if plotFlag:
            f_axis = np.fft.fftshift(np.fft.fftfreq(len(y), d=1 / sampleRate))
            plt.figure()
            plt.plot(f_axis, np.abs(np.fft.fftshift(np.fft.fft(y))))
            plt.title("wideband Noise")
            bw = noise_bw - 0.001
            b, a = cheby1(6, 10, bw / (sampleRate))
            filtered_noise = np.convolve(y, b, mode='same')

            f_axis = np.fft.fftshift(np.fft.fftfreq(len(filtered_noise), d=1 / sampleRate))
            plt.figure()
            plt.plot(f_axis, np.abs(np.fft.fftshift(np.fft.fft(filtered_noise))))
            plt.title("filtered wideband Noise")

        # Modulation
        filtered_noise = y
        ook = filtered_noise * carrier

        if plotFlag:
            f_axis = np.fft.fftshift(np.fft.fftfreq(len(ook), d=1 / sampleRate))
            plt.figure()
            plt.plot(f_axis, np.abs(np.fft.fftshift(np.fft.fft(ook))))
            plt.title("OOK wideband Noise")
            plt.xlabel('f_axis')

        # Resample
        ookResampled = resample(ook, int(Fs), t=resample(ook, int(Fs)).shape[0]) / (Fs / (4 * noise_bw))
        f_IF = f_IF_arr[bwIdx]
        ookResampled *= np.exp(1j * 2 * np.pi * f_IF * np.arange(0, totalSignaltime, 1 / Fs))

        if plotFlag:
            f_axis = np.fft.fftshift(np.fft.fftfreq(len(ookResampled), d=1 / Fs))
            plt.figure()
            plt.plot(f_axis, np.abs(np.fft.fftshift(np.fft.fft(ookResampled))))
            plt.title("OOKresampled wideband Noise")
            plt.xlabel('f_axis')

        sampleSignal += ookResampled

    plt.figure(1)
    f_axis = np.fft.fftshift(np.fft.fftfreq(len(sampleSignal), d=1 / Fs))
    plt.plot(f_axis, np.abs(np.fft.fftshift(np.fft.fft(sampleSignal)) / np.max(np.fft.fft(sampleSignal))))
    plt.title("OOK resampled summation wideband Noise - noiseBwArr = [500e3];duty= [50%];fcArr = [10e4]")
    plt.xlabel('f_axis')

    plt.figure(2)
    sampleSignal_power = np.sqrt(np.dot(sampleSignal, sampleSignal))
    sampleSignal_norm = sampleSignal / sampleSignal_power
    freq_axis_range = np.linspace(-Fs / 2, Fs / 2, len(sampleSignal_norm))
    plt.plot(freq_axis_range / 1e3, np.log10(np.fft.fftshift(np.abs(np.fft.fft(np.abs(sampleSignal_norm))))))
    plt.xlim([-20, 20])
    plt.ylim([0.324253829905057, 3.00461335564254])
    plt.xlabel('Freq in kHz', fontsize=24)
    plt.ylabel('Power (dB)', fontsize=24)
    plt.title('FFT over absolute of OOK signal.', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()


