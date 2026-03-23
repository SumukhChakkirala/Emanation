import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import kaiser
# import sys
# sys.path.insert(1,'/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Emanations/')
from EstimatePeaks_search import *
from EstimateHarmonic_search import *
# from EstimateHarmonic_LowFreq_ParametricSearch import EstimateHarmonic_LowFreq

from scipy import signal
import pickle
import yaml
from scipy.signal.windows import kaiser

from Plot_functions_search import *
# import matplotlib



# ###################################################################################################################
# ###################################################################################################################

# Read from YAML config file

#/home/vesathya/Emanations/Journal/Emanations_JournalCode/Emanations/synapse_emanation.yaml
# def load_config(config):
# #     with open(yaml_filename, 'r') as config_file:
# #         config = yaml.safe_load(config_file)
# #     global SF_eman, EF_eman, f_step, dur_ensemble, perc_overlap, min_duration, min_samprate, min_peaks_detect, \
# #         numpeaks_crossthresh, kaiser_beta, numtaps, fs_lh, wt_meas_pred_hh, wt_meas_pred_lh, p_hh, p_lh, \
# #         gb_thresh_hh, gb_thresh_lh

#     # ###################################################################################################################
#     # ###################################################################################################################
#     # # Candidates for config files
#     SF_eman = config['EmanationDetection']['SF_eman']
#     EF_eman = config['EmanationDetection']['EF_eman']
#     f_step = config['EmanationDetection']['f_step']

#     dur_ensemble = config['EmanationDetection']['dur_ensemble']  # duration of each ensemble ( time slices of a long sequence) over whose FFT,
#     # we average over.
#     perc_overlap = config['EmanationDetection']['perc_overlap']  # percentage overlap between successive time slices
#     # kaiser_beta = config['EmanationDetection']['kaiser_beta']
#     min_duration = config['EmanationDetection']['min_duration']

#     min_samprate = config['EmanationDetection']['min_samprate']
#     min_peaks_detect = config['EmanationDetection']['min_peaks_detect']
#     numpeaks_crossthresh = config['EmanationDetection']['numpeaks_crossthresh'] # we detect atleast these many peaks as part of harmonic series above an SNR threshold.

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


#     # Threshold used to set SNR_thresh for checking if peaks are atleast meeting this SNR.
#     # There is a gb_thresh inside of EstimatePeaks. We need to combine those likely.
#     gb_thresh_lh = config['EmanationDetection']['gb_thresh_lh']
#     gb_thresh_hh = config['EmanationDetection']['gb_thresh_hh']
#     log10_nan_check_limit = config['EmanationDetection']['log10_nan_check_limit']
#     num_runs_fundharmonic_estimate_perfreqslot = config['EmanationDetection']['num_runs_fundharmonic_estimate_perfreqslot']
#     dur = config['EmanationDetection']['dur']
#     lowerHarmonic_window = config['EmanationDetection']['lowerHarmonic_window']
#     ntimes_ns = config['EmanationDetection']['ntimes_ns']
#     Maxpeaks_hh = config['EmanationDetection']['Maxpeaks_hh']
#     Maxpeaks_lh = config['EmanationDetection']['Maxpeaks_lh']
#     return SF_eman, EF_eman, f_step, dur_ensemble, perc_overlap, min_duration, min_samprate, min_peaks_detect, \
#         numpeaks_crossthresh, kaiser_beta_hh, kaiser_beta_lh, numtaps, fs_lh, \
#         gb_thresh_hh, gb_thresh_lh, log10_nan_check_limit, num_runs_fundharmonic_estimate_perfreqslot, dur, \
#            lowerHarmonic_window, ntimes_ns, Maxpeaks_hh, Maxpeaks_lh
    # ###################################################################################################################
    # ###################################################################################################################

class EmanationInput:
    def __init__(self, input_info):
        #meta_data is a dictionary with details of the IQ snapshot such as sample rate, center frequency, timestamp, hardware ID.
        self.iq = input_info["iq"]
        self.fs = input_info["sample_rate"]
        self.CF = input_info["center_freq"]
        self.scenario = input_info["scenario"]
        # self.TS = input_info["time_duration"]
        # self.HW_ID = input_info["hw_id"]
        # not using test_phase and hw_id and timestamp TS, therefore commented them on July 31st.
        # THey were useful only for IARPA project point of view.
        # self.phase = input_info["test_phase"]
        self.folder = input_info["path"]
        self.pythonfiles_location = input_info['pythonfiles_location']
#         self.config_file = input_info['config_dict']
        self.plot_flags = {'spectrogram': input_info['plot_dict']['spectrogram'],
                           'PSD': input_info['plot_dict']['PSD'],
                           'Objfunc_ErrvsFreq': input_info['plot_dict']['Objfunc_ErrvsFreq'], \
                           'peaks': input_info['plot_dict']['peaks']}
        self.CF_p1_p2 = input_info['CF_p1_p2']
        self.cmap = input_info['plot_dict']['cmap']
        self.PSD_plot_param = input_info['PSD_plot_param']
# class EmanationOutput:
#     def __init__(self, freqlist, SNRlist, Eman_Flag):
#         self.freqlist = freqlist
#         self.SNRlist = SNRlist
#         self.Eman_Flag = Eman_Flag


#####################################################################################################################
# 1. We take in an IQ snapshot say 100 MHz, 100 ms data and slice it into 20 Mhz slices using kaiser filter.
# 2. Each slice, we esimate welch PSD - periodogram averaging.
# 3. Output of averaging passed through peak finding to find dominant peaks.
# 4. Dominant peak frequency and SNR estimates are imrpoved via parabolic interpolation.
# 5. Peak frequencies and SNR estimates are fed to EstimateHarmonic function.
# Steps 1 code is in EmanationDetection function in EmanationDetection.py
# Steps 2, 3 and 4 code is in WelchPSD, EstimatePeaks, FreqEstiamteParabolic functions are in EstimatePeaks.py
# Steps 5 code is in EstimateHarmonic.py.
#####################################################################################################################
def EmanationDetection(input_info, config_dict):
    SF_eman, EF_eman, f_step, dur_ensemble, perc_overlap, min_duration, min_samprate, min_peaks_detect, \
        numpeaks_crossthresh, kaiser_beta_hh, kaiser_beta_lh, numtaps, fs_lh,\
        gb_thresh_hh, gb_thresh_lh, log10_nan_check_limit, num_runs_fundharmonic_estimate_perfreqslot, dur,\
     lowerHarmonic_window, ntimes_ns, Maxpeaks_hh, Maxpeaks_lh = config_dict['EmanationDetection']['SF_eman'], config_dict['EmanationDetection']['EF_eman'], config_dict['EmanationDetection']['f_step'], config_dict['EmanationDetection']['dur_ensemble'], config_dict['EmanationDetection']['perc_overlap'], config_dict['EmanationDetection']['min_duration'], config_dict['EmanationDetection']['min_samprate'], config_dict['EmanationDetection']['min_peaks_detect'], \
        config_dict['EmanationDetection']['numpeaks_crossthresh'], config_dict['EmanationDetection']['kaiser_beta_hh'], config_dict['EmanationDetection']['kaiser_beta_lh'], config_dict['EmanationDetection']['numtaps'], config_dict['EmanationDetection']['fs_lh'],\
        config_dict['EmanationDetection']['gb_thresh_hh'], config_dict['EmanationDetection']['gb_thresh_lh'], config_dict['EmanationDetection']['log10_nan_check_limit'], config_dict['EmanationDetection']['num_runs_fundharmonic_estimate_perfreqslot'], config_dict['EmanationDetection']['dur'],\
     config_dict['EmanationDetection']['lowerHarmonic_window'], config_dict['EmanationDetection']['ntimes_ns'], config_dict['EmanationDetection']['Maxpeaks_hh'], config_dict['EmanationDetection']['Maxpeaks_lh']
    
#     load_config(input_info['config_dict'])
    emanationInputObj = EmanationInput(input_info)
    # fund_freq = SNR = []
#     fs = emanationInputObj.fs

    # save_plot_location = '/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Emanations/Plots/'
    # plot_flag_spectrogram = False
    #
    # plot_flag_FFToverIQ_IQFeature = False
    # plot_PSD_flag = False
    # plotpeaks_flag = False

    fs = emanationInputObj.fs#emanationInputDict['SampleRate']
    harmonic_freq_list  = []
    harmonic_SNR_list = []
    # iq_whole = np.fromfile(emanationInputObj.iq, dtype=np.csingle)
    #iq_whole = emanationInputObj.iq
    iq = emanationInputObj.iq#emanationInputDict['FileName']
    # if len(iq)/fs > 0.1:
    #     iq = iq[0:int(dur*fs)]
    d = (len(iq)/fs)
    cutoff = f_step / 2

    # We only process emanations within a range of frequencies start freq. SF_eman to end freq. EF_eman.
    # everything else, we do not process and store an empty dictioanry along meta-data. This is to save time
    #. Since emanation block takes time to process each snapshot, this technique can help focus compute on
    # snapshots that are likely to have most of the emanations.
     # step size of each frequency slot. We process emanation per frequency slot and estimate fund_harmonics, components and SNR.
    CF_snapshot = emanationInputObj.CF
    # SF_snapshot = CF_snapshot - fs/2
    # n_frs = (EF_eman - SF_eman)/f_step # number of frequency slots
    # d = 10e-3  # in mS
    #print("Duration of data processing is hardcoded to " + str(int(1e3*d)) + " mS. Should be changed as per need in actual deployment")

    # we only use portion of the samples from the file
    # iq_samplen = int(d*fs)
    # iq = iq_whole[0:iq_samplen]
    #print(iq)

    win_len = np.floor(dur_ensemble * f_step).astype(int)

    N = len(iq)
    Ts = 1 / fs


    # We process IQ snpashot only if meets the following conditions:
    # 1. IQ snapshot duration is a minimum of min_duration (50 ms)
    # 2. Sample rate is a minumum of min_samprate (25 MHz)
    # 3. Frequency of IQ snapshot is atleast less than EF_Eman.
    harmonic_complete_list = {}
    Snapshot_processed = False # This flag is set to indicate if IQ snapshot is processed or not. If not processed, empty rsults dictioanry is saved

    # in the pickle file alongside other meta-data of snapshot.
    if (d > min_duration) and (fs > min_samprate) and (CF_snapshot <= (EF_eman - fs/2)):
        Snapshot_processed = True

        # start frequency and end frequency of a freq. slot. For eg: span of freq. slot 0 is 100 to 120 MHz, 100 is SF, 120 EF.
        SF_freqslot, EF_freqslot = -fs / 2, -fs / 2 + f_step
        while EF_freqslot <= fs / 2:
            CF_freqslot = (SF_freqslot + EF_freqslot) / 2
            # frs_id = 0 + int((SF_freqslot + CF_snapshot - SF_eman)/f_step) # frequency slot ID
            harmonic_list = {}
            # cutoff = f_step/2
            shift_freq = -1 * (SF_freqslot + EF_freqslot) / 2
            ####### SLicing/bandpass filtering via kaiser filter
            # This check is - if 25MHz is both sample rate and frequency slice value, we do not filter and use IQ as is.
            if fs > 2*cutoff:
                taps_firwin = signal.firwin(numtaps, cutoff, window=('kaiser', kaiser_beta_hh), fs=fs)
                iq_shift = np.multiply(iq, np.exp(1j * 2 * np.pi * shift_freq * np.arange(0, N) * Ts))
                filt_signal = signal.lfilter(taps_firwin, 1, iq_shift)
                dec_order = int(fs / f_step)
                filt_dec_signal = signal.decimate(filt_signal, dec_order)
            else:
                filt_dec_signal = iq




            ####### Feature extraction
            iq_feature = np.real(np.multiply(filt_dec_signal, np.conj(filt_dec_signal))) # feature extraction
            iq_feature = iq_feature - np.mean(iq_feature) # remove DC
            if emanationInputObj.plot_flags['spectrogram']:
                spectrogram_emanations(iq, fs, CF_snapshot, emanationInputObj.folder,emanationInputObj.CF_p1_p2, \
                                       emanationInputObj.cmap, emanationInputObj.scenario)
            # if plot_flag_spectrogram:
            #     spectrogram_emanations(iq, fs, CF_snapshot, emanationInputObj.folder)
            #`
            # if plot_flag_FFToverIQ_IQFeature:
            #     fft_iq_iqpower_plot(iq, f_step, iq_feature, fs)
            # if plot_PSD_flag:
            #     plot_PSD(iq_feature, fs, dur_ensemble, perc_overlap, kaiser_beta_hh,f_step, emanationInputObj.folder)


            # Higher frequency harmonics, Welch PSD estimate done.
            high_ff_search = True # This flag is used in estimate harmonic function only, to pick different search spaces and coarse/fine search options
            fft_iq_hh = WelchPSDEstimate(iq_feature, f_step, dur_ensemble, perc_overlap, kaiser_beta_hh, config_dict)
            f_range_hh = np.arange(-f_step / 2, f_step / 2, f_step / win_len)

            if emanationInputObj.plot_flags['PSD']:
                if high_ff_search:
                    PSD_filename_addendum = 'Scenario_' + emanationInputObj.scenario + emanationInputObj.CF_p1_p2 + '_highFF_'
                else:
                    PSD_filename_addendum = 'Scenario_' + emanationInputObj.scenario + emanationInputObj.CF_p1_p2 + '_lowFF_'

                setaxislim_flag = False
                plot_PSD(config_dict, iq, fs, emanationInputObj.PSD_plot_param['dur_ensemble'][0], perc_overlap, kaiser_beta_hh, \
                         f_range_hh, emanationInputObj.folder, \
                         PSD_filename_addendum+ 'withoutPreprocessing', high_ff_search, emanationInputObj.PSD_plot_param['zoom_perc'][0], 'withoutPreprocessing',f_step, setaxislim_flag)
                plot_PSD(config_dict, iq_feature, fs, emanationInputObj.PSD_plot_param['dur_ensemble'][1], perc_overlap, kaiser_beta_hh,\
                         f_range_hh, emanationInputObj.folder, \
                        PSD_filename_addendum+'withPreprocessing', high_ff_search, emanationInputObj.PSD_plot_param['zoom_perc'][1], 'withPreprocessing',f_step, setaxislim_flag)
                plot_PSD(config_dict, iq_feature, fs, emanationInputObj.PSD_plot_param['dur_ensemble'][0], perc_overlap,\
                         kaiser_beta_hh, f_range_hh, emanationInputObj.folder, PSD_filename_addendum + 'withPreprocessing_noAveraging', high_ff_search,\
                         emanationInputObj.PSD_plot_param['zoom_perc'][1], 'withPreprocessing_noAveraging',f_step, setaxislim_flag)

            # print(fft_iq)
            bad_indices = np.argwhere(fft_iq_hh < log10_nan_check_limit)
            fft_iq_hh[bad_indices] = log10_nan_check_limit
            fft_iq_dB_hh = 10 * np.log10(fft_iq_hh)
            Fb_hh = f_step / len(fft_iq_dB_hh)
            # Dominant peaks frequency and SNR estimated.
            SNR, f_est_hh, locpeaks_hh = EstimatePeaks(fft_iq_dB_hh, f_range_hh, Fb_hh, min_peaks_detect,\
                                                       gb_thresh_hh, ntimes_ns, Maxpeaks_hh, config_dict)
            # If we detect atleast 5 dominant peaks, we pass it through the next module to estimate frequencies
            # of harmonic series present within it, if any, along with the SNR for each peak.
            # We check if there are minimum number of peaks. The FFT of real signal has a mirror image of the peaks
            # Therefore the true min_peaks in positive frequency region is twice of it plus the tone at DC.
            # Gather only positive frequency peaks. This is because, the input to FFT is real and therefore positive and
            # negative frequecnies are duplicate of each other.
            # Also DC is not of use and needs ot be removed.
            fest_pos, SNR_pos, pos_idx = [], [], []
            if f_est_hh.size > 0:  # ensuring we have non-empty list of peaks
                pos_idx = np.argwhere(f_est_hh > 0)
                if pos_idx.size > 0:
                    fest_pos = f_est_hh[pos_idx]
                    SNR_pos = SNR[pos_idx]

                    # rounding fest_pos to one decimal place to reduce chances of finding and removing peaks
                    # after first harmonic estimate.
                    fest_pos = np.around(fest_pos, 1)
            harmonic_found = 1

            # Iteratively find fundamental harmonics. If there are 200 peaks, out of which 60 belong to say 22 kHz fundamental harmonic.
            # The rest belong to fundamental harmonic 67 kHz. We first estimate 22 kHz and peaks corresponding to it,
            # remove those from the main list of peaks and iteratively estimates harmonics from remaining set of peaks.
            num_runs = 1
            # We also ensure that we have a config to allow only num_runs_fundharmonic_estimate_perfreqslot number of runs to estimate
            # fundamental harmonic. If that number is 2, we iteratively call Estimate harmonic twice.
            # If we are doing search for higher harmonics, Esimate harmonics does not use fine search based on this flag
            # higher_harmonic_search = True
            while (len(fest_pos) > min_peaks_detect) and harmonic_found and \
                    (num_runs <=num_runs_fundharmonic_estimate_perfreqslot):
                num_runs = num_runs +1
                # ensuring that we have atleast 6 peaks from which we estimate the harmonic

                # min_peaks_detect = 5
                fundamental_harmonic, harmonic_freq, harmonic_SNR, obj_func_val  = EstimateHarmonic(SNR_pos, fest_pos,\
                                                                   min_peaks_detect, high_ff_search, config_dict)
                # We check if this is an emanation. It should have atleast numpeaks_crossthresh peaks part of
                # the estimated harmonic series that have SNR greater than SNRThresh
                # This additional check on top of the one done inside EstimateHarmonic: this is to help not
                #report harmonics that are sometimes too weak and we find only say two of the series.
                # we are thus very strict in reporting only correct harmonics.
                # SNR_Thresh = 3


                harmonic_SNR = np.array(harmonic_SNR)
                harmonic_freq = np.array(harmonic_freq)
                # numpeaks_crossthresh = 3
                ns,__,__ = Noisespread(fft_iq_dB_hh, config_dict)
                SNR_Thresh = np.min([ns + gb_thresh_hh, ntimes_ns * ns])
                if np.sum(np.array(harmonic_SNR) > SNR_Thresh) > numpeaks_crossthresh:
                    # fundamental_harmonic = harmonic_freq[0]
                    harmonic_freq = harmonic_freq[harmonic_SNR > SNR_Thresh]
                    harmonic_SNR = harmonic_SNR[harmonic_SNR > SNR_Thresh]
                    harmonic_absolutefreq = list(np.asarray(harmonic_freq) + CF_freqslot + CF_snapshot)
                    harmonic_list[fundamental_harmonic] = {'components': harmonic_absolutefreq, \
                                                           'components_relativefreq':harmonic_freq, 'SNR': harmonic_SNR,\
                                                           'obj_func_val':obj_func_val}
                    # Need to find peaks that are not part of the previous frequency list
                    fest_idx = np.in1d(fest_pos, harmonic_freq).nonzero()[0]

                    fest_pos = np.delete(fest_pos, fest_idx)
                    SNR_pos = np.delete(SNR_pos, fest_idx)
                    # harmonic_freq_list.append(harmonic_freq[0])
                    # harmonic_SNR_list.append(harmonic_SNR[0])
                else:
                    harmonic_found = 0

            ########## Repeat the steps for lower frequency harmonics.
            # For Welch PSD - we do averaging over 1mS epochs, we have a frequency resolution over 1000 Hz.
            # Thus we do the processing without averaging for estimating lower frequency harmonics PSD.


            # iq_whole_feature = np.real(np.multiply(iq_whole, np.conj(iq_whole)))
            # iq_whole_feature = iq_whole_feature - np.mean(iq_whole_feature)  # remove DC


            high_ff_search = False
            
            if lowerHarmonic_window == 'Kaiser':
                w = kaiser(len(iq_feature), kaiser_beta_lh)
                w /= np.sum(w)
            else:
                w = np.ones(len(iq_feature))

            w_energy = (np.real(np.vdot(w, w))) / len(w)
            iq_w = np.multiply(iq_feature, w)
            fft_iq = np.fft.fftshift(np.abs(np.fft.fft(iq_w)))
            # fft_iq_slice = np.fft.fftshift(np.abs(np.fft.fft(iq_feature)))

            fft_power = np.multiply(fft_iq, fft_iq)/(w_energy*len(w))

            bad_indices = np.argwhere(fft_power<log10_nan_check_limit)
            fft_power[bad_indices] = np.percentile(fft_power,50)#log10_nan_check_limit
            fft_power_dB = 10 * np.log10(fft_power)


            frange_slice = np.arange(-f_step / 2, f_step / 2, f_step / len(fft_power_dB))
            idx_left = np.argmin(np.abs(frange_slice + fs_lh / 2))
            idx_right = np.argmin(np.abs(frange_slice - fs_lh / 2))
            Fb_lh = f_step / len(fft_power_dB)
            frange_lh = np.arange(frange_slice[idx_left], frange_slice[idx_right + 1], Fb_lh)
            low_fh_indices = np.arange(idx_left, idx_left + len(frange_lh), 1)
            fft_iq_dB_lh = fft_power_dB[low_fh_indices]
            # fft_iq_lh = fft_power_dB[idx_left: idx_left + len(frange_lh)]

            if emanationInputObj.plot_flags['PSD']:
                if high_ff_search:
                    PSD_filename_addendum = 'Scenario_' + emanationInputObj.scenario + emanationInputObj.CF_p1_p2 + '_highFF_'
                else:
                    PSD_filename_addendum = 'Scenario_' + emanationInputObj.scenario + emanationInputObj.CF_p1_p2 + '_lowFF_'
                setaxislim_flag = False
                plot_PSD(config_dict, iq[low_fh_indices], fs, emanationInputObj.PSD_plot_param['dur_ensemble'][2], perc_overlap, \
                         kaiser_beta_hh, frange_lh, emanationInputObj.folder, \
                         PSD_filename_addendum + '_withoutPreprocessing', high_ff_search, emanationInputObj.PSD_plot_param['zoom_perc'][2], '_withoutPreprocessing',f_step, setaxislim_flag)
                plot_PSD(config_dict, iq_feature[low_fh_indices], fs, emanationInputObj.PSD_plot_param['dur_ensemble'][3], perc_overlap, \
                         kaiser_beta_hh, frange_lh, emanationInputObj.folder, \
                         PSD_filename_addendum + '_withPreprocessing', high_ff_search, emanationInputObj.PSD_plot_param['zoom_perc'][3], '_withPreprocessing',f_step, setaxislim_flag)

            SNR, f_est_lh, locpeaks_lh = EstimatePeaks(fft_iq_dB_lh, frange_lh, Fb_lh, min_peaks_detect, \
                                                       gb_thresh_lh, ntimes_ns,Maxpeaks_lh, config_dict)
            fest_pos = []

            #Plotting

            if f_est_lh.size > numpeaks_crossthresh:  # ensuring we have non-empty list of peaks
                pos_idx = np.argwhere(f_est_lh > 0)
                fest_pos = f_est_lh[pos_idx]
                SNR_pos = SNR[pos_idx]

                fest_pos = np.around(fest_pos, 1)
                harmonic_found = 1
                num_runs = 1
                # We also ensure that we have a config to allow only num_runs_fundharmonic_estimate_perfreqslot number of runs to estimate
                # fundamental harmonic. If that number is 2, we iteratively call Estimate harmonic twice.
                # harmonic_CF = (SF_freqslot + EF_freqslot) / 2
                # If we are doing search for higher harmonics, Esimate harmonics does not use fine search based on this flag
                # higher_harmonic_search = False
                while (len(fest_pos) > min_peaks_detect) and harmonic_found and \
                        (num_runs <=num_runs_fundharmonic_estimate_perfreqslot):
                    num_runs = num_runs +1
                    fundamental_harmonic, harmonic_freq, harmonic_SNR, obj_func_val  = \
                        EstimateHarmonic(SNR_pos, fest_pos, min_peaks_detect, high_ff_search, config_dict)
                    harmonic_SNR = np.array(harmonic_SNR)
                    harmonic_freq = np.array(harmonic_freq)

                    ns,__,__ = Noisespread(fft_iq_dB_lh, config_dict)

                    SNR_Thresh = np.min([ns + gb_thresh_lh, 2 * ns])
                    if np.sum(np.array(harmonic_SNR) > SNR_Thresh) > numpeaks_crossthresh:
                        # fundamental_harmonic = harmonic_freq[0]
                        harmonic_freq_full = harmonic_freq
                        harmonic_freq = harmonic_freq[harmonic_SNR > SNR_Thresh]
                        harmonic_SNR = harmonic_SNR[harmonic_SNR > SNR_Thresh]

                        harmonic_absolutefreq = list(np.asarray(harmonic_freq) + CF_freqslot + CF_snapshot)
                        harmonic_list[fundamental_harmonic] = {'components': harmonic_absolutefreq,
                                                               'components_relativefreq': harmonic_freq,
                                                               'SNR': harmonic_SNR,\
                                                           'obj_func_val':obj_func_val}

                        # harmonic_list[fundamental_harmonic] = {'components': harmonic_freq, 'SNR': harmonic_SNR}
                        # Need to find peaks that are not part of the previous frequency list
                        fest_idx = np.in1d(fest_pos, harmonic_freq_full).nonzero()[0]
                        fest_pos = np.delete(fest_pos, fest_idx)
                        SNR_pos = np.delete(SNR_pos, fest_idx)
                        # harmonic_freq_list.append(harmonic_freq[0])
                        # harmonic_SNR_list.append(harmonic_SNR[0])
                    else:
                        harmonic_found = 0

            # harmonic_complete_list[frs_id] = harmonic_list
            print("harmonic_list: ", harmonic_list.keys())
            print("SF_freqslot: ", (SF_freqslot + emanationInputObj.CF) / 1e6, ' MHz')
            print("EF_freqslot: ", (EF_freqslot + emanationInputObj.CF) / 1e6, ' MHz')
            SF_freqslot += f_step
            EF_freqslot += f_step
    # output_dict = {}

    if emanationInputObj.plot_flags['peaks'] == 1:
        setaxislim_flag = False
        PSD_filename_addendum_highff = 'Scenario_' + emanationInputObj.scenario + emanationInputObj.CF_p1_p2 + '_highFF_'
        PSD_filename_addendum_lowff = 'Scenario_' + emanationInputObj.scenario + emanationInputObj.CF_p1_p2 + '_lowFF_'
        plotpeaks(f_est_hh, harmonic_list, f_range_hh, fft_iq_dB_hh, locpeaks_hh, True, emanationInputObj.PSD_plot_param['zoom_perc'][1], \
                  emanationInputObj.PSD_plot_param['diffcolor_eachharmonic'],  emanationInputObj.folder, PSD_filename_addendum_highff, setaxislim_flag)
        plotpeaks(f_est_lh, harmonic_list, frange_lh, fft_iq_dB_lh, locpeaks_lh, False, emanationInputObj.PSD_plot_param['zoom_perc'][3], \
                  emanationInputObj.PSD_plot_param['diffcolor_eachharmonic'],  emanationInputObj.folder, PSD_filename_addendum_lowff, setaxislim_flag)
        # plotpeaks(f_est_hh, harmonic_list, f_range_hh, fft_iq_dB_hh, locpeaks_hh, True)
        # plotpeaks(f_est_lh, harmonic_list, frange_lh, fft_iq_dB_lh, locpeaks_lh, False)
        # Plot same color - that means we just do scatter plot for all peaks identified.
        plotpeaks(f_est_hh, harmonic_list, f_range_hh, fft_iq_dB_hh, locpeaks_hh, True,
                  emanationInputObj.PSD_plot_param['zoom_perc'][1], \
                  False, emanationInputObj.folder,
                  PSD_filename_addendum_highff, setaxislim_flag)
        plotpeaks(f_est_lh, harmonic_list, frange_lh, fft_iq_dB_lh, locpeaks_lh, False,
                  emanationInputObj.PSD_plot_param['zoom_perc'][3], \
                  False, emanationInputObj.folder,
                  PSD_filename_addendum_lowff, setaxislim_flag)

    if emanationInputObj.plot_flags['Objfunc_ErrvsFreq']:
        # fft_iq_iqpower_plot(iq, f_step, iq_feature, fs)
        Objfunc_ErrvsFreq_plot(harmonic_list,emanationInputObj.CF_p1_p2, emanationInputObj.folder)

    # phase_dict = {1:'BG', 2:'BL', 3:'TE'}
    output_dict = {'sample_rate': emanationInputObj.fs, 'center_frequency': emanationInputObj.CF, \
                   'Snapshot_processed':Snapshot_processed ,'results':harmonic_list}
    pickle_file = emanationInputObj.folder + 'Scenario_' + emanationInputObj.scenario + '_CF_' + \
                  str(int(emanationInputObj.CF/1e6)) + 'MHz' + '.pkl'
    # pickle_file = filename
    with open(pickle_file, 'wb') as filename:
        pickle.dump(output_dict, filename, pickle.HIGHEST_PROTOCOL)
    # Define the text file path
    file_path = emanationInputObj.folder + 'Scenario_' + emanationInputObj.scenario + '_CF_' + \
                  str(int(emanationInputObj.CF/1e6)) + 'MHz' + '_pitchEstimates.txt'

    # Open the text file in write mode
    with open(file_path, 'w') as file:
        # Write each element of the list to the file
        for item in list(harmonic_list.keys()):
            file.write(str(item) + '\n')

    # return pickle_file


### CODE TO save input to estimate harmonic - that can be used to independently do parametric search tests and further debugging offline.
# high_freq_Desktop1601_125to150e6_twotones = {'a_est':a_est, 'f_est':f_est,'wt_meas_pred':wt_meas_pred, 'p':p, 'numpeaks_crossthresh':numpeaks_crossthresh}
# from scipy.io import savemat
# import pickle
# savemat("/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Results/high_freq_Desktop1601_125to150e6_twotones.mat",high_freq_Desktop1601_125to150e6_twotones)
# with open("/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Results/high_freq_Desktop1601_125to150e6_twotones.pkl", 'wb') as filename:
#     pickle.dump(high_freq_Desktop1601_125to150e6_twotones, filename, pickle.HIGHEST_PROTOCOL)