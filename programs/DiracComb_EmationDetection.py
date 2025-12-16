import yaml
import sys
import os
import json
import pickle
import numpy as np
#from pytictoc import TicToc
from decimal import Decimal

############# What does this file do, why is it here ###########
'''
This file reads IQ data, adds Gaussian noise synthetically for the specified range of SNR levels. 
It is then passed through the Emanation detection funciton to estimate the fundamental frequencies present in the slices of the IQ.
'''
##############

pythonfiles_location = '/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Emanations/ParamSearch/'
sys.path.insert(1, pythonfiles_location)

pythonfiles_location = os.getcwd() + '/'

import os

# Use EMANATION_RESULTS_DIR env var if set; otherwise default to project Results folder
results_folder_top = os.environ.get('EMANATION_RESULTS_DIR', os.path.join(os.getcwd(), 'Results'))
os.makedirs(results_folder_top, exist_ok=True)

#IQ_folder = '/Users/venkat/Documents/scisrs/Emanations/IQSamples/June16thCollect/LaptopConnectedToMonitorviaAdaptor/Laptop_MonitorViaAdaptor_25MHzSlice/'

from EmanationDetection_search import *

scenario_IQfolder_dict = {}
scenario_IQfolder_dict['Laptop_MonitorViaAdaptor'] = {}
scenario_IQfolder_dict['Laptop_MonitorViaAdaptor']['durationcapture_ms'] = '300' # this is not used. Not sure why it is mentioned as 300.
scenario_list = ['DiracComb_varyingSNR']



# Split 200 Mhz capture into 8 logical slot indices: 0 to 7.
# for CF = 200, capture range is 100 to 300 MHz.
# if you want to detect emanation on 150 to 175 MHz, we give slot as 2.
CF_range_start = 200  # in MHz
freq_slot_start = 1
CF_range_end = 300  # in MHz
freq_slot_end = 2

freq_slot_range = np.arange(freq_slot_start, freq_slot_end, 1)
CF_range = np.arange(CF_range_start, CF_range_end, 200)

hyper_param = {}

# wt_range = np.array([0.001, 0.005, 0.01, 0.05, 0.1])
os.makedirs(results_folder_top, exist_ok=True)

PSD_plot_param = {}
# we sometimes want to plot only a percentage of x-axis to get e zoomed version.
# we therefore take a percentage of the x-axis and zoom accordingly and make the plot.
PSD_plot_param['zoom_perc'] = [[4, 30, 100], [4, 30, 100], [4, 30, 100], [4, 30, 100]]
diffcolor_eachharmonic_flag = True  # when overlaying peaks on the PSD< we pick a different color for each of
PSD_plot_param['diffcolor_eachharmonic'] = diffcolor_eachharmonic_flag
spectrogram_flag, PSD_flag, Objfunc_ErrvsFreq, peaks_flag = True, True, True, True
plot_flags = [spectrogram_flag, PSD_flag, Objfunc_ErrvsFreq, peaks_flag]
cmap = 'viridis'
plot_dict = {'spectrogram': plot_flags[0],
             'PSD': plot_flags[1],
             'Objfunc_ErrvsFreq': plot_flags[2], \
             'peaks': plot_flags[3], 'cmap': cmap}
hyper_param['err_thresh_perc'] = 2

s_range = [-1]  # np.arange(-1,-10,-2)


########
######
def Iteration_perHyperParam(scenario_list, scenario_IQfolder_dict, CF_range, freq_slot_range, results_folder, \
                            results_folder_top, hyper_param_string, config_dict, plot_dict, PSD_plot_param, SNR):
    samprate = 200e6
    BW = 200  # Mhz
    results_dict = {}
    numtaps1 = 1000
    f_step1 = 25e6
    samprate_slice = f_step1
    kaiser_beta1 = 20
    trial_num = 0

    CF_slice_range = []
    for scenario in scenario_list:

        # duration_capture = scenario_IQfolder_dict[scenario]['durationcapture_ms']

        results_folder = os.path.join(results_folder_top, 'Results_DiracComb', 'SNR_minus14', hyper_param_string)
        os.makedirs(results_folder, exist_ok=True)

        for CF in CF_range:
            for freq_slot in freq_slot_range:
                SF_freqslot, EF_freqslot = -samprate / 2 + (freq_slot) * f_step1, -samprate / 2 + (
                        freq_slot + 1) * f_step1
                CF_freqslot = (SF_freqslot + EF_freqslot) / 2
                # shift_freq = -1 * (SF_freqslot + EF_freqslot) / 2
                # cutoff1 = f_step1 / 2

                #filename = IQ_dict#'Scenario_' + scenario + '_CF_' + str(int(CF + CF_freqslot / 1e6)) + 'MHz' + '.pkl'
                with open(iq_dict_folder + iq_filename, 'rb') as dict_file:
                    dict_IQ = pickle.load(dict_file)

                iq = dict_IQ["SNR_" + str(SNR)]

                # Generating complex noise for specified SNR
                # var_y = np.var(iq)  # np.average(np.abs(iq))
                # var_s = 0.5 * (var_y / (np.power(10, (SNR / 10))))
                # w_s_I = np.random.normal(loc=0, scale=np.sqrt(var_s), size=len(iq))
                # w_s_Q = np.random.normal(loc=0, scale=np.sqrt(var_s), size=len(iq))
                # w_s = w_s_I + 1j * w_s_Q
                # compute_SNR = 10 * np.log10(var_y / np.var(w_s))  # 10*np.log10(var_y/np.average(np.abs(w_s)))
                # print("Expected SNR is: ", SNR, " and computed SNR is: ", compute_SNR)
                # iq_s = iq + w_s

                ####### SLicing/bandpass filtering via kaiser filter
                CF_p1_p2 = str(int(CF_freqslot / 1e6 + CF)) + hyper_param_string
                print("Scenario is: ", scenario)
                print(" CF: ", CF_freqslot + CF * 1e6)
                print("Range of frequencies: Start freq: ", CF + SF_freqslot / 1e6, ' MHz. End freq: ',
                      CF + EF_freqslot / 1e6, ' MHz.')
                data = {'iq': iq, 'sample_rate': samprate_slice, 'center_freq': CF_freqslot + CF * 1e6, \
                        'time_duration': trial_num, 'path': results_folder, \
                        'scenario': scenario, 'pythonfiles_location': pythonfiles_location, 'plot_dict': plot_dict, \
                        'CF_p1_p2': CF_p1_p2, 'PSD_plot_param': PSD_plot_param}

                #t = TicToc()
                #t.tic()
                dict_resultsval = EmanationDetection(data, config_dict)
                #t.toc()
                CF_slice_range.append(int(data['center_freq'] / 1e6))

        # scenario = 'busy_conference_environ'
        original_stdout = sys.stdout  # Save a reference to the original standard output

    measured_partials_path = results_folder + 'MeasuredPartials_divide_PitchEstimate_AcrossFreq.txt'
    os.makedirs(os.path.dirname(measured_partials_path), exist_ok=True)
    with open(measured_partials_path, 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            # print('This message will be written to a file.')

            for CF_slice in CF_slice_range:  # np.arange(137, 1113, 25):
                results_file = results_folder + 'Scenario_' + scenario + '_CF_' + str(
                    CF_slice) + 'MHz.pkl'

                with open(results_file, 'rb') as file:
                    resultval = pickle.load(file)
                file.close()
                for keyval in resultval['results'].keys():
                    components_relativefreq = resultval['results'][keyval]['components_relativefreq']
                    # We are picking the median SNR of the lowest 5 harmonics as the SNR for the entire harmonic series.
                    SNR_val = np.median(resultval['results'][keyval]['SNR'][0:5])
                    remainder = np.divide(components_relativefreq, keyval)
                    print("CF: ", CF_slice)
                    print("Pitch: ", keyval)
                    print("SNR: ", SNR_val)
                    print(remainder)
            sys.stdout = original_stdout

            # scenario = 'busy_conference_environ'
            original_stdout = sys.stdout  # Save a reference to the original standard output

            pitch_snr_path = results_folder + 'Pitch_SNR_acrossFreq.txt'
            os.makedirs(os.path.dirname(pitch_snr_path), exist_ok=True)
            with open(pitch_snr_path, 'w') as f:
                sys.stdout = f  # Change the standard output to the file we created.
                # print('This message will be written to a file.')

                for CF_slice in CF_slice_range:  # np.arange(137, 1113, 25):
                    results_file = results_folder + 'Scenario_' + scenario + '_CF_' + str(
                        CF_slice) + 'MHz.pkl'

                    with open(results_file, 'rb') as file:
                        resultval = pickle.load(file)
                    file.close()
                    for keyval in resultval['results'].keys():
                        # We are picking the median SNR of the lowest 5 harmonics as the SNR for the entire harmonic series.
                        SNR_val = np.median(resultval['results'][keyval]['SNR'][0:5])
                        print("CF: ", CF_slice, "MHz. Pitch: ", round(keyval, 2), ". SNR: ", SNR_val)
                sys.stdout = original_stdout

            ####### Write values for each scenario to a single file that sits top of scenario folders.
            ####### Values shud be just the total number of pitch values each scenario for that chosen value of hyper parameters
            original_stdout = sys.stdout  # Save a reference to the original standard output
            Results_file = results_folder + 'Num_EmanationsPerScenario.txt'
            os.makedirs(os.path.dirname(Results_file), exist_ok=True)
            if os.path.exists(Results_file):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'  # make a new file if not

            with open(Results_file, append_write) as f:
                sys.stdout = f  # Change the standard output to the file we created.
                # print('This message will be written to a file.')
                total_count_num_eman = 0
                for CF_slice in CF_slice_range:  # np.arange(137, 1113, 25):
                    results_file = results_folder + 'Scenario_' + scenario + '_CF_' + str(
                        CF_slice) + 'MHz.pkl'

                    with open(results_file, 'rb') as file:
                        resultval = pickle.load(file)
                    total_count_num_eman = total_count_num_eman + len(list(resultval['results'].keys()))
                print(hyper_param_string, ' ', scenario, ' is: ', total_count_num_eman)
                sys.stdout = original_stdout


################
########################################################################################################################
# UPDATE YAML FILE
########################################################################################################################
# Function to update values in the YAML file
def update_yaml_file(hyper_param, file_path="synapse_emanation_search.yaml"):
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    # yaml_data['EstimateHarmonic']['s1'] = float(str(hyper_param['s1']))
    # yaml_data['EstimateHarmonic']['s2'] = float(str(hyper_param['s2']))
    yaml_data['EstimateHarmonic']['Err_thresh_dict'][100000000000] = float(str(hyper_param['err_thresh_perc']))
    yaml_data['EstimateHarmonic']['Err_thresh_dict'][1000] = float(str(hyper_param['err_thresh_perc']))
    yaml_data['EstimateHarmonic']['Err_thresh_dict'][50000] = float(str(hyper_param['err_thresh_perc']))
    # Update the data
    yaml_data['EmanationDetection']['gb_thresh_hh'] = float(str(hyper_param['gb_thresh_hh']))
    yaml_data['EmanationDetection']['gb_thresh_lh'] = float(str(hyper_param['gb_thresh_lh']))
    yaml_data['EstimateHarmonic']['p_hh_1'] = float(str(hyper_param['p1']))
    yaml_data['EstimateHarmonic']['p_hh_2'] = float(str(hyper_param['p2']))
    yaml_data['EstimateHarmonic']['p_lh_1'] = float(str(hyper_param['p1']))
    yaml_data['EstimateHarmonic']['p_lh_2'] = float(str(hyper_param['p2']))
    yaml_data['EmanationDetection']['ntimes_ns'] = float(str(hyper_param['ntimes_ns']))
    yaml_data['EstimateHarmonic']['wt_meas_pred_hh'] = float(str(hyper_param['wt_meas_pred_hh']))
    yaml_data['EstimateHarmonic']['num_steps_coarse'] = int(str(hyper_param['num_steps_coarse'] ))
    yaml_data['EstimateHarmonic']['num_steps_finesearch'] = int(str(hyper_param['num_steps_finesearch']))
    yaml_data['EmanationDetection']['min_peaks_detect'] = hyper_param['min_peaks_detect']
    yaml_data['EmanationDetection']['numpeaks_crossthresh'] = hyper_param['numpeaks_crossthresh']
    yaml_data['EmanationDetection']['dur'] = hyper_param['dur']
    #yaml_data['EstimateHarmonic']['wt_meas_pred_lh'] = float(str(hyper_param['wt_meas_pred_lh']))
    print("yaml_data['EstimateHarmonic']: ", yaml_data['EstimateHarmonic'])


    return yaml_data


duty_cycle= 0.1
F_h = 220e3  # <-- change from 250e3 to 220e3
# iq_filename = 'iq_dict_SNR_20_toMinus40_dc_pt'+ str(int(hyper_param['dur']*10)) +'secsdata_' + str(int(duty_cycle*10)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' + '.pkl'
hyper_param['dur'] = 0.5 # typically it is 0.1
iq_dict_folder = './IQData/'
os.makedirs(iq_dict_folder, exist_ok=True)
iq_filename = 'iq_dict_SNR_20_toMinus40_dc_' + str(int(hyper_param['dur']*10)) + '_ptsecsdata_' + str(int(duty_cycle*10)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz.pkl'
# iq_filename = 'iq_dict_SNR_20_toMinus40_dc_pt'+ str(int(hyper_param['dur']*10)) +'secsdata_' + str(int(duty_cycle*10)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' + '.pkl' # iq_dict.pkl
#'iq_dict_SNR_20_toMinus40_dc_' + str(int(10*duty_cycle)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' + '.pkl'
print("duty_cycle: ", duty_cycle)
SNR_range = np.arange(20,-42,-2)
with open(iq_dict_folder + iq_filename, 'rb') as file:
    iq_dict = pickle.load(file)
hyper_param['p2'] = 0.5
# We are trying to prune weaker peaks. This is done in in EstimatePeaks module, where the peaks SNR have to be above a specific
# threshold "thresh = np.min([ns + gb_thresh, ntimes_ns*ns])". We have to increase both ntimes_ns and gb_thresh to achieve this.
#hyper_param['gb_thresh_hh'] = 0.5#1 is the default value used.

hyper_param['gb_thresh_lh'] =0.6#1 is the default value used.
hyper_param['num_steps_coarse'] =6000
hyper_param['num_steps_finesearch'] = 6000
hyper_param['min_peaks_detect'] = 6
hyper_param['numpeaks_crossthresh'] = 5
for hyper_param['gb_thresh_hh'] in [0.6]:#np.concatenate([np.arange(0.1,1,0.1), np.arange(1,10,1)]):
    for hyper_param['wt_meas_pred_hh'] in [0.5,1]: # Results of the parametric serarch of wt for SNR=0, gave wt =1 as a good value. Plot can be seen from .png in below location
    # from the ⁨Users⁩ ▸ ⁨venkat⁩ ▸ ⁨Documents⁩ ▸ ⁨scisrs⁩ ▸ ⁨Emanations⁩ ▸ ⁨Phase1⁩ ▸ ⁨Emanations_JournalCode⁩ ▸ ⁨Emanations⁩ ▸ ⁨StandaloneRunningCode⁩ ▸ ⁨Results_DiracComb⁩
        for hyper_param['p1'] in [0.5]:
            for hyper_param['ntimes_ns'] in [2]:#[1.5,2,2.5,3,3.5,4]: #[2]:
                for SNR in SNR_range:  # ([np.arange(100,11,-5), np.arange(10,0,-1)]):
                    # if SNR >= -12:
                    #     hyper_param['ntimes_ns'] = 2
                    # else:
                    #     hyper_param['ntimes_ns'] = 3
                    print("hyper_param['ntimes_ns']: ", hyper_param['ntimes_ns'])
                    config_dict = update_yaml_file(hyper_param)

                    PSD_plot_param['dur_ensemble'] = [0.1, config_dict['EmanationDetection']['dur_ensemble'], 0.1, 0.1]

                    hyper_param_string = 'Results_C0219_dc'+ str(int(10*duty_cycle)) + '_dur_capt_'+str(int(hyper_param['dur']*10)) +'_p' + str(hyper_param['p1']) + '_ns' + str(hyper_param['ntimes_ns']) + '_wt'+ str(hyper_param['wt_meas_pred_hh']) + \
                        '_gb' + str(hyper_param['gb_thresh_hh']) + '_cs'+ str(hyper_param['num_steps_coarse']) + '_mpd' + str(hyper_param['min_peaks_detect']) + '_np_ct' + str(hyper_param['numpeaks_crossthresh'])
                    # + 'p_' + str(hyper_param['p1']) + "_Errthresh_2"

                    results_folder = results_folder_top + 'Results_DiracComb' + '/SNR_minus14/' + hyper_param_string
                    os.makedirs(results_folder, exist_ok=True)
                    with open(results_folder + '/config_dict.yaml', "w") as outfile:
                        yaml.dump(config_dict, outfile)

                    Iteration_perHyperParam(
                        scenario_list, scenario_IQfolder_dict, CF_range, freq_slot_range, results_folder,
                        results_folder_top, hyper_param_string, config_dict, plot_dict, PSD_plot_param, SNR
                    )