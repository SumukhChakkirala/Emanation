import os
import sys
import json
import yaml
import pickle
import numpy as np
from decimal import Decimal
from scipy.signal.windows import kaiser

# Optional GPU: use CuPy if available; otherwise stay on CPU
import numpy as np
cp = None
GPU_AVAILABLE = False
try:
    import cupy as _cp
    try:
        # ensure CUDA runtime (nvrtc) is available and initialize
        _cp.cuda.runtime.runtimeGetVersion()
        cp = _cp
        GPU_AVAILABLE = True
    except Exception as _e:
        print('CuPy is installed but CUDA runtime init failed:', _e)
        cp = None
        GPU_AVAILABLE = False
except Exception:
    cp = None
    GPU_AVAILABLE = False

# Patch Welch PSD to run on GPU (CuPy) when available
def WelchPSDEstimate_gpu(iq_feature, fs, dur_ensemble, perc_overlap, kaiser_beta, config_dict):
    """GPU-capable Welch PSD. Falls back to NumPy if GPU not available."""
    xp = cp if (GPU_AVAILABLE and cp is not None) else np

    win_len = int(np.floor(dur_ensemble * fs)) if dur_ensemble > 0 else len(iq_feature)
    if win_len <= 0:
        win_len = len(iq_feature)
    shift = int(np.floor(win_len * ((100 - perc_overlap) / 100))) if win_len > 0 else 0
    start_idx = 0

    if len(iq_feature) > win_len:
        end_idx = win_len
        inc_ens_avg_power = xp.zeros(win_len, dtype=xp.float64)
        w_cpu = kaiser(int(win_len), kaiser_beta)
    else:
        end_idx = len(iq_feature)
        inc_ens_avg_power = xp.zeros(end_idx, dtype=xp.float64)
        w_cpu = kaiser(int(end_idx), kaiser_beta)

    # Attempt GPU path (including window creation); if NVRTC or kernel compilation fails, fall back to CPU implementation
    try:
        # move window to xp (GPU or CPU)
        w = xp.asarray(w_cpu)
        w = w / xp.sum(w)
        w_energy = (xp.real(xp.vdot(w, w))) / len(w)
        num_ensembles = 0

        # Prepare data array in xp if using GPU; otherwise use numpy
        if xp is not np:
            iq_xp = cp.asarray(iq_feature)
        else:
            iq_xp = np.asarray(iq_feature)

        while end_idx <= len(iq_feature):
            block = iq_xp[start_idx:end_idx]
            iqpower_win = block * w
            inc_ensemble_fft = xp.fft.fftshift(xp.abs(xp.fft.fft(iqpower_win)))
            inc_ensemble_power = (inc_ensemble_fft * inc_ensemble_fft) / (w_energy * len(w))
            inc_ens_avg_power += inc_ensemble_power
            start_idx += shift
            end_idx += shift
            num_ensembles += 1

        if num_ensembles > 0:
            inc_ens_avg_power /= num_ensembles

        print("num_ensembles:", num_ensembles)

        if xp is not np:
            return cp.asnumpy(inc_ens_avg_power)
        return inc_ens_avg_power
    except Exception as e:
        # GPU kernel compilation or runtime failed (e.g., missing nvrtc DLL). Fall back to CPU.
        print('GPU path failed, falling back to CPU. Error:', e)
        # Recompute using NumPy
        n_out = win_len if len(iq_feature) > win_len else len(iq_feature)
        inc_ens_avg_power_cpu = np.zeros(n_out, dtype=np.float64)
        w_cpu_arr = np.asarray(w_cpu)
        w_cpu_arr = w_cpu_arr / np.sum(w_cpu_arr)
        w_energy_cpu = (np.real(np.vdot(w_cpu_arr, w_cpu_arr))) / len(w_cpu_arr)
        start_idx_cpu = 0
        end_idx_cpu = win_len if len(iq_feature) > win_len else len(iq_feature)
        num_ensembles_cpu = 0
        while end_idx_cpu <= len(iq_feature):
            block = np.asarray(iq_feature[start_idx_cpu:end_idx_cpu])
            iqpower_win = block * w_cpu_arr
            inc_ensemble_fft = np.fft.fftshift(np.abs(np.fft.fft(iqpower_win)))
            inc_ensemble_power = (inc_ensemble_fft * inc_ensemble_fft) / (w_energy_cpu * len(w_cpu_arr))
            inc_ens_avg_power_cpu[:len(inc_ensemble_power)] += inc_ensemble_power
            start_idx_cpu += shift
            end_idx_cpu += shift
            num_ensembles_cpu += 1
        if num_ensembles_cpu > 0:
            inc_ens_avg_power_cpu /= num_ensembles_cpu
        return inc_ens_avg_power_cpu

# Make this module use GPU Welch when available
import EmanationDetection_search as eds
if GPU_AVAILABLE:
    eds.WelchPSDEstimate = WelchPSDEstimate_gpu
else:
    print("CuPy not found; falling back to CPU WelchPSDEstimate")

pythonfiles_location = '/Users/venkat/Documents/scisrs/Emanations/Phase1/Emanations_JournalCode/Emanations/ParamSearch/'
sys.path.insert(1, pythonfiles_location)
pythonfiles_location = os.getcwd() + '/'

# Results root (env override supported)
results_folder_top = os.environ.get('EMANATION_RESULTS_DIR', os.path.join(os.getcwd(), 'Results'))
os.makedirs(results_folder_top, exist_ok=True)

scenario_IQfolder_dict = {'Laptop_MonitorViaAdaptor': {'durationcapture_ms': '300'}}
scenario_list = ['DiracComb_varyingSNR']

CF_range_start = 200  # MHz
freq_slot_start = 1
CF_range_end = 300    # MHz
freq_slot_end = 2

freq_slot_range = np.arange(freq_slot_start, freq_slot_end, 1)
CF_range = np.arange(CF_range_start, CF_range_end, 200)

hyper_param = {}
PSD_plot_param = {
    'zoom_perc': [[4, 30, 100]] * 4,
    'diffcolor_eachharmonic': True
}
spectrogram_flag, PSD_flag, Objfunc_ErrvsFreq, peaks_flag = True, True, True, True
plot_flags = [spectrogram_flag, PSD_flag, Objfunc_ErrvsFreq, peaks_flag]
plot_dict = {
    'spectrogram': plot_flags[0],
    'PSD': plot_flags[1],
    'Objfunc_ErrvsFreq': plot_flags[2],
    'peaks': plot_flags[3],
    'cmap': 'viridis',
}
hyper_param['err_thresh_perc'] = 2
s_range = [-1]

def Iteration_perHyperParam(scenario_list, scenario_IQfolder_dict, CF_range, freq_slot_range,
                            results_folder, results_folder_top, hyper_param_string,
                            config_dict, plot_dict, PSD_plot_param, SNR):
    samprate = 200e6
    f_step1 = 25e6
    samprate_slice = f_step1
    trial_num = 0
    CF_slice_range = []

    for scenario in scenario_list:
        results_folder = os.path.join(results_folder_top, 'Results_DiracComb', 'SNR_minus14', hyper_param_string)
        os.makedirs(results_folder, exist_ok=True)

        for CF in CF_range:
            for freq_slot in freq_slot_range:
                SF_freqslot = -samprate / 2 + (freq_slot) * f_step1
                EF_freqslot = -samprate / 2 + (freq_slot + 1) * f_step1
                CF_freqslot = (SF_freqslot + EF_freqslot) / 2

                with open(os.path.join(iq_dict_folder, iq_filename), 'rb') as dict_file:
                    dict_IQ = pickle.load(dict_file)
                iq = dict_IQ["SNR_" + str(SNR)]

                CF_p1_p2 = str(int(CF_freqslot / 1e6 + CF)) + hyper_param_string
                print("Scenario is: ", scenario)
                print(" CF: ", CF_freqslot + CF * 1e6)
                print("Range of frequencies: Start freq: ", CF + SF_freqslot / 1e6, ' MHz. End freq: ', CF + EF_freqslot / 1e6, ' MHz.')

                data = {
                    'iq': iq,
                    'sample_rate': samprate_slice,
                    'center_freq': CF_freqslot + CF * 1e6,
                    'time_duration': trial_num,
                    'path': results_folder,
                    'scenario': scenario,
                    'pythonfiles_location': pythonfiles_location,
                    'plot_dict': plot_dict,
                    'CF_p1_p2': CF_p1_p2,
                    'PSD_plot_param': PSD_plot_param
                }

                dict_resultsval = eds.EmanationDetection(data, config_dict)
                CF_slice_range.append(int(data['center_freq'] / 1e6))

    measured_partials_path = os.path.join(results_folder, 'MeasuredPartials_divide_PitchEstimate_AcrossFreq.txt')
    os.makedirs(os.path.dirname(measured_partials_path), exist_ok=True)
    with open(measured_partials_path, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        for CF_slice in CF_slice_range:
            results_file = os.path.join(results_folder, f"Scenario_{scenario}_CF_{CF_slice}MHz.pkl")
            with open(results_file, 'rb') as file:
                resultval = pickle.load(file)
            for keyval in resultval['results'].keys():
                components_relativefreq = resultval['results'][keyval]['components_relativefreq']
                SNR_val = np.median(resultval['results'][keyval]['SNR'][0:5])
                remainder = np.divide(components_relativefreq, keyval)
                print("CF: ", CF_slice)
                print("Pitch: ", keyval)
                print("SNR: ", SNR_val)
                print(remainder)
        sys.stdout = original_stdout

    pitch_snr_path = os.path.join(results_folder, 'Pitch_SNR_acrossFreq.txt')
    os.makedirs(os.path.dirname(pitch_snr_path), exist_ok=True)
    with open(pitch_snr_path, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        for CF_slice in CF_slice_range:
            results_file = os.path.join(results_folder, f"Scenario_{scenario}_CF_{CF_slice}MHz.pkl")
            with open(results_file, 'rb') as file:
                resultval = pickle.load(file)
            for keyval in resultval['results'].keys():
                SNR_val = np.median(resultval['results'][keyval]['SNR'][0:5])
                print("CF: ", CF_slice, "MHz. Pitch: ", round(keyval, 2), ". SNR: ", SNR_val)
        sys.stdout = original_stdout

    Results_file = os.path.join(results_folder, 'Num_EmanationsPerScenario.txt')
    os.makedirs(os.path.dirname(Results_file), exist_ok=True)
    append_write = 'a' if os.path.exists(Results_file) else 'w'
    with open(Results_file, append_write) as f:
        original_stdout = sys.stdout
        sys.stdout = f
        total_count_num_eman = 0
        for CF_slice in CF_slice_range:
            results_file = os.path.join(results_folder, f"Scenario_{scenario}_CF_{CF_slice}MHz.pkl")
            with open(results_file, 'rb') as file:
                resultval = pickle.load(file)
            total_count_num_eman += len(list(resultval['results'].keys()))
        print(hyper_param_string, ' ', scenario, ' is: ', total_count_num_eman)
        sys.stdout = original_stdout

def update_yaml_file(hyper_param, file_path="synapse_emanation_search.yaml"):
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    yaml_data['EstimateHarmonic']['Err_thresh_dict'][100000000000] = float(str(hyper_param['err_thresh_perc']))
    yaml_data['EstimateHarmonic']['Err_thresh_dict'][1000] = float(str(hyper_param['err_thresh_perc']))
    yaml_data['EstimateHarmonic']['Err_thresh_dict'][50000] = float(str(hyper_param['err_thresh_perc']))
    yaml_data['EmanationDetection']['gb_thresh_hh'] = float(str(hyper_param['gb_thresh_hh']))
    yaml_data['EmanationDetection']['gb_thresh_lh'] = float(str(hyper_param['gb_thresh_lh']))
    yaml_data['EstimateHarmonic']['p_hh_1'] = float(str(hyper_param['p1']))
    yaml_data['EstimateHarmonic']['p_hh_2'] = float(str(hyper_param['p2']))
    yaml_data['EstimateHarmonic']['p_lh_1'] = float(str(hyper_param['p1']))
    yaml_data['EstimateHarmonic']['p_lh_2'] = float(str(hyper_param['p2']))
    yaml_data['EmanationDetection']['ntimes_ns'] = float(str(hyper_param['ntimes_ns']))
    yaml_data['EstimateHarmonic']['wt_meas_pred_hh'] = float(str(hyper_param['wt_meas_pred_hh']))
    yaml_data['EstimateHarmonic']['num_steps_coarse'] = int(str(hyper_param['num_steps_coarse']))
    yaml_data['EstimateHarmonic']['num_steps_finesearch'] = int(str(hyper_param['num_steps_finesearch']))
    yaml_data['EmanationDetection']['min_peaks_detect'] = hyper_param['min_peaks_detect']
    yaml_data['EmanationDetection']['numpeaks_crossthresh'] = hyper_param['numpeaks_crossthresh']
    yaml_data['EmanationDetection']['dur'] = hyper_param['dur']
    print("yaml_data['EstimateHarmonic']: ", yaml_data['EstimateHarmonic'])
    return yaml_data

duty_cycle = 0.1
F_h = 220e3
hyper_param['dur'] = 0.5
iq_dict_folder = './IQData/'
os.makedirs(iq_dict_folder, exist_ok=True)
iq_filename = f"iq_dict_SNR_20_toMinus40_dc_{int(hyper_param['dur']*10)}_ptsecsdata_{int(duty_cycle*10)}_Fh_{int(F_h/1e3)}_kHz.pkl"

print("duty_cycle: ", duty_cycle)
SNR_range = np.arange(20, -42, -2)
with open(os.path.join(iq_dict_folder, iq_filename), 'rb') as file:
    iq_dict = pickle.load(file)

hyper_param['p2'] = 0.5
hyper_param['gb_thresh_lh'] = 0.6
hyper_param['num_steps_coarse'] = 6000
hyper_param['num_steps_finesearch'] = 6000
hyper_param['min_peaks_detect'] = 6
hyper_param['numpeaks_crossthresh'] = 5

for hyper_param['gb_thresh_hh'] in [0.6]:
    for hyper_param['wt_meas_pred_hh'] in [0.5, 1]:
        for hyper_param['p1'] in [0.5]:
            for hyper_param['ntimes_ns'] in [2]:
                for SNR in SNR_range:
                    print("hyper_param['ntimes_ns']: ", hyper_param['ntimes_ns'])
                    config_dict = update_yaml_file(hyper_param)
                    PSD_plot_param['dur_ensemble'] = [0.1, config_dict['EmanationDetection']['dur_ensemble'], 0.1, 0.1]
                    hyper_param_string = (
                        f"Results_C0219_dc{int(10*duty_cycle)}_dur_capt_{int(hyper_param['dur']*10)}"
                        f"_p{hyper_param['p1']}_ns{hyper_param['ntimes_ns']}_wt{hyper_param['wt_meas_pred_hh']}"
                        f"_gb{hyper_param['gb_thresh_hh']}_cs{hyper_param['num_steps_coarse']}"
                        f"_mpd{hyper_param['min_peaks_detect']}_np_ct{hyper_param['numpeaks_crossthresh']}"
                    )

                    results_folder = os.path.join(results_folder_top, 'Results_DiracComb', 'SNR_minus14', hyper_param_string)
                    os.makedirs(results_folder, exist_ok=True)
                    with open(os.path.join(results_folder, 'config_dict.yaml'), "w") as outfile:
                        yaml.dump(config_dict, outfile)

                    Iteration_perHyperParam(
                        scenario_list, scenario_IQfolder_dict, CF_range, freq_slot_range,
                        results_folder, results_folder_top, hyper_param_string,
                        config_dict, plot_dict, PSD_plot_param, SNR
                    )