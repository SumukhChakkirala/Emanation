import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import pickle
from scipy.signal.windows import kaiser
from EstimatePeaks_search import WelchPSDEstimate
file_path="synapse_emanation_search.yaml"
with open(file_path, 'r') as file:
    config_dict = yaml.safe_load(file)


######### We are plotting a recntangular pulse and its Fourier transform ##########
F_h = 220e3 # fundamental frequency in Hz
T_h = 1/F_h # time gap between two rectangular pulses
duty_cycle = 0.1 # assuming a square pulse
T = T_h*duty_cycle # duration of pulse
F = 1/T # main lobe width at +/- F Hz
time_duration = 0.5 # in seconds
#num_pts = 10000 # number of points we get in a single period of a rectangular pulse
Fs = 25e6# sampling rate.
f_step = Fs

results_folder_top = os.getcwd()
hyper_param_string = os.path.join('Results_DiracComb', 'Plots_withoutEmanationDetection', f"{int(duty_cycle*100)} pctDutyCycle")
results_folder = os.path.join(results_folder_top, hyper_param_string)
os.makedirs(results_folder, exist_ok=True)

# helper to save figures into the results folder
def save_plot(name, **kwargs):
    filepath = os.path.join(results_folder, name)
    plt.savefig(filepath, **kwargs) 
# We are using the sampling rate consistent with the 25MHz slice we are processing for emanations
Ts = 1/Fs
SNR_list = np.arange(20,-42,-2)
SNR_list_plotting = [0, -10, -18, -20, -22]#SNR_list
print("duty cycle: ", duty_cycle, "SNR list: ", SNR_list)
generate_plot_harmonictemplate_flag = True
high_ff_search = True
apply_feature_extraction = True
addendum_str_plot_filename = 'WithAvg_withFeatExtrac_pt' + str(int(time_duration*10)) +'_secsdata_'
legend_enable_flag = True # This really slows down. Set legend only for final plots
compute_PSD_noAvg = False # We compute PSD without calling Welch to do averaging when set to True.
save_file_type = 'png'#'pdf'
iq_dict_filename = 'iq_dict_SNR_20_toMinus40_dc_'+ str(int(time_duration*10)) +'_ptsecsdata_' + str(int(duty_cycle*10)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' + '.pkl' # iq_dict.pkl
print("IQ_dict_filename: ", iq_dict_filename)
# this file generated on specified data has a more narrower range of SNRs in it compared to iq_dict.pkl
if compute_PSD_noAvg:
    dur_ensemble = time_duration
else:
    dur_ensemble = 0.001
# Pulse has a zero portion, and non-zero portion within a period
perc_overlap = 75

iq_dict_folder = './IQData/'
os.makedirs(iq_dict_folder, exist_ok=True)
if generate_plot_harmonictemplate_flag: # We do not run this everytime since this takes a lot of time
    t_rp = np.linspace(0, T_h, int(T_h/Ts))
    y_rp = np.zeros(len(t_rp))
    ps_idx = np.abs(t_rp - (T_h/2 - T/2)).argmin() # index of t_rp closest to pulse start
    pe_idx = np.abs(t_rp - (T/2 + T_h/2)).argmin() # index of t_rp closest to pulse end
    y_rp[ps_idx:pe_idx] = 1

    plt.figure(1)
    #plt.subplot(2,2,1)
    plt.plot(t_rp*1e6, y_rp)
    plt.xlabel(r"Time ($\mu$s)", fontsize = 40)
    plt.ylabel("Amplitude", fontsize = 40)
    plt.ylim([0, 1.1])
    plt.xticks([0,1,2,3,4],fontsize = 40)
    plt.yticks([0, 0.5, 1.0],fontsize = 40)
    #plt.title("Rectangular pulse", fontsize = 40)
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.9, top=0.95, wspace=0.25, hspace=0.25)
    plt.tight_layout()
    # plt.savefig(results_folder + 'diraccomb_rectpulse_time_freq.pdf', format='pdf', bbox_inches='tight', pad_inches=.01)
    save_plot(addendum_str_plot_filename + 'RectangularPulse' + str(int(duty_cycle*10)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' + '.' + save_file_type, format=save_file_type,
                bbox_inches='tight', pad_inches=.01)
    plt.close()
    ####################################################################
    ###### Plot the Fourier transform of the recntangular pulse ########
    ####################################################################
    plt.figure(2)
    #plt.subplot(2,2,2)
    Y_rp = np.fft.fftshift(np.fft.fft(y_rp))
    f_rp = np.arange(-Fs/2, Fs/2 + Fs/len(Y_rp), Fs/len(Y_rp))
    # we plot from +/- 5F i.e. across five fundamental freq. range
    f_start = -Fs/2#-40*F_h
    f_end = +Fs/2#+40*F_h
    f_start_idx = np.abs(f_rp - f_start).argmin()
    f_end_idx = np.abs(f_rp - f_end).argmin()
    FT_rectangularPulse = np.abs(Y_rp[f_start_idx:f_end_idx])
    FT_rectangularPulse_norm = np.divide(FT_rectangularPulse, np.max(FT_rectangularPulse))
    plt.plot(f_rp[f_start_idx:f_end_idx]/1e6, FT_rectangularPulse_norm)
    main_lob_freq_pos = F/1e3
    #plt.axvline(x=main_lob_freq_pos, color='r', linestyle='--', label= 'Main lobe transition')  # Add vertical line at 50 kHz
    main_lob_freq_neg = -F/1e3
    #plt.axvline(x=main_lob_freq_neg, color='r', linestyle='--', label= 'Main lobe transition')  # Add vertical line at 50 kHz
    plt.ylabel("FFT Magnitude", fontsize = 40)
    plt.xlabel("Frequency (MHz)", fontsize = 40)
    plt.ylim([0, 1.1])
    plt.xticks(fontsize = 40)
    plt.yticks(fontsize = 40)
    #plt.title("Fourier transform of rectangular pulse", fontsize = 40)
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.9, top=0.95, wspace=0.25, hspace=0.25)
    plt.tight_layout()

    # plt.savefig(results_folder + 'diraccomb_rectpulse_time_freq.pdf', format='pdf', bbox_inches='tight', pad_inches=.01)
    save_plot(addendum_str_plot_filename + 'FT_RectangularPulse'+str(int(duty_cycle*10)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz'  + '.' + save_file_type, format=save_file_type,
                bbox_inches='tight', pad_inches=.01)
    plt.close()
    ####################################################################
    ###### Plot dirac comb function ########
    ####################################################################
    plt.figure(3)
    num_periods_plot = 40 # we plot for 20 periods
    #time_duration = num_periods_plot*T_h
    t_dc = np.linspace(0, time_duration, int(time_duration/Ts))
    s_t = t_dc[0] # start time
    imp_t = s_t + T_h # impulse time of occurrence
    # width_dir_c = 0#int(0.001*len(dir_c))
    dir_c = np.zeros(len(t_dc))
    # below implementation is really slow since it does argmin for each iteration of the numerous thousand iterations.
    #therefore we do it differently.
    # while imp_t < t_dc[-1]:
    #     imp_idx = np.abs(t_dc - imp_t).argmin()
    #     dir_c[imp_idx-width_dir_c: imp_idx+width_dir_c+1] = 1
    #     imp_t = imp_t + T_h


    # Convert impulse times to nearest sample indices.
    # Because t_dc is uniformly spaced, the index is essentially impulse_time / Ts.
    dirac_implementation_closestindexbased = True
    if dirac_implementation_closestindexbased == True:
        impulse_times = np.arange(s_t + T_h, t_dc[-1], T_h)
        impulse_indices = np.rint(impulse_times / Ts).astype(int)
        dir_c[impulse_indices] = 1

    # Issue with below implementation where we have constant index spacing is since Fs = 25e6 is not a multiple of F_h= 220e3
    # , therefore the spacing is 113 samples which maps to 221.238 kHz.
    else:
        dir_c_ns = int((len(t_dc)/F_h)/time_duration) # periodicity of dirac comb in unit of sample number
        dir_c[::dir_c_ns] = 1

    tdc_start = 0*T_h
    tdc_end = 10*T_h
    tdcs_idx = np.abs(t_dc - tdc_start).argmin()
    tdce_idx = np.abs(t_dc - tdc_end).argmin()

    # we only plot the points corresponding to non-zero values as stem.
    dir_c_tmp = dir_c[tdcs_idx:tdce_idx]
    t_dc_tmp = t_dc[tdcs_idx:tdce_idx] * 1e6
    dir_c_tmp_idx = np.where(dir_c_tmp > 0.1)
    dir_c_tmp2 = dir_c_tmp[dir_c_tmp_idx]
    t_dc_tmp2 = t_dc_tmp[dir_c_tmp_idx]

    # The baseline is set to the color of the default plot
    stem_container = plt.stem(t_dc_tmp2, dir_c_tmp2)
    plt.setp(stem_container.baseline, color='C0')

    # plt.stem(t_dc[tdcs_idx:tdce_idx]*1e6, dir_c[tdcs_idx:tdce_idx])
    plt.xlabel(r"Time ($\mu$s)", fontsize = 40)
    plt.ylabel("Amplitude",fontsize = 40)
    plt.ylim([0, 1.1])
    plt.xticks(fontsize = 40)
    plt.yticks([int(0), 0.5, 1.0], fontsize = 40)
    #plt.title("Dirac comb", fontsize = 40)#(f"Dirac comb at periodicity of {T_h*1e3} milliseconds")
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.9, top=0.95, wspace=0.25, hspace=0.25)
    plt.tight_layout()

    # plt.savefig(results_folder + 'diraccomb_rectpulse_time_freq.pdf', format='pdf', bbox_inches='tight', pad_inches=.01)
    save_plot(addendum_str_plot_filename +'DiracComb' + str(int(duty_cycle*10)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' + '.' + save_file_type, format=save_file_type,
                bbox_inches='tight', pad_inches=.01)
    plt.close()
    plt.figure(4)
    dirc_F = np.fft.fftshift(np.fft.fft(dir_c))
    f_dc = np.linspace(-Fs/2, Fs/2 , len(dirc_F))
    #plt.subplot(2,2,4)
    f_start = (1/4)*(-Fs/2)#-10.1*F_h
    f_end = (1/4)*(Fs/2)#+10.1*F_h
    f_start_idx = np.abs(f_dc - f_start).argmin()
    f_end_idx = np.abs(f_dc - f_end).argmin()
    y_FT_diracComb = np.abs(dirc_F[f_start_idx:f_end_idx])
    y_FT_diracComb_norm = np.divide(y_FT_diracComb, np.max(y_FT_diracComb))

    # we only plot the points corresponding to non-zero values as stem.
    f_dc_tmp = f_dc[f_start_idx:f_end_idx]/1e6
    y_FT_diracComb_norm_idx = np.where(y_FT_diracComb_norm > 0.1)
    y_FT_diracComb_norm_tmp = y_FT_diracComb_norm[y_FT_diracComb_norm_idx]
    f_dc_tmp2 = f_dc_tmp[y_FT_diracComb_norm_idx]

    # The baseline is set to the color of the default plot
    stem_container = plt.stem(f_dc_tmp2, y_FT_diracComb_norm_tmp)
    plt.setp(stem_container.baseline, color='C0')

    plt.ylabel("FFT Magnitude", fontsize = 40)
    plt.xlabel("Frequency (MHz)", fontsize = 40)
    plt.xticks(fontsize = 40)
    plt.yticks(fontsize = 40)
    plt.ylim([0, 1.1])
    #plt.title("Fourier transform of Dirac comb", fontsize = 40)#(f"Fourier transform of dirac comb at fundamental frequency of {F_h/1e3} kHz")
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(left=0.12,bottom=0.1,right=0.9,top=0.95,wspace=0.25,hspace=0.25)
    plt.tight_layout()

    #plt.savefig(results_folder + 'diraccomb_rectpulse_time_freq.pdf', format='pdf', bbox_inches='tight', pad_inches=.01)
    save_plot(addendum_str_plot_filename +'FT_DiracComb' + str(int(duty_cycle*10)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' + '.' +save_file_type, format=save_file_type, bbox_inches='tight', pad_inches=.01)

    # plt.show()
    plt.close()

    # Convolution of dirac comb with rectangular pulse
    plt.figure(5)
    #plt.subplot(3,1,1)
    rect_tr = np.convolve(y_rp, dir_c, 'same')
    tdc_start = 0*T_h
    tdc_end = 5.1*T_h
    tdcs_idx = np.abs(t_dc - tdc_start).argmin()
    tdce_idx = np.abs(t_dc - tdc_end).argmin()
    plt.plot(t_dc[tdcs_idx:tdce_idx]*1e6, rect_tr[tdcs_idx:tdce_idx])
    plt.xlabel(r"Time ($\mu$s)", fontsize = 40)
    plt.ylabel("Magnitude",fontsize = 40)
    plt.xticks([0,5,10,15,20], fontsize = 40)
    plt.yticks([0, 0.5, 1.0], fontsize = 40)
    plt.ylim([0, 1.1])
    #plt.title("Rectangular pulse train", fontsize=40)
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.9, top=0.95, wspace=0.25, hspace=0.25)
    plt.tight_layout()
    
    save_plot(addendum_str_plot_filename + 'PulseTrain' + str(int(duty_cycle*10)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' +'.' + save_file_type, format=save_file_type, bbox_inches='tight',
                pad_inches=.01)

    # plt.show()
    plt.close()

    plt.figure(6)
    rect_tr_F = np.fft.fftshift(np.fft.fft(rect_tr))
    f_rect_tr = np.linspace(-Fs/2, Fs/2 , len(rect_tr_F))
    #plt.subplot(3,1,2)
    f_start = -Fs/2#-40.1*F_h
    f_end = +Fs/2#+40.1*F_h
    f_start_idx = np.abs(f_rect_tr - f_start).argmin()
    f_end_idx = np.abs(f_rect_tr - f_end).argmin()
    FT_pulsetrain = np.abs(rect_tr_F[f_start_idx:f_end_idx])
    FT_pulsetrain_norm = np.divide(FT_pulsetrain, np.max(FT_pulsetrain))
    plt.plot(f_rect_tr[f_start_idx:f_end_idx]/1e6, FT_pulsetrain_norm)
    plt.ylabel("FFT Magnitude",fontsize = 40)
    plt.xlabel("Frequency (MHz)",fontsize = 40)
    plt.xticks(fontsize = 40)
    plt.yticks(fontsize = 40)
    #plt.title("Fourier transform of pulse train",fontsize=40)
    plt.ylim([0, 1.1])
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.9, top=0.95, wspace=0.25, hspace=0.25)
    plt.tight_layout()
    save_plot(addendum_str_plot_filename +'FT_PulseTrain' + str(int(duty_cycle*10)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' +  '.' + save_file_type, format=save_file_type, bbox_inches='tight',
                pad_inches=.01)
    plt.close()
    results_dict = {}
    numtaps1 = 1000
    #f_step1 = 25e6

    win_len = np.floor(dur_ensemble * f_step).astype(int)

    if high_ff_search:
        kaiser_beta = 10
        f_range = np.arange(-f_step / 2, f_step / 2, f_step / win_len)
    else:
        kaiser_beta = 3
        f_range_hh = np.arange(-f_step / 2, f_step / 2, f_step / win_len)

    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    iq_feature = np.real(np.multiply(rect_tr, np.conj(rect_tr)))  # feature extraction
    iq_feature = iq_feature - np.mean(iq_feature)  # remove DC

    if compute_PSD_noAvg:
        w = kaiser(len(iq_feature), kaiser_beta)
        w /= np.sum(w)
        w_energy = (np.real(np.vdot(w, w))) / len(w)
        iq_w = np.multiply(iq_feature, w)
        fft_iq = np.fft.fftshift(np.abs(np.fft.fft(iq_w)))
        # fft_iq_slice = np.fft.fftshift(np.abs(np.fft.fft(iq_feature)))
        psd_val = np.multiply(fft_iq, fft_iq) / (w_energy * len(w))
    else:
        # fft_power_dB = 10 * np.log10(fft_power)
        psd_val = WelchPSDEstimate(iq_feature, Fs, dur_ensemble, perc_overlap, kaiser_beta, config_dict)
    psd_val_dB = 10 * np.log10(psd_val)
    psd_val_dB_no_noise = psd_val_dB
    f_range_zoom = f_range  # np.linspace(x_lim_min, x_lim_max, num=len(psd_val))
    psd_val_dB_zoom = psd_val_dB
    if high_ff_search:
        f_range_updated = np.divide(f_range_zoom, 1e6)
        x_label = 'Frequency (MHz)'

    else:
        f_range_updated = np.divide(f_range_zoom, 1e3)
        x_label = 'Frequency (kHz)'
    plt.figure(7)
    plt.plot(f_range_updated, psd_val_dB_zoom)
    plt.xlabel(x_label, fontsize = 40)
    plt.ylabel("Power (dB)", fontsize = 40)
    plt.xticks(fontsize = 40)
    plt.yticks(fontsize = 40)
    #plt.title("PSD of pulse train", fontsize = 40)
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(left=0.12,bottom=0.1,right=0.9,top=0.95,wspace=0.25,hspace=0.25)
    plt.tight_layout()
    save_plot(addendum_str_plot_filename + 'PSD_PulseTrain' + str(int(duty_cycle*10)) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' +  '.' + save_file_type, format=save_file_type,
                bbox_inches='tight', pad_inches=.01)
    plt.close()


# sumukh : till here we synthetically generated a pulse train, from here on out we add noise to it 
    iq_dict = {}
    for SNR in SNR_list:#[0,-10, -12, -14, -16]:
        # Generating complex noise for specified SNR
        var_y = np.var(rect_tr)  # np.average(np.abs(iq))
        var_s = 0.5 * (var_y / (np.power(10, (SNR / 10))))
        np.random.seed(SNR + 100)
        w_s_I = np.random.normal(loc=0, scale=np.sqrt(var_s), size=len(rect_tr))
        np.random.seed(SNR + 100+1)
        w_s_Q = np.random.normal(loc=0, scale=np.sqrt(var_s), size=len(rect_tr))
        w_s = w_s_I + 1j * w_s_Q
        compute_SNR = 10 * np.log10(var_y / np.var(w_s))  # 10*np.log10(var_y/np.average(np.abs(w_s)))
        print("Expected SNR is: ", SNR, " and computed SNR is: ", compute_SNR)
        iq_s = rect_tr + w_s
        iq_dict["SNR_" + str(SNR)] = iq_s
    with open(iq_dict_folder + iq_dict_filename, 'wb') as file:
        pickle.dump(iq_dict, file)
else:
    with open(iq_dict_folder + iq_dict_filename, 'rb') as file:
        iq_dict = pickle.load(file)


    win_len = np.floor(dur_ensemble * f_step).astype(int)
    if high_ff_search:
        kaiser_beta = 10
        f_range = np.arange(-f_step / 2, f_step / 2, f_step / win_len)
    else:
        kaiser_beta = 3
        f_range_hh = np.arange(-f_step / 2, f_step / 2, f_step / win_len)

# option_processing = 'withoutPreprocessing'
# setaxislim_flag = False # Set this flag to true so taht we manually specify the y limits.
#results_folder = results_folder_top + '/' + hyper_param_string + '/'
fig_idx = 1
for zoom_perc in [100, 30, 6]:
    plt.figure(8+fig_idx)
    fig_idx = fig_idx + 1
    # for zoom_perc in zoom_perc_list:
    x_lim_max = (zoom_perc/100)*np.max(f_range)
    x_lim_min = -x_lim_max
    max_idx = np.argmin(np.abs(f_range - x_lim_max))
    min_idx = np.argmin(np.abs(f_range - x_lim_min))
    if max_idx >len(f_range):
        max_idx = len(f_range) - 1
    if min_idx < 0:
        min_idx = 0


    #SNR_str = "SNR"
    for SNR in SNR_list_plotting:
        SNR_str = "SNR" + '_' + str(SNR)
        iq = iq_dict[SNR_str]
        if apply_feature_extraction:
            iq_feature = np.real(np.multiply(iq, np.conj(iq))) # feature extraction
            iq_feature = iq_feature - np.mean(iq_feature) # remove DC bias
        else:
            iq_feature = iq
            #iq_feature = iq_feature - np.mean(iq_feature) # remove DC

        if compute_PSD_noAvg:
            w = kaiser(len(iq_feature), kaiser_beta)
            w /= np.sum(w)
            w_energy = (np.real(np.vdot(w, w))) / len(w)
            iq_w = np.multiply(iq_feature, w)
            fft_iq = np.fft.fftshift(np.abs(np.fft.fft(iq_w)))
            # fft_iq_slice = np.fft.fftshift(np.abs(np.fft.fft(iq_feature)))
            psd_val = np.multiply(fft_iq, fft_iq) / (w_energy * len(w))
        else:
        # fft_power_dB = 10 * np.log10(fft_power)
            psd_val = WelchPSDEstimate(iq_feature, Fs, dur_ensemble, perc_overlap, kaiser_beta, config_dict)
        psd_val_dB = 10 * np.log10(psd_val)
        f_range_zoom = f_range[min_idx:max_idx]#np.linspace(x_lim_min, x_lim_max, num=len(psd_val))
        psd_val_dB_zoom = psd_val_dB[min_idx:max_idx]
        if high_ff_search:
            f_range_updated = np.divide(f_range_zoom,1e6)
            x_label = 'Frequency (MHz)'

        else:
            f_range_updated = np.divide(f_range_zoom, 1e3)
            x_label = 'Frequency (kHz)'

        plt.plot(f_range_updated, psd_val_dB_zoom)
    if legend_enable_flag:
        leg = plt.legend(SNR_list_plotting, title="SNR", fontsize = 40, title_fontsize= 40)
         # plt.legend()
        # get the individual lines inside legend and set line width
        for line in leg.get_lines():
            line.set_linewidth(4)
    plt.xlabel(x_label, fontsize=40)
    # if zoom_perc == 100:
    # plt.yticks([])
    if zoom_perc == 100:
        plt.ylabel("Power (dB)", fontsize=40)
    plt.xticks(fontsize=40)
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
    plt.tight_layout()
    save_plot(addendum_str_plot_filename + '_zp_' + str(zoom_perc) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' +  '.' +save_file_type, format=save_file_type, bbox_inches=None, pad_inches=.01)
    #plt.show()
# plt.show()
    plt.close()

# Plof time and frequency domain to indicate that in time domain, for SNR=0, we do not see a good signature in time domain, but
# we do see the same in frequency domain.

plt.figure(15)
SNR_plot_comparison_time_freq = -14
SNR_str = "SNR" + '_' + str(SNR_plot_comparison_time_freq)
iq = iq_dict[SNR_str]
dur_plot = 50e-6
end_idx = int(dur_plot*Fs)
t_dc = np.linspace(0, time_duration, int(time_duration/Ts))
plt.plot(t_dc[:end_idx]*1e6, np.abs(iq[:end_idx]))
# rect_tr = np.convolve(y_rp, dir_c, 'same')
# tdc_start = 0 * T_h
# tdc_end = 5.1 * T_h
# tdcs_idx = np.abs(t_dc - tdc_start).argmin()
# tdce_idx = np.abs(t_dc - tdc_end).argmin()
# plt.plot(t_dc[tdcs_idx:tdce_idx] * 1e6, rect_tr[tdcs_idx:tdce_idx])
plt.xlabel(r"Time ($\mu$s)", fontsize=40)
plt.ylabel("Amplitude", fontsize=40)
plt.xticks(range(0, int(dur_plot/(1e-6))+10, 10), fontsize=40)
plt.yticks(range(0,5), fontsize=40)
# plt.yticks([0, 0.5, 1.0], fontsize=40)
# plt.ylim([0, 1.1])
# plt.title("Rectangular pulse train", fontsize=40)
plt.gcf().set_size_inches(15, 10)
plt.subplots_adjust(left=0.12, bottom=0.1, right=0.9, top=0.95, wspace=0.25, hspace=0.25)
plt.tight_layout()

save_plot(addendum_str_plot_filename +'TimeDomainPulseTrain' + '_Fh_' + str(
    int(F_h / 1e3)) + '_kHz' + '.' + save_file_type, format=save_file_type, bbox_inches=None,
            pad_inches=.01)

# plt.show()
plt.close()

fig_idx = 1
for zoom_perc in [100]:
    plt.figure(15+fig_idx)
    fig_idx = fig_idx + 1
    # for zoom_perc in zoom_perc_list:
    x_lim_max = (zoom_perc/100)*np.max(f_range)
    x_lim_min = -x_lim_max
    max_idx = np.argmin(np.abs(f_range - x_lim_max))
    min_idx = np.argmin(np.abs(f_range - x_lim_min))
    if max_idx >len(f_range):
        max_idx = len(f_range) - 1
    if min_idx < 0:
        min_idx = 0


    #SNR_str = "SNR"
    for SNR in [SNR_plot_comparison_time_freq]:
        SNR_str = "SNR" + '_' + str(SNR)
        iq = iq_dict[SNR_str]
        if apply_feature_extraction:
            iq_feature = np.real(np.multiply(iq, np.conj(iq))) # feature extraction
            iq_feature = iq_feature - np.mean(iq_feature) # remove DC bias
        else:
            iq_feature = iq
            #iq_feature = iq_feature - np.mean(iq_feature) # remove DC

        if compute_PSD_noAvg:
            w = kaiser(len(iq_feature), kaiser_beta)
            w /= np.sum(w)
            w_energy = (np.real(np.vdot(w, w))) / len(w)
            iq_w = np.multiply(iq_feature, w)
            fft_iq = np.fft.fftshift(np.abs(np.fft.fft(iq_w)))
            # fft_iq_slice = np.fft.fftshift(np.abs(np.fft.fft(iq_feature)))
            psd_val = np.multiply(fft_iq, fft_iq) / (w_energy * len(w))
        else:
        # fft_power_dB = 10 * np.log10(fft_power)
            psd_val = WelchPSDEstimate(iq_feature, Fs, dur_ensemble, perc_overlap, kaiser_beta, config_dict)
        psd_val_dB = 10 * np.log10(psd_val)
        f_range_zoom = f_range[min_idx:max_idx]#np.linspace(x_lim_min, x_lim_max, num=len(psd_val))
        psd_val_dB_zoom = psd_val_dB[min_idx:max_idx]
        if high_ff_search:
            f_range_updated = np.divide(f_range_zoom,1e6)
            x_label = 'Frequency (MHz)'

        else:
            f_range_updated = np.divide(f_range_zoom, 1e3)
            x_label = 'Frequency (kHz)'

        plt.plot(f_range_updated, psd_val_dB_zoom)
    # if legend_enable_flag:
    #     leg = plt.legend(SNR_list_plotting, title="SNR", fontsize = 40, title_fontsize= 40)
    #      # plt.legend()
    #     # get the individual lines inside legend and set line width
    #     for line in leg.get_lines():
    #         line.set_linewidth(4)
    plt.xlabel(x_label, fontsize=40)
    # if zoom_perc == 100:
    # plt.yticks([])
    if zoom_perc == 100:
        plt.ylabel("Power (dB)", fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(range(6,26,4),fontsize=40)
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True,
                    length=8, width=3, direction='out')

    # fig.tight_layout()
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(left=0.12,bottom=0.1, right=0.9,top=0.95,wspace=0.25,hspace=0.25)
    plt.tight_layout()
    save_plot(addendum_str_plot_filename +'FreqDomainComparisonPlot' + '_zp_' + str(zoom_perc) + '_Fh_' + str(int(F_h/1e3)) + '_kHz' +  '.' +save_file_type, format=save_file_type, bbox_inches=None, pad_inches=.01)
    #plt.show()
# plt.show()
    plt.close()



