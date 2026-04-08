import numpy as np
import scipy.interpolate as interpolate
def CFOArtifact(samples, XO_val, clockeffects_dict, samp_rate):
    Ts = 1 / samp_rate  # sample interval
    samples_length = samples.shape[0]
    n = np.arange(0, samples_length)
    CFO_val = XO_val*clockeffects_dict['LOScalingFactor']
    CFO_val = CFO_val[0:samples_length]
    CFO_mult_n = np.multiply(CFO_val, n)

    exp_val = np.exp(1j * 2 * np.pi * CFO_mult_n * Ts)
    samples_withCFO = np.multiply(exp_val, samples)
    return samples_withCFO



def phaseOffset(samples, seed):
    # theta = np.random.uniform(0,2*np.pi)
    np.random.seed(seed)  # setting the random seed
    # To check if the seed has been set, use the command np.random.get_state()[1][0]
    phaseOffsetVal = np.random.uniform(0, 2 * np.pi)
    samples_phaseOffset = np.multiply(samples, np.exp(1j * phaseOffsetVal))
    return samples_phaseOffset


from gnuradio import channels

def gnuradio_channel(samples, samp_rate):
    channel = channels.channel_model(
        noise_voltage=0.01,
        frequency_offset=0.0,
        epsilon=1.0,
        taps=[1+0j, 0.5+0.2j, 0.1-0.1j],
        noise_seed=42
    )
    
    # GNU Radio expects streaming, so this is simplified
    return channel.work([samples], [len(samples)])

