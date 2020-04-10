import os
import mne
from mne.filter import next_fast_len
from mayavi import mlab

j = ['file_names_without_extension']
k = '.fif' #file extension 

for i in j:
    raws = dict()
    raw_erms = dict()
    new_sfreq = 90.
    raws['vv'] = mne.io.read_raw_fif(os.path.join('raw file path', i+k), verbose='error')
    raws['vv'].load_data().resample(new_sfreq)
    titles = dict(vv='VectorView', opm='OPM')
    n_fft = 2048
    print('Using n_fft=%d (%0.1f sec)' % (n_fft, n_fft / raws['vv'].info['sfreq']))
    bem = os.path.join('BEM solution file path')
    src = os.path.join('src solution file path')
    trans = os.path.join('trans file path')
    fwd = mne.make_forward_solution(raws['vv'].info, trans, src, bem, eeg=False, verbose=True)
    topos = dict(vv=dict())
    stcs = dict(vv=dict())
    snr = 3.
    lambda2 = 1. / snr ** 2
    noise_cov = mne.make_ad_hoc_cov(raws['vv'].info, std=None, verbose=None)
    inverse_operator = mne.minimum_norm.make_inverse_operator(raws['vv'].info, forward=fwd, noise_cov=noise_cov, verbose=True)
    stc_psd, sensor_psd = mne.minimum_norm.compute_source_psd(raws['vv'], inverse_operator, lambda2=lambda2, method='sLORETA', n_fft=n_fft, dB=False, return_sensor=True, verbose=True)
    stc_psd.save(os.path.join('path for result file to be saved', i, 'stc_psd'))
