import os
import mne
from mne.filter import next_fast_len
from mayavi import mlab

j = ['CC120120', 'CC120137', 'CC120264', 'CC122016', 'CC220107', 'CC220203', 'CC221336', 'CC222120', 'CC320022', 'CC320059', 'CC320814', 'CC321544', 'CC410119', 'CC410182', 'CC420061', 'CC420286', 'CC420435', 'CC420589', 'CC420776', 'CC510255', 'CC510438', 'CC520239']
k = '.fif'
A = '-trans.fif'
Z = '/transdef_mf2pt2_rest_raw.fif'

for i in j:
    raws = dict()
    raw_erms = dict()
    new_sfreq = 90.
    raws['vv'] = mne.io.read_raw_fif(os.path.join('/home/siddharth/Work/vivek/', i+k), verbose='error')
    raws['vv'].load_data().resample(new_sfreq)
    titles = dict(vv='VectorView', opm='OPM')
    n_fft = 2048
    print('Using n_fft=%d (%0.1f sec)' % (n_fft, n_fft / raws['vv'].info['sfreq']))
    bem = os.path.join('/home/siddharth/Downloads/freesurfer/subjects/collin/bem_sol.fif')
    src = os.path.join('/home/siddharth/Downloads/freesurfer/subjects/collin/src.fif')
    trans = os.path.join('/home/siddharth/Vivek/Working/Data/fiducials/new_22_fiducials/', i+A)
    fwd = mne.make_forward_solution(raws['vv'].info, trans, src, bem, eeg=False, verbose=True)
    topos = dict(vv=dict())
    stcs = dict(vv=dict())
    snr = 3.
    lambda2 = 1. / snr ** 2
    noise_cov = mne.make_ad_hoc_cov(raws['vv'].info, std=None, verbose=None)
    inverse_operator = mne.minimum_norm.make_inverse_operator(raws['vv'].info, forward=fwd, noise_cov=noise_cov, verbose=True)
    stc_psd, sensor_psd = mne.minimum_norm.compute_source_psd(raws['vv'], inverse_operator, lambda2=lambda2, method='sLORETA', n_fft=n_fft, dB=False, return_sensor=True, verbose=True)
    stc_psd.save(os.path.join('/home/siddharth/Vivek/Working/Data/results/source_psd/new_added/', i, 'stc_psd'))
