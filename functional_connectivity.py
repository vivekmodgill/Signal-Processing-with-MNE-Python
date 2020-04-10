import numpy as np
import scipy
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
import os

j = ['file_names_without_extension']
k = '.fif'
D = '.npy'

for i in j:
    subjects_dir = 'path of the freesurfer subject directory'
    subject = 'subject name'
    bem = os.path.join('BEM solution file path')
    src = os.path.join('src solution file path')
    trans = os.path.join('trans solution file path')
    raw = mne.io.read_raw_fif(os.path.join('raw file path', i+k), verbose='error')
    raw.crop(0, None).load_data().pick_types(meg=True, eeg=False).resample(90)
    raw.apply_gradient_compensation(0)
    cov = mne.make_ad_hoc_cov(raw.info, std=None, verbose=None)
    events = mne.make_fixed_length_events(raw, duration=5.)
    epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5.0, baseline=None, reject=dict(), preload=True)
    src = mne.read_source_spaces(src)
    fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
    inv = make_inverse_operator(epochs.info, fwd, cov)
    snr = 3.0  
    lambda2 = 1.0 / snr ** 2
    method = "sLORETA" 
    stcs = apply_inverse_epochs(epochs, inv, lambda2, method, pick_ori="normal", return_generator=True)
    labels = mne.read_labels_from_annot('collin', parc='aparc', subjects_dir=subjects_dir)
    label_colors = [label.color for label in labels]
    src = inv['src']
    label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip', return_generator=True)
    fmin0 = 0
    fmax0 = 3.5
    fmin1 = 4
    fmax1 = 7.5
    fmin2 = 8
    fmax2 = 13
    fmin3 = 14
    fmax3 = 30
    sfreq = raw.info['sfreq']
    con0, freqs, times, n_epochs, n_tapers = spectral_connectivity(label_ts, method='wpli2_debiased', mode='multitaper', sfreq=sfreq, fmin=fmin0, fmax=fmax0, faverage=True, mt_adaptive=True, n_jobs=1)
    np.save(os.path.join('path for result files to be saved', i+D), con0)
    con1, freqs, times, n_epochs, n_tapers = spectral_connectivity(label_ts, method='wpli2_debiased', mode='multitaper', sfreq=sfreq, fmin=fmin1, fmax=fmax1, faverage=True, mt_adaptive=True, n_jobs=1)
    np.save(os.path.join('path for result files to be saved', i+D), con1)
    con2, freqs, times, n_epochs, n_tapers = spectral_connectivity(label_ts, method='wpli2_debiased', mode='multitaper', sfreq=sfreq, fmin=fmin2, fmax=fmax2, faverage=True, mt_adaptive=True, n_jobs=1)
    np.save(os.path.join('path for result files to be saved', i+D), con2)
    con3, freqs, times, n_epochs, n_tapers = spectral_connectivity(label_ts, method='wpli2_debiased', mode='multitaper', sfreq=sfreq, fmin=fmin3, fmax=fmax3, faverage=True, mt_adaptive=True, n_jobs=1)
    np.save(os.path.join('path for result files to be saved', i+D), con3)
