import numpy as np
import scipy
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
import os

j = ['CC120120', 'CC120137', 'CC120264', 'CC122016', 'CC220107', 'CC220203', 'CC221336', 'CC222120', 'CC320022', 'CC320059', 'CC320814', 'CC321544', 'CC410119', 'CC410182', 'CC420061', 'CC420286', 'CC420435', 'CC420589', 'CC420776', 'CC510255', 'CC510438', 'CC520239']
k = '.fif'
A = '-trans.fif'
Z = '/rest/transdef_mf2pt2_rest_raw.fif'
D = '.npy'

for i in j:
    subjects_dir = '/home/siddharth/Downloads/freesurfer/subjects/'
    subject = 'collin'
    bem = os.path.join('/home/siddharth/Downloads/freesurfer/subjects/collin/bem_sol.fif')
    src = os.path.join('/home/siddharth/Downloads/freesurfer/subjects/collin/src.fif')
    trans = os.path.join('/home/siddharth/Vivek/Working/Data/fiducials/new_22_fiducials/', i+A)
    raw = mne.io.read_raw_fif(os.path.join('/home/siddharth/Work/vivek/', i+k), verbose='error')
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
    np.save(os.path.join('/home/siddharth/Vivek/Working/Data/results/wPLI/delta/new_22_added/', i+D), con0)
    con1, freqs, times, n_epochs, n_tapers = spectral_connectivity(label_ts, method='wpli2_debiased', mode='multitaper', sfreq=sfreq, fmin=fmin1, fmax=fmax1, faverage=True, mt_adaptive=True, n_jobs=1)
    np.save(os.path.join('/home/siddharth/Vivek/Working/Data/results/wPLI/theta/new_22_added/', i+D), con1)
    con2, freqs, times, n_epochs, n_tapers = spectral_connectivity(label_ts, method='wpli2_debiased', mode='multitaper', sfreq=sfreq, fmin=fmin2, fmax=fmax2, faverage=True, mt_adaptive=True, n_jobs=1)
    np.save(os.path.join('/home/siddharth/Vivek/Working/Data/results/wPLI/alpha/new_22_added/', i+D), con2)
    con3, freqs, times, n_epochs, n_tapers = spectral_connectivity(label_ts, method='wpli2_debiased', mode='multitaper', sfreq=sfreq, fmin=fmin3, fmax=fmax3, faverage=True, mt_adaptive=True, n_jobs=1)
    np.save(os.path.join('/home/siddharth/Vivek/Working/Data/results/wPLI/beta/new_22_added/', i+D), con3)








