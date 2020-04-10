import numpy as np
import matplotlib.pyplot as plt
import os
import mne
from mne.connectivity import envelope_correlation
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.preprocessing import compute_proj_ecg, compute_proj_eog

j = ['CC120120', 'CC120137', 'CC120264', 'CC122016', 'CC220107', 'CC220203', 'CC221336', 'CC222120', 'CC320022', 'CC320059', 'CC320814', 'CC321544', 'CC410119', 'CC410182', 'CC420061', 'CC420286', 'CC420435', 'CC420589', 'CC420776', 'CC510255', 'CC510438', 'CC520239']
k = '.fif'
A = '-trans.fif'
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
    raw.filter(0, 3.5)
    events = mne.make_fixed_length_events(raw, duration=5.)
    epochs = mne.Epochs(raw, events=events, tmin=0, tmax=.5, baseline=None, reject=dict(), preload=True)
    del raw
    src = mne.read_source_spaces(src)
    fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
    inv = make_inverse_operator(epochs.info, fwd, cov)
    del fwd, src
    labels = mne.read_labels_from_annot(subject, 'aparc', subjects_dir=subjects_dir)
    epochs.apply_hilbert()
    stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9., method='sLORETA', pick_ori='normal', return_generator=True)
    label_ts = mne.extract_label_time_course(stcs, labels, inv['src'], return_generator=True)
    corr = envelope_correlation(label_ts)
    np.fill_diagonal(corr, 0)
    b = np.tril(corr, k=0)
    np.save(os.path.join('/home/siddharth/Vivek/Working/Data/results/En_Corr/delta/new_22_added', i+D), b)

