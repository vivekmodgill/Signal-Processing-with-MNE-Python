import os
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle

j = ['CC120120', 'CC120137', 'CC120264', 'CC122016', 'CC220107', 'CC220203', 'CC221336', 'CC222120', 'CC320022', 'CC320059', 'CC320814', 'CC321544', 'CC410119', 'CC410182', 'CC420061', 'CC420286', 'CC420435', 'CC420589', 'CC420776', 'CC510255', 'CC510438', 'CC520239']
k = '.fif'
A = '-trans.fif'
Z = '/transdef_mf2pt2_rest_raw.fif'
D = '.npy'
for i in j:
    subjects_dir = '/home/siddharth/Downloads/freesurfer/subjects/'
    subject = 'collin'
    bem = '/home/siddharth/Downloads/freesurfer/subjects/collin/bem_sol.fif'
    src = '/home/siddharth/Downloads/freesurfer/subjects/collin/src.fif'
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
    label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip', return_generator=False)
    c = []
    for y in range(0, 68):
        a = []
        for h in range(0, len(label_ts)):
            a.append(label_ts[h][y])
        b = np.reshape(a, (1, (451*len(label_ts))))
        globals()['source%s' % y] = b
        c.append(globals()['source%s' % y][0])
        np.save(os.path.join('/home/siddharth/Vivek/Working/Data/results/reconstructed_source_time_series/new_22_added/', i+D), c)
