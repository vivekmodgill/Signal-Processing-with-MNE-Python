import os
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle

j = ['file_names_without_extension']
k = '.fif'
D = '.npy'
for i in j:
    subjects_dir = 'path of the freesurfer subject directory'
    subject = 'subject name'
    bem = 'BEM solution file path'
    src = 'src solution file path'
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
    label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip', return_generator=False)
    c = []
    for y in range(0, 68):
        a = []
        for h in range(0, len(label_ts)):
            a.append(label_ts[h][y])
        b = np.reshape(a, (1, (451*len(label_ts))))
        globals()['source%s' % y] = b
        c.append(globals()['source%s' % y][0])
        np.save(os.path.join('path for result files to be saved', i+D), c)
