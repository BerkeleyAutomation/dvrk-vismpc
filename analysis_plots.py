"""Analyze pickle files.

It's like the other analysis script, except I use this to make some plots.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import os
import cv2
import sys
import pickle
import time
import numpy as np
np.set_printoptions(precision=4, linewidth=200)
from os.path import join
from collections import defaultdict
SAVE_IMAGES = False

# matplotlib
titlesize = 42
xsize = 40
ysize = 40
ticksize = 36
legendsize = 36
er_alpha = 0.25
bar_width = 0.3
bar_alpha = 0.9
lw = 5


def plot(ss_il, ss_vf):
    """We have lists of the pickle files of IL and VF."""
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharey='row',
                           figsize=(11*ncols, 8*nrows))
    #gridspec_kw={'width_ratios': [1, 1.5]}) # can do if needed

    ax[0,0].hist(ss_il)
    ax[0,1].hist(ss_vf)

    ax[0,0].set_xlim([0.0, 0.6])
    ax[0,1].set_xlim([0.0, 0.6])

    # Bells and whistles
    ax[0,0].set_xlabel('Delta Magnitudes', fontsize=ysize)
    ax[0,1].set_xlabel('Delta Magnitudes', fontsize=ysize)
    ax[0,0].set_ylabel('Frequency', fontsize=ysize)
    ax[0,0].set_title('Imitation Learning', fontsize=titlesize)
    ax[0,1].set_title('Visuospatial Foresight', fontsize=titlesize)

    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
            # I think it's better to share axes in the x direction to be
            # consistent with steps, but doing so removes the axis ticks. This
            # reverts it so we get the ticks on all the axis.
            #ax[r,c].xaxis.set_tick_params(which='both', labelbottom=True)

    plt.tight_layout()
    figname = 'fig_action_magnitudes.png'.format()
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


def stats(heads_l):
    ep_files = []
    for h in heads_l:
        files = sorted([join(h,x) for x in os.listdir(h) if x[-4:]=='.pkl'])
        ep_files.extend(files)
    print('\nLoaded {} files'.format(len(ep_files)))

    # what we actually want to plot
    ss_data = []

    for ep in ep_files:
        print('episode: {}'.format(ep))
        with open(ep, 'r') as fh:
            data = pickle.load(fh)
        ep_len = len(data['actions'])
        print('  max/min/avg/start/end: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
                np.max(data['coverage']), np.min(data['coverage']),
                np.mean(data['coverage']), data['coverage'][0],
                data['coverage'][-1])
        )
        #actions_arr = np.array(data['actions'])
        #assert actions_arr.shape == (ep_len, 4), actions_arr.shape
        #ss['actions'].append(actions_arr)
        for a in data['actions']:
            magnitude = np.linalg.norm( np.array([a[2],a[3]]) )
            ss_data.append(magnitude)

    print('ss_data length: {}'.format(len(ss_data)))
    ss_data = np.array(ss_data)
    print(ss_data)
    return ss_data


if __name__ == "__main__":
    il_heads = [
        join('results', 'tier1_rgbd'),
        join('results', 'tier2_rgbd'),
        join('results', 'tier3_rgbd'),
    ]
    vf_heads = [
        join('results', 'vf_tier1_rgbd'),
        join('results', 'vf_tier2_rgbd'),
        join('results', 'vf_tier3_rgbd'),
    ]
    il_stats = stats(il_heads)
    vf_stats = stats(vf_heads)
    plot(il_stats, vf_stats)
