"""Analyze pickle files from the folding.
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
titlesize = 55
xsize = 48
ysize = 48
ticksize = 44
legendsize = 44
er_alpha = 0.25
bar_width = 0.3
bar_alpha = 0.9
lw = 5


def plot(data_br, data_bl):
    pass


def analyze(stuff_l):
    ep_dir = stuff_l['datadir']  # currently just one
    targ_c_dir = stuff_l['goal_c_img_dir']
    targ_d_dir = stuff_l['goal_d_img_dir']

    with open(ep_dir, 'rb') as fh:
        data = pickle.load(fh)
    t_c_img = cv2.imread(targ_c_dir)
    t_d_img = cv2.imread(targ_d_dir)
    assert t_c_img.shape == t_d_img.shape == (56,56,3)

    head_dir = 'tmp/'
    if not os.path.exists(head_dir):
        os.makedirs(head_dir)
    L2_c = []
    L2_d = []
    for idx,(c_img,d_img) in enumerate(zip(data['c_img'], data['d_img'])):
        pth_c = join(head_dir,'c_img_{}.png'.format(str(idx).zfill(3)))
        pth_d = join(head_dir,'d_img_{}.png'.format(str(idx).zfill(3)))
        L2_c.append( np.linalg.norm(c_img - t_c_img) )
        L2_d.append( np.linalg.norm(d_img - t_d_img) )
        cv2.imwrite(pth_c, c_img)
        cv2.imwrite(pth_d, d_img)
    L2_c = np.array(L2_c)
    L2_d = np.array(L2_d)

    print('L2s (c): {}'.format(L2_c))
    print('L2s (d): {}'.format(L2_d))
    print('coverage: {}'.format(data['coverage']))
    print('coverage len: {}'.format(len(data['coverage'])))
    print('c,d_img len:  {}, {}'.format(len(data['c_img']), len(data['d_img'])))
    # Report this?
    print('  (to report)      RGB, Depth')
    min_c = np.min(L2_c[1:])
    min_d = np.min(L2_d[1:])
    print('  start L2:        {:.2f}, {:.2f}'.format(L2_c[0], L2_d[0]))
    print('  min L2 attained: {:.2f}, {:.2f}'.format(min_c, min_d))
    reduct_c = 100.0 * (L2_c[0] - min_c) / L2_c[0]
    reduct_d = 100.0 * (L2_d[0] - min_d) / L2_d[0]
    print('  reduction: {:.1f}%, {:.1f}%'.format(reduct_c, reduct_d))
    data = {
        'L2_c': L2_c,
        'L2_d': L2_d,
    }
    return data


if __name__ == "__main__":
    # See https://github.com/ryanhoque/cloth-visual-mpc/blob/master/vismpc/make_pkl.py for key
    print('\nLoading bottom RIGHT data.')
    br = {
        'datadir':        join('results', 'special', 'ep_005_2020-01-30-17-56.pkl'),
        'goal_c_img_dir': join('goal_imgs', '002-c_img_crop_proc_56.png'),
        'goal_d_img_dir': join('goal_imgs', '002-d_img_crop_proc_56.png'),
        #join('goal_imgs', 'rgbd_corner_56x56_bottomright.pkl'),
    }
    data_br = analyze(br)

    print('\nLoading bottom LEFT data.')
    bl = {
        'datadir':        join('results', 'special', 'ep_004_2020-01-30-17-10.pkl'),
        'goal_c_img_dir': join('goal_imgs', '003-c_img_crop_proc_56.png'),
        'goal_d_img_dir': join('goal_imgs', '003-d_img_crop_proc_56.png'),
        #join('goal_imgs', 'rgbd_corner_56x56_bottomleft.pkl'),
    }
    data_bl = analyze(bl)

    plot(data_br, data_bl)
