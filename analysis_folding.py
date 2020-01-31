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


def _criteria(x, MONTH_BEGIN=9, DAY_BEGIN=7):
    """Filter older entries, `x` is full path.

    I started on the 6th but changed the protocol a bit afterwards.
    """
    # As of Jan 2020 I am using all.
    return True

    # Get only the pickle file name.
    x = os.path.basename(x)

    # Handle first few cases without date.
    assert x[-4:] == '.pkl', x
    x = x[:-4]  # remove `.pkl`
    ss = x.split('_')
    assert ss[0] == 'ep', ss

    if len(ss) == 2:
        # Then this was saved without a date. Only include if 'day begin' was 6.
        return DAY_BEGIN == 6
    else:
        date = (ss[2]).split('-')
        assert len(date) == 5, date
        year, month, day = int(date[0]), int(date[1]), int(date[2])
        assert year == 2020, year
        assert month == 1, month
        #print(x, date, year, month, day, day >= DAY_BEGIN)
        begin = day >= DAY_BEGIN
        return begin


def analyze_single(pth, episode_idx=None):
    target = 'tmp'
    with open(pth, 'r') as fh:
        data = pickle.load(fh)
    print('coverage: {}'.format(data['coverage']) )
    print('coverage len: {}'.format(len(data['coverage'])))
    print('c_img len:    {}'.format(len(data['c_img'])))
    print('d_img len:    {}'.format(len(data['d_img'])))
    print('actions len:  {}'.format(len(data['actions'])))
    for idx,(c_img,d_img) in enumerate(zip(data['c_img'], data['d_img'])):
        pth_c = join(target,'c_img_{}.png'.format(idx))
        pth_d = join(target,'d_img_{}.png'.format(idx))
        cv2.imwrite(pth_c, c_img)
        cv2.imwrite(pth_d, d_img)


def analyze_group(head):
    """Go through all experiments of a certain condition."""
    ep_files = sorted([join(head,x) for x in os.listdir(head) if x[-4:]=='.pkl'])
    ss = defaultdict(list)
    num_counted = 0

    image_path = head.replace('tier','img_tier')
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for ep in ep_files:
        if not _criteria(ep):
            print('SKIPPING {}; it is an older trial.'.format(ep))
            continue
        num_counted += 1

        print('{}'.format(ep))
        with open(ep, 'r') as fh:
            data = pickle.load(fh)
        print('  coverage len {}, and values:   {}'.format(len(data['coverage']),
                    np.array(data['coverage'])))
        print('  c,d_img len: {} {}'.format(len(data['c_img']), len(data['d_img'])))
        print('  actions len: {}'.format(len(data['actions'])))
        print('  max/min/avg/start/end: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
                np.max(data['coverage']), np.min(data['coverage']),
                np.mean(data['coverage']), data['coverage'][0],
                data['coverage'][-1])
        )
        # WAIT --- I think we want max/min/avg AFTER the starting coverage!
        ss['max'].append( np.max(data['coverage'][1:]) )
        ss['min'].append( np.min(data['coverage'][1:]) )
        # Special case ... I think I recorded duplicate coverage.
        # Update: actually I think I got rid of this after the first case (T1,
        # episode1, DAgger) so I should change this if it poses a problem.
        if (len(data['actions']) == 9) and (len(data['d_img']) == 11):
            ss['avg'].append( np.mean(data['coverage'][1:-1]) )  # duplicate at end, so up to -1
        else:
            ss['avg'].append( np.mean(data['coverage'][1:]) )
        ss['beg'].append( data['coverage'][0] )
        ss['end'].append( data['coverage'][-1] )
        ss['nacts'].append( len(data['actions']) )

        # Save images here.
        nc = num_counted  # Use as an episode index, sort of ...
        for t, (cimg, dimg) in enumerate(zip(data['c_img'],data['d_img'])):
            c_path = join(image_path,
                          'c_img_ep_{}_t_{}.png'.format(str(nc).zfill(2), str(t).zfill(2)))
            d_path = join(image_path,
                          'd_img_ep_{}_t_{}.png'.format(str(nc).zfill(2), str(t).zfill(2)))
            cv2.imwrite(filename=c_path, img=cimg)
            cv2.imwrite(filename=d_path, img=dimg)

    # Multiply by 100 :-)
    for key in ss.keys():
        if key != 'nacts':
            ss[key] = np.array(ss[key]) * 100
    print('\nOverall stats across {} trials:'.format(num_counted))
    print('start: {:.1f} +/- {:.1f}'.format(np.mean(ss['beg']), np.std(ss['beg'])) )
    print('end:   {:.1f} +/- {:.1f}'.format(np.mean(ss['end']), np.std(ss['end'])) )
    print('max:   {:.1f} +/- {:.1f}'.format(np.mean(ss['max']), np.std(ss['max'])) )
    print('min:   {:.1f} +/- {:.1f}'.format(np.mean(ss['min']), np.std(ss['min'])) )
    print('avg:   {:.1f} +/- {:.1f}'.format(np.mean(ss['avg']), np.std(ss['avg'])) )
    print('nacts: {:.1f} +/- {:.1f}'.format(np.mean(ss['nacts']), np.std(ss['nacts'])) )

    # In readable format for LaTeX: (Update Jan 2020, don't want avg, but max acts)
    #_str = '& {:.1f} +/- {:.1f} & {:.1f} +/- {:.1f} & {:.1f} +/- {:.1f} & {:.1f} +/- {:.1f} \\\\'.format(
    #        np.mean(ss['beg']),np.std(ss['beg']),
    #        np.mean(ss['end']),np.std(ss['end']),
    #        np.mean(ss['max']),np.std(ss['max']),
    #        np.mean(ss['avg']),np.std(ss['avg']),
    #)
    _str = '& {:.1f} +/- {:.0f} & {:.1f} +/- {:.0f} & {:.1f} +/- {:.0f} & {:.1f} +/- {:.0f} \\\\'.format(
            np.mean(ss['beg']),np.std(ss['beg']),
            np.mean(ss['end']),np.std(ss['end']),
            np.mean(ss['max']),np.std(ss['max']),
            np.mean(ss['nacts']),np.std(ss['nacts']),
    )
    print(_str)
    return _str, num_counted


def analyze_icra_submission():
    print('\n*********************************************')
    print('ANALYZING TIER 1 COLOR AND DEPTH ON YELLOW')
    print('*********************************************\n')
    head = join('results', 'tier1_color_yellowcloth')
    str0, nb0 = analyze_group(head)
    print('Over {} episodes. For LaTeX:\nstart, end, max, mean'.format(nb0))
    print('T1 Color on Yellow '+ str0 +'\n')
    head = join('results', 'tier1_depth_yellowcloth')
    str0, nb0 = analyze_group(head)
    print('Over {} episodes. For LaTeX:\nstart, end, max, mean'.format(nb0))
    print('T1 Depth on Yellow '+ str0)

    print('\n*********************************************')
    print('ANALYZING TIER 1 COLOR')
    print('*********************************************\n')
    head = join('results', 'tier1_color')
    str1, nb1 = analyze_group(head)

    print('\n*********************************************')
    print('ANALYZING TIER 1 DEPTH')
    print('*********************************************\n')
    head = join('results', 'tier1_depth')
    str2, nb2 = analyze_group(head)

    print('\n*********************************************')
    print('ANALYZING TIER 2 COLOR')
    print('*********************************************\n')
    head = join('results', 'tier2_color')
    str3, nb3 = analyze_group(head)

    print('\n*********************************************')
    print('ANALYZING TIER 2 DEPTH')
    print('*********************************************\n')
    head = join('results', 'tier2_depth')
    str4, nb4 = analyze_group(head)

    print('\n*********************************************')
    print('ANALYZING TIER 3 COLOR')
    print('*********************************************\n')
    head = join('results', 'tier3_color')
    str5, nb5 = analyze_group(head)

    print('\n*********************************************')
    print('ANALYZING TIER 3 DEPTH')
    print('*********************************************\n')
    head = join('results', 'tier3_depth')
    str6, nb6 = analyze_group(head)

    print('\nNumber of trials we record:')
    print(nb1, nb2, nb3, nb4, nb5, nb6)
    print('\n\nCopy and paste this for LaTeX:\nstart, end, max, mean')
    print('T1 RGB  '+ str1)
    print('T1 Dep. '+ str2)
    print('T2 RGB  '+ str3)
    print('T2 Dep. '+ str4)
    print('T3 RGB  '+ str5)
    print('T3 Dep. '+ str6)


if __name__ == "__main__":
    #pth = join('results','tier1_color','ep_000.pkl')
    #analyze_single(pth)
    #analyze_icra_submission()

    print('\n*********************************************')
    print('ANALYZING TIER 1 DAGGER')
    print('*********************************************\n')
    head = join('results', 'tier1_rgbd')
    str1, nb1 = analyze_group(head)

    print('\n*********************************************')
    print('ANALYZING TIER 1 VF')
    print('*********************************************\n')
    head = join('results', 'vf_tier1_rgbd')
    str2, nb2 = analyze_group(head)

    print('\n*********************************************')
    print('ANALYZING TIER 2 COLOR')
    print('*********************************************\n')
    head = join('results', 'tier2_rgbd')
    str3, nb3 = analyze_group(head)

    print('\n*********************************************')
    print('ANALYZING TIER 2 VF')
    print('*********************************************\n')
    head = join('results', 'vf_tier2_rgbd')
    str4, nb4 = analyze_group(head)

    print('\n*********************************************')
    print('ANALYZING TIER 3 COLOR')
    print('*********************************************\n')
    head = join('results', 'tier3_rgbd')
    str5, nb5 = analyze_group(head)

    print('\n*********************************************')
    print('ANALYZING TIER 3 VF')
    print('*********************************************\n')
    head = join('results', 'vf_tier3_rgbd')
    str6, nb6 = analyze_group(head)

    print('\nNumber of trials we record:')
    print(nb1, nb2, nb3, nb4, nb5, nb6)
    print('\n\nCopy and paste this for LaTeX:\nstart, end, max, nacts')
    print('T1 IL '+ str1)
    print('T1 VF '+ str2)
    print('T2 IL '+ str3)
    print('T2 VF '+ str4)
    print('T3 IL '+ str5)
    print('T3 VF '+ str6)
