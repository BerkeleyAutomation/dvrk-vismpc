"""Use this for the main experiments. It runs one episode only.
"""
import argparse
import os
import sys
import time
import datetime
import cv2
import pickle
import logging
import numpy as np
np.set_printoptions(suppress=True)
from collections import defaultdict
from os.path import join
from skimage.measure import compare_ssim
# Stuff from our code base.
from dvrkClothSim import dvrkClothSim
#sys.path.append('call_network')
#import load_config as cfg
import config as C
import utils as U


def run(args, p, img_shape, save_path):
    """Run one episode, record statistics, etc.

    TODO: record timing?
    """
    stats = defaultdict(list)
    COVERAGE_SUCCESS = 0.92
    exponent = 0

    for i in range(args.max_ep_length):
        print('\n*************************************')
        print('ON TIME STEP (I.E., ACTION) NUMBER {}'.format(i+1))
        print('*************************************\n')

        # ----------------------------------------------------------------------
        # STEP 1: move to the tab with the camera, click ENTER. This will save
        # images, and also call the neural network code to produce an action.
        # Then, load the action.
        # ----------------------------------------------------------------------
        results = U.get_net_results()  # Results from the neural network.
        print('Waiting for one more result to the {} we have so far'.format(len(results)))
        while len(results) == i:
            time.sleep(1)
            results = U.get_net_results()
        assert len(results) >= i
        assert len(results) == i+1, '{} vs {}, {}'.format(i, results, len(results))

        # ----------------------------------------------------------------------
        # STEP 2: load those images, using same code as in the network loading
        # code. If coverage is high enough, exit now. Note: we load 56x56 for
        # training but use 100x100 for computing coverage as we have several
        # values that are heavily tuned for that.
        # ----------------------------------------------------------------------
        c_path_100x100 = join(C.DVRK_IMG_PATH,
                              '{}-c_img_crop_proc.png'.format(str(i).zfill(3)))
        c_path = join(C.DVRK_IMG_PATH,
                      '{}-c_img_crop_proc_56.png'.format(str(i).zfill(3)))
        d_path = join(C.DVRK_IMG_PATH,
                      '{}-d_img_crop_proc_56.png'.format(str(i).zfill(3)))
        c_img_100x100 = cv2.imread(c_path_100x100)
        coverage = U.calculate_coverage(c_img_100x100)  # tuned for 100x100
        c_img = cv2.imread(c_path)
        d_img = cv2.imread(d_path)
        U.single_means(c_img, depth=False)
        U.single_means(d_img, depth=True)
        assert c_img.shape == d_img.shape == img_shape
        assert args.use_rgbd
        # img = np.dstack( (c_img, d_img[:,:,0]) )  # we don't call net code here

        # Ensures we save the final image in case we exit and get high coverage.
        # Make sure it happens BEFORE the `break` command below so we get final imgs.
        stats['coverage'].append(coverage)
        stats['c_img'].append(c_img)
        stats['d_img'].append(d_img)

        if coverage > COVERAGE_SUCCESS:
            print('\nCOVERAGE SUCCESS: {:.3f} > {:.3f}, exiting ...\n'.format(
                    coverage, COVERAGE_SUCCESS))
            break
        else:
            print('\ncurrent coverage: {:.3f}\n'.format(coverage))
        # Before Jan 2020, action selection happened later. Now it happens earlier.
        #print('  now wait a few seconds for network to run')
        #time.sleep(5)

        # ----------------------------------------------------------------------
        # STEP 3: show the output of the action to the human. HUGE ASSUMPTION:
        # that we can just look at the last item of `results` list.
        # ----------------------------------------------------------------------
        action = np.loadtxt(results[-1])
        print('neural net says: {}'.format(action))
        stats['actions'].append(action)

        # ----------------------------------------------------------------------
        # STEP 3.5, only if we're not on the first action, if current image is
        # too similar to the old one, move the target points closer towards the
        # center of the cloth plane. An approximation but likely 'good enough'.
        # It does assume the net would predict a similiar action, though ...
        # ----------------------------------------------------------------------
        if i > 0:
            # AH! Go to -2 because I modified code to append (c_img,d_img) above.
            prev_c = stats['c_img'][-2]
            prev_d = stats['d_img'][-2]
            diff_l2_c = np.linalg.norm(c_img - prev_c) / np.prod(c_img.shape)
            diff_l2_d = np.linalg.norm(d_img - prev_d) / np.prod(d_img.shape)
            diff_ss_c = compare_ssim(c_img, prev_c, multichannel=True)
            diff_ss_d = compare_ssim(d_img[:,:,0], prev_d[:,:,0])
            print('\n  (c) diff L2: {:.3f}'.format(diff_l2_c))
            print('  (d) diff L2: {:.3f}'.format(diff_l2_d))
            print('  (c) diff SS: {:.3f}'.format(diff_ss_c))
            print('  (d) diff SS: {:.3f}\n'.format(diff_ss_d))
            stats['diff_l2_c'].append(diff_l2_c)
            stats['diff_l2_d'].append(diff_l2_d)
            stats['diff_ss_c'].append(diff_ss_c)
            stats['diff_ss_d'].append(diff_ss_d)

            # Apply action 'compression'? A 0.95 cutoff empirically works well.
            ss_thresh = 0.95
            if diff_ss_c > ss_thresh:
                exponent += 1
                print('NOTE structural similiarity exceeds {}'.format(ss_thresh))
                action[0] = action[0] * (0.9 ** exponent)
                action[1] = action[1] * (0.9 ** exponent)
                print('revised action after \'compression\': {} w/exponent {}'.format(
                        action, exponent))
            else:
                exponent = 0

        # ----------------------------------------------------------------------
        # STEP 4. If the output would result in a dangerous position, human
        # stops by hitting ESC key. Otherwise, press any other key to continue.
        # The human should NOT normally be using this !!
        # ----------------------------------------------------------------------
        title = '{} -- ESC TO CANCEL (Or if episode done)'.format(action)
        if args.use_rgbd:
            stacked_img = np.hstack( (c_img, d_img) )
            exit = U.call_wait_key( cv2.imshow(title, stacked_img) )
        elif args.use_color:
            exit = U.call_wait_key( cv2.imshow(title, c_img) )
        else:
            exit = U.call_wait_key( cv2.imshow(title, d_img) )
        cv2.destroyAllWindows()
        if exit:
            print('Warning: why are we exiting here?')
            print('It should exit naturally due to (a) coverage or (b) time limits.')
            print('Make sure I clear any results that I do not want to record.')
            break

        # ----------------------------------------------------------------------
        # STEP 5: Watch the robot do its action. Terminate the script if the
        # resulting action makes things fail spectacularly.
        # ----------------------------------------------------------------------
        x  = action[0]
        y  = action[1]
        dx = action[2]
        dy = action[3]
        start_t = time.time()
        U.move_p_from_net_output(x, y, dx, dy,
                                 row_board=C.ROW_BOARD,
                                 col_board=C.COL_BOARD,
                                 data_square=C.DATA_SQUARE,
                                 p=p)
        elapsed_t = time.time() - start_t
        stats['act_time'].append(elapsed_t)
        print('Finished executing action in {:.2f} seconds.'.format(elapsed_t))

    # If we ended up using all actions above, we really need one more image.
    if len(stats['c_img']) == args.max_ep_length:
        assert len(stats['coverage']) == args.max_ep_length, len(stats['coverage'])
        i = args.max_ep_length

        # Results from the neural network -- still use to check if we get a NEW image.
        results = U.get_net_results()
        print('Waiting for one more result to the {} we have so far'.format(len(results)))
        while len(results) == i:
            time.sleep(1)
            results = U.get_net_results()
        assert len(results) >= i
        assert len(results) == i+1, '{} vs {}, {}'.format(i, results, len(results))

        # Similar loading as earlier.
        c_path_100x100 = join(C.DVRK_IMG_PATH,
                              '{}-c_img_crop_proc.png'.format(str(i).zfill(3)))
        c_path = join(C.DVRK_IMG_PATH,
                      '{}-c_img_crop_proc_56.png'.format(str(i).zfill(3)))
        d_path = join(C.DVRK_IMG_PATH,
                      '{}-d_img_crop_proc_56.png'.format(str(i).zfill(3)))
        c_img_100x100 = cv2.imread(c_path_100x100)
        coverage = U.calculate_coverage(c_img_100x100)  # tuned for 100x100
        c_img = cv2.imread(c_path)
        d_img = cv2.imread(d_path)
        assert c_img.shape == d_img.shape == img_shape

        # Record final stats.
        stats['coverage'].append(coverage)
        stats['c_img'].append(c_img)
        stats['d_img'].append(d_img)
        print('(for full length episode) final coverage: {:.3f}'.format(coverage))

    # Final book-keeping and return statistics.
    stats['coverage'] = np.array(stats['coverage'])
    print('\nEPISODE DONE!')
    print('  coverage: {}'.format(stats['coverage']))
    print('  len(coverage): {}'.format(len(stats['coverage'])))
    print('  len(c_img): {}'.format(len(stats['c_img'])))
    print('  len(d_img): {}'.format(len(stats['d_img'])))
    print('  len(actions): {}'.format(len(stats['actions'])))
    print('All done with episode! Saving stats to: {}'.format(save_path))
    with open(save_path, 'wb') as fh:
        pickle.dump(stats, fh)
    return stats


if __name__ == "__main__":
    # I would just set all to reasonable defaults, or put them in the config file.
    parser= argparse.ArgumentParser()
    parser.add_argument('--use_other_color', action='store_true')
    #parser.add_argument('--use_color', type=int) # 1 = True
    parser.add_argument('--tier', type=int)
    parser.add_argument('--max_ep_length', type=int, default=10)
    args = parser.parse_args()
    assert args.tier is not None
    args.use_rgbd = True
    print('Running with arguments:\n{}'.format(args))
    assert os.path.exists(C.CALIB_FILE), C.CALIB_FILE

    # With newer code, we run the camera script in a separate file.
    # The camera script will save in the dvrk config directory. Count up index.
    #cam = camera.RGBD()
    img_shape = (56,56,3)

    # Assume a clean directory where we store things FOR THIS EPISODE ONLY.
    dvrk_img_paths = U.get_sorted_imgs()
    net_results = U.get_net_results()
    if len(dvrk_img_paths) > 0:
        print('There are {} images in {}. Please remove it/them.'.format(
                len(dvrk_img_paths), C.DVRK_IMG_PATH))
        print('It should be empty to start an episode.')
        sys.exit()
    if len(net_results) > 0:
        print('There are {} results in {}. Please remove it/them.'.format(
                len(net_results), C.DVRK_IMG_PATH))
        print('It should be empty to start an episode.')
        sys.exit()

    # Determine the file name to save, for permanent storage.
    if args.use_rgbd:
        save_path = join('results', 'tier{}_rgbd'.format(args.tier))
    else:
        raise ValueError()
    # Ignore for now, but may re-visit when we do benchmarks with earlier networks?
    #if args.use_color:
    #    if args.use_other_color:
    #        save_path = join('results', 'tier{}_color_yellowcloth'.format(args.tier))
    #    else:
    #        save_path = join('results', 'tier{}_color'.format(args.tier))
    #else:
    #    if args.use_other_color:
    #        save_path = join('results', 'tier{}_depth_yellowcloth'.format(args.tier))
    #    else:
    #        save_path = join('results', 'tier{}_depth'.format(args.tier))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    count = len([x for x in os.listdir(save_path) if 'ep_' in x and '.pkl' in x])
    save_path = join(save_path,
        'ep_{}_{}.pkl'.format(str(count).zfill(3), U.get_date())
    )
    print('Saving to: {}'.format(save_path))

    # Set up the dVRK. Larger y axis value = avoids shadow in our setup.
    p = dvrkClothSim()
    p.set_position_origin([0.003, 0.025, -0.060], 0, 'deg')

    # Run one episode.
    stats = run(args, p, img_shape=img_shape, save_path=save_path)
