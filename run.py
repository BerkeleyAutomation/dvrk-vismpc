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


def action_correction(act, freq, c_img_100x100, display=True):
    """Action correction if we just barely miss the cloth.

    Should only be called if we trigger a structural similiarity check between
    two consecutive images. Here's how this works:

    (1) Crop the 100x100 image to only consider the foam rubber. We do this
    even if we have 56x56 images as much of the coverage code is heavily tuned
    towards 100x100-sized images.

    (2) Measure pixels of the cloth, thresholded.

    (3) Measure the center of that cloth region.

    (4) The 'direction vector' goes from the pick point to that cloth center.

    (5) Adjust the pick point based on the direction vector, and multiply by
    freq, so that we go further if needed. The deltas in x and y are the same.

    To make this better we could better tune the distance to travel.
    """
    print('\nWe are correcting for action: {}, freq {}'.format(act, freq))
    assert c_img_100x100.shape == (100,100,3)
    c_img_orig = c_img_100x100.copy()
    bounding_dims = (10,90,10,90)  # copy whatever's in `utils.py`
    min_x, max_x, min_y, max_y = bounding_dims
    c_img = c_img_orig[min_x:max_x,min_y:max_y,:]
    CHANGE_CONST = 5
    _THRESHOLD = 100

    # Convert from action space to pixels (with some caveats). Ah, with B we may
    # get a B-sized list but with index at B, so do -1?
    B = c_img.shape[0] - 1
    XX = B / 2
    pix_pick = (act[0] * XX + XX,
                act[1] * XX + XX)
    pix_targ = ((act[0]+act[2]) * XX + XX,
                (act[1]+act[3]) * XX + XX)

    # thresh is grayscale image. Should be 255 for white values, right?
    # Since the cloth is BLACK we want to check if thresh[pick_point] is 0.
    imgray = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, _THRESHOLD, 255, cv2.THRESH_BINARY)
    tot_w = np.sum(thresh >= 255.0)
    tot_b = np.sum(thresh <= 0.0)
    print('resized c_img, thresh: {}, {}'.format(c_img, thresh.shape)) # should be same
    print('  white, black pixels: {}, {},  sum {}'.format(tot_w, tot_b, tot_w+tot_b))

    # Find out if pick point is on the cloth or not, `thresh` is a numpy array.
    # Ah, it can be inaccurate due to boundary conditions? Not sure how to easily fix.
    x, y = int(pix_pick[0]), int(pix_pick[1])
    if (thresh[B-y, x] <= 0.0):
        # Tricky, not sure the best way, we may just want to keep going but change
        # the delta magnitudes so they're smaller?
        print('ON the cloth, arr[{},{}], thresh: {}'.format(B-x, y, _THRESHOLD))
        print('  WARNING: CODE THINKS PICK POINT IS ON THE CLOTH!')
        on_cloth = True
    else:
        print('NOT on the cloth, arr[{},{}], thresh: {}'.format(B-x, y, _THRESHOLD))
        on_cloth = False

    # some more threshold stuff, find the center pixel?
    # I know it has some of the boundary stuff, but part of that is unavoidable imo.
    cloth_indices = np.argwhere(thresh < 0.1)
    avg_xy = np.mean(cloth_indices, axis=0)
    print('  avg pixels of threshold: {:.1f},{:.1f} from np mean {}'.format(
            avg_xy[0], avg_xy[1], avg_xy.shape))
    avg_x_th = int(avg_xy[0])
    avg_y_th = int(avg_xy[1])

    # Annotate with 'opencv y', SO MUST INVERT for visualizing pick points.
    pix_pick = int(pix_pick[0]), B - int(pix_pick[1])
    pix_targ = int(pix_targ[0]), B - int(pix_targ[1])
    print('pix_pick for opencv: {}'.format(pix_pick))
    c_img = cv2.circle(c_img, center=pix_pick, radius=4, color=cfg.RED, thickness=-1)

    # CENTER OF THE FABRIC. Annoying, tests show we need the reverse, y and then x.
    c_img = cv2.circle(c_img, center=(avg_y_th,avg_x_th), radius=4, color=cfg.WHITE, thickness=-1)

    # OK but now we have pixels original, and pixels target. Get vector direction.
    dir_vector = np.array([avg_y_th - pix_pick[0],
                           pix_pick[1] - avg_x_th])  # yeah yeah we have to subtract
    Mag = np.linalg.norm(dir_vector)
    dir_norm = dir_vector / Mag
    print('vector direction: {}'.format(dir_vector))
    print('      magnitude:  {:.1f}'.format(Mag))
    print('      normalized: {}'.format(dir_norm))
    print('      interpreted as direction we should adjust pick point')
    c_img = cv2.line(c_img, pt1=(avg_y_th,avg_x_th), pt2=pix_pick, color=cfg.BLACK, thickness=1)

    # PICK POINT THAT IS RE-MAPPED. Get it in [-1,1] then convert to pixels.
    # The old pick point was at (act[0], act[1]). Also, in the actual code, if
    # we call this many times consecutively, MULTIPLY `change_{x,y}` by `freq`.
    change_x = (dir_norm[0] / CHANGE_CONST) * freq
    change_y = (dir_norm[1] / CHANGE_CONST) * freq
    if on_cloth:
        change_x /= 2.0
        change_y /= 2.0
    print('      change actx space: {:.2f}'.format(change_x))
    print('      change acty space: {:.2f}'.format(change_x))
    new_pick = (act[0]+change_x, act[1]+change_y)
    pix_new = (int(new_pick[0]*XX + XX),
               int(B - (new_pick[1]*XX + XX)))
    print('      old pick pt: {}'.format((act[0],act[1])))
    print('      new pick pt: {}'.format(new_pick))
    c_img = cv2.circle(c_img, center=pix_new, radius=4, color=cfg.BLUE, thickness=-1)

    if display:
        # Display a bunch of images for debugging.
        display_img = Image.new(mode='RGB', size=(400,200), color=200)
        draw = ImageDraw.Draw(display_img)
        display_img.paste(PIL.Image.fromarray(c_img),      (  0, 0)) # resized + annotated
        display_img.paste(PIL.Image.fromarray(thresh),     (100, 0)) # detect cloth
        display_img.paste(PIL.Image.fromarray(fake_image), (200, 0)) # detect coverage
        display_img.paste(PIL.Image.fromarray(c_img_orig), (  0, 100)) # original
        coverage = 1.0 - (np.sum(is_not_covered) / float(is_not_covered.size))
        cv2.imshow("coverage: {:.3f}".format(coverage), np.array(display_img) )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    new_act = np.array( [new_pick[0], new_pick[1], act[2], act[3]] )
    print('our new corrected action: {} w/freq {}'.format(new_act, freq))
    return new_act


def run(args, p, img_shape, save_path):
    """Run one episode, record statistics, etc.
    """
    stats = defaultdict(list)
    COVERAGE_SUCCESS = 0.92
    SS_THRESH = 0.95
    dumb_correction = False
    freq = 0

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
        stats['c_img_100x100'].append(c_img_100x100)
        stats['c_img'].append(c_img)
        stats['d_img'].append(d_img)

        if coverage >= COVERAGE_SUCCESS:
            print('\nCOVERAGE SUCCESS: {:.3f} >= {:.3f}, exiting ...\n'.format(
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
            if diff_ss_c > SS_THRESH:
                freq += 1
                print('NOTE structural similiarity exceeds {}'.format(SS_THRESH))
                if dumb_correction:
                    action[0] = action[0] * (0.9 ** freq)
                    action[1] = action[1] * (0.9 ** freq)
                    print('revised action after \'dumb compression\': {}, freq {}'.format(
                            action, freq))
                else:
                    action = action_correction(action, freq, c_img_100x100)
            else:
                freq = 0

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
    # Edit: need to handle case of when we exceeded coverage after 9th action.
    if len(stats['c_img']) == args.max_ep_length and coverage < COVERAGE_SUCCESS:
        assert len(stats['coverage']) == args.max_ep_length, len(stats['coverage'])
        i = args.max_ep_length

        # Results from the neural network -- still use to check if we get a NEW image.
        results = U.get_net_results()
        print('Just get the very last image we need! (Have {} so far)'.format(len(results)))
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
        stats['c_img_100x100'].append(c_img_100x100)
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
    p.set_position_origin([0.005, 0.033, -0.060], 0, 'deg')

    # Run one episode.
    stats = run(args, p, img_shape=img_shape, save_path=save_path)
