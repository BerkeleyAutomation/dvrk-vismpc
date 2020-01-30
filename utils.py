"""Shared methods, to be loaded in other code.
"""
import os
import sys
## Maybe? Might need this if running `python utils.py`.
## Actually, if running `python utils.py` just use system python.
#for p in sys.path:
#    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
#        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import time
import numpy as np
from os import path
from os.path import join
import datetime
sys.path.append('call_network')
import load_config as cfg
import PIL
from PIL import (Image, ImageDraw)


# Useful constants.
ESC_KEYS = [27, 1048603]
MILLION = float(10**6)


def rad_to_deg(rad):
    return np.array(rad) * 180./np.pi


def deg_to_rad(deg):
    return np.array(deg) * np.pi/180.


# Duplicated in `call_network/load_net`, careful if changing!
def get_sorted_imgs():
    res = sorted(
        [join(cfg.DVRK_IMG_PATH,x) for x in os.listdir(cfg.DVRK_IMG_PATH) \
            if x[-4:]=='.png']
    )
    return res


# Duplicated in `call_network/load_net`, careful if changing!
def get_net_results():
    net_results = sorted(
        [join(cfg.DVRK_IMG_PATH,x) for x in os.listdir(cfg.DVRK_IMG_PATH) \
            if 'result_' in x and '.txt' in x]
    )
    return net_results


def debug_print_img(img):
    print('  max: {:.3f}'.format(np.max(img)))
    print('  min: {:.3f}'.format(np.min(img)))
    print('  mean: {:.3f}'.format(np.mean(img)))
    print('  medi: {:.3f}'.format(np.median(img)))
    print('  std: {:.3f}'.format(np.std(img)))


def normalize(v):
    norm=np.linalg.norm(v, ord=2)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def save_image_numbers(head, img, indicator=None, debug=False):
    """Save image in a directory, but numbered at the end.

    Example, indicator might be `c_img`. Note: if we import os.path like `from
    os import path`, then please avoid name conflicts!
    """
    if indicator is None:
        nb = len([x for x in os.listdir(head) if '.png' in x])
        new_path = join(head, 'img_{}.png'.format(str(nb).zfill(4)))
    else:
        nb = len([x for x in os.listdir(head) if indicator in x])
        new_path = join(head, '{}_{}.png'.format(indicator, str(nb).zfill(4)))
    if debug:
        print('saving to: {}'.format(new_path))
    cv2.imwrite(new_path, img)


def get_date():
    """Make save path for whatever agent we are training.
    """
    date = '{}'.format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )
    return date


def call_wait_key(nothing=None, force_exit=False):
    """Call this like: `utils.call_wait_key( cv2.imshow(...) )`."""
    key = cv2.waitKey(0)
    if key in ESC_KEYS:
        print("Pressed ESC key. Terminating program...")
        if force_exit:
            sys.exit()
        else:
            return True
    return False


def crop_then_resize(img, ix, iy, offset, final_x, final_y, skip_crop=False):
    """Crop and THEN resize the image for the neural network.

    Experiment with processing before or after. I think processing, then
    filtering, may result in blurrier images than if we did filtering, then
    processing. But I think we did the latter for the ICRA 2020 submission.

    First component 'height', second component 'width'.  Decrease 'height'
    values to get images higher up, decrease 'width' to make it move left.

    We skip because sometimes we may already have a correct-sized image.
    """
    if not skip_crop:
        img = img[ix:ix+offset, iy:iy+offset]
    assert img.shape[0] == img.shape[1]
    img = cv2.resize(img, (final_x, final_y))
    return img


def depth_to_3ch(d_img, cutoff_min, cutoff_max):
    """Process depth images like in the ISRR 2019 paper, second step."""
    w,h = d_img.shape
    n_img = np.zeros([w, h, 3])
    d_img = d_img.flatten()

    # Instead of this:
    #d_img[d_img>cutoff] = 0.0
    # Do this? The cutoff_max means beyond the cutoff, pixels become white.
    #d_img[ d_img>cutoff_max ] = 0.0
    d_img[ d_img>cutoff_max ] = cutoff_max
    d_img[ d_img<cutoff_min ] = cutoff_min
    print('max/min depth after cutoff: {:.3f} {:.3f}'.format(np.max(d_img), np.min(d_img)))

    d_img = d_img.reshape([w,h])
    for i in range(3):
        n_img[:, :, i] = d_img
    return n_img


def depth_3ch_to_255(d_img):
    """Process depth images like in the ISRR 2019 paper, second step."""
    # Instead of this:
    #d_img = 255.0/np.max(d_img)*d_img
    # Do this:
    d_img = d_img * (255.0 / (np.max(d_img)-np.min(d_img)) )  # pixels within a 255-interval
    d_img = d_img - np.min(d_img)                             # pixels actually in [0,255]

    d_img = np.array(d_img, dtype=np.uint8)
    for i in range(3):
        d_img[:, :, i] = cv2.equalizeHist(d_img[:, :, i])
    return d_img


def inpaint_depth_image(d_img, ix, iy, offset):
    """Inpaint depth image on raw depth values.

    Only import code here to avoid making them required if we're not inpainting.

    (1) Applying inpainting on a cropped area of the real depth image, to avoid
    noise from area outside the fabric. DO NOT RESIZE THE DEPTH IMAGE (via
    cv2.imresize) THEN INPAINT. That changes the resolution of the depth image.

    (2) We also need to ensure we are consistent in cropping for later code. I
    think it makes sense to use the same cropping code.

    (3) The window size is 3 which I think means we can get away with a pixel
    difference of 3 when cropping but to be safe let's add a bit more, 50 pix
    to each side.
    """
    d_img = d_img[ix:ix+offset, iy:iy+offset]
    from perception import (ColorImage, DepthImage)
    print('now in-painting the depth image (shape {}), ix, iy = {}, {}...'.format(
            d_img.shape, ix, iy))
    start_t = time.time()
    d_img = DepthImage(d_img)
    d_img = d_img.inpaint()     # inpaint, then get d_img right away
    d_img = d_img.data          # get raw data back from the class
    cum_t = time.time() - start_t
    print('finished in-painting in {:.2f} seconds, result is {}-sized img'.format(
            cum_t, d_img.shape))
    return d_img


def calculate_coverage(c_img, bounding_dims=(10,91,10,91), rgb_cutoff=90, display=False):
    """
    Given precomputed constant preset locations that represent the corners in a
    clockwise order, it computes the percent of pixels that are above a certain
    threshold in that region which represents the percent coverage.

    The bounding dimensions represent (min_x, max_x, min_y, max_y). The default
    bounding_dims work well empirically but are dependent on camera and foam
    rubber position!! COLOR IMAGES ONLY!! Example: if we are looking at images,
    and see stuff at the *bottom* that we need to include then increase max_x,
    if we want to remove stuff from the bottom, decrease max_x. To tune that, I
    just run `python utils.py` with an example set of images.

    Returns a coverage value between [0,1].

    NOTE: this should be called with the system python.
    """
    min_x, max_x, min_y, max_y = bounding_dims
    substrate = c_img[min_x:max_x,min_y:max_y,:]
    is_not_covered = np.logical_and(np.logical_and(substrate[:,:,0] > rgb_cutoff,
        substrate[:,:,1] > rgb_cutoff), substrate[:,:,2] > rgb_cutoff)
    fake_image = np.array(is_not_covered * 255, dtype = np.uint8)

    # Display a bunch of images for debugging
    display_img = Image.new(mode='L', size=(300,300), color=200)
    draw = ImageDraw.Draw(display_img)
    display_img.paste(PIL.Image.fromarray(c_img), (0, 0))
    display_img.paste(PIL.Image.fromarray(substrate), (100+min_x, min_y))
    display_img.paste(PIL.Image.fromarray(substrate), (min_x, 100+min_y))
    display_img.paste(PIL.Image.fromarray(fake_image), (200+min_x, min_y))

    coverage = 1.0 - (np.sum(is_not_covered) / float(is_not_covered.size))

    if display:
        cv2.imshow("coverage: {:.3f}".format(coverage), np.array(display_img) )
        #cv2.imshow("1", substrate)
        #cv2.imshow("2", fake_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return coverage


def test_action_mapping(c_img, bounding_dims=(10,90,10,90), rgb_cutoff=90, display=False,
        act=None):
    """
    Similar to coverage code, but now test if we can map an action to the closest
    cloth point. Useful if we are 'just missing' the fabric with learned policies.

    Can do stuff like this to annotate an image if it helps:
        cv2.circle(img, center=pix_pick, radius=5, color=cfg.BLUE, thickness=1)
        cv2.circle(img, center=pix_targ, radius=3, color=cfg.RED, thickness=1)

    For contours, help(cv2.findContours()) told me:
        findContours(image, mode, method[, contours[, hierarchy[, offset]]]) ->
            image, contours, hierarchy
    so it returns three values. See also:
    https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/#comment-378667

    For general comments on how the procedure works, see the action correction
    method in `run.py`.
    """
    # Convert from (-1,1) to the image pixels. Edit: well if we annotate these in opencv,
    # we have to invert the y, so basically we do 100-y for the image annotations. But
    # for numpy we still use the standard y.
    print('\nDebugging the \'re-mapping\', act: {}'.format(act))
    assert c_img.shape == (100,100,3)
    c_img_orig = c_img.copy()

    min_x, max_x, min_y, max_y = bounding_dims
    c_img = c_img[min_x:max_x,min_y:max_y,:]
    print('resized c_img: {}'.format(c_img.shape))

    # Convert from action space to pixels (with some caveats). Ah, with B we may
    # get a B-sized list but with index at B, so do -1?
    B = c_img.shape[0] - 1
    XX = B / 2
    pix_pick = (act[0] * XX + XX,
                act[1] * XX + XX)
    pix_targ = ((act[0]+act[2]) * XX + XX,
                (act[1]+act[3]) * XX + XX)

    min_x, max_x, min_y, max_y = bounding_dims
    #substrate = c_img[min_x:max_x,min_y:max_y,:]
    substrate = c_img.copy()
    is_not_covered = np.logical_and(np.logical_and(substrate[:,:,0] > rgb_cutoff,
        substrate[:,:,1] > rgb_cutoff), substrate[:,:,2] > rgb_cutoff)
    fake_image = np.array(is_not_covered * 255, dtype = np.uint8)

    imgray = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
    # Hmm ... I'm seeing 100 as a better threshold than the rgb_cutoff=90?
    # thresh is a grayscale image. Should be 255 for white values, right?
    # Since the cloth is BLACK we want to check if thresh[pick_point] is 0.
    _threshold = 100
    ret, thresh = cv2.threshold(imgray, _threshold, 255, cv2.THRESH_BINARY)
    tot_w = np.sum(thresh >= 255.0)
    tot_b = np.sum(thresh <= 0.0)
    print('thresh: size {}'.format(thresh.shape))
    print('  white, black pixels: {}, {},  sum {}'.format(tot_w, tot_b, tot_w+tot_b))

    # Find out if pick point is on the cloth or not, `thresh` is a numpy array.
    # Ah, it can be inaccurate due to boundary conditions? Not sure how to easily fix.
    x, y = int(pix_pick[0]), int(pix_pick[1])
    if (thresh[B-y, x] <= 0.0):
        print('ON the cloth, arr[{},{}], thresh: {}'.format(B-y,x,_threshold))
        print('  (we do not need to do any re-mapping)')
        ON_CLOTH = True
    else:
        print('NOT on the cloth, arr[{},{}], thresh: {}'.format(B-y,x,_threshold))
        print('  (should map action towards the cloth, ideally)')
        ON_CLOTH = False

    # some more threshold stuff, find the center pixel?
    # I know it has some of the boundary stuff, but part of that is unavoidable imo.
    cloth_indices = np.argwhere(thresh < 0.1)
    #print('thresh.shape: {}, tot {}'.format(thresh.shape, np.prod(thresh.shape)))
    #print('cloth_indices: {}'.format(cloth_indices.shape))
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

    # -------------------------------------------------------------------------------
    # PICK POINT THAT IS RE-MAPPED. Get it in [-1,1] then convert to pixels.
    # The old pick point was at (act[0], act[1]). Also, in the actual code, if
    # we called this many times, just multiply change_{x,y} by freq.
    # -------------------------------------------------------------------------------
    # UPDATE: actually no we should just see if we can keep going until we hit
    # a non-cloth point, and if we are on the cloth, we should not be doing anything.
    # -------------------------------------------------------------------------------

    # This is how much we change each time.
    CHANGE_CONST = 10
    delta_x = dir_norm[0] / CHANGE_CONST
    delta_y = dir_norm[1] / CHANGE_CONST
    change_x = delta_x
    change_y = delta_y
    if ON_CLOTH:
        change_x = 0
        change_y = 0
    else:
        # I would say we just keep going until we hit the threshold?
        while not ON_CLOTH:
            # Get action representation `new_pick` then expand it to the image size, giving
            # us pixels we can draw assuming we do B-y for the y value.
            new_pick = (act[0]+change_x, act[1]+change_y)
            pix_new = (int(new_pick[0]*XX + XX),
                       int(new_pick[1]*XX + XX))
            x, y = pix_new[0], pix_new[1]
            if not (0 <= x < B) or not (0 <= y < B):
                ON_CLOTH = True
                continue
            if (thresh[B-y, x] <= 0.0):
                print('ON the cloth, arr[{},{}], thresh: {}'.format(B-y,x,_threshold))
                ON_CLOTH = True
            else:
                print('NOT on the cloth, arr[{},{}], thresh: {}'.format(B-y,x,_threshold))
                ON_CLOTH = False
                # keep increasing change_x and change_y
                change_x += delta_x
                change_y += delta_y

    new_pick = (act[0]+change_x, act[1]+change_y)
    pix_new = (int(new_pick[0]*XX + XX),
               int(B - (new_pick[1]*XX + XX)))
    print('      change actx space: {:.2f}'.format(change_x))
    print('      change acty space: {:.2f}'.format(change_x))
    print('      old pick pt: {}'.format((act[0],act[1])))
    print('      new pick pt: {} NEW PICK POINT IN BLUE'.format(new_pick))
    c_img = cv2.circle(c_img, center=pix_new, radius=4, color=cfg.BLUE, thickness=-1)

    # Display a bunch of images for debugging
    # These go from right to left, generally.
    display_img = Image.new(mode='RGB', size=(500,200), color=200)
    draw = ImageDraw.Draw(display_img)
    display_img.paste(PIL.Image.fromarray(c_img),      (  0, 0)) # original image w/annotations
    display_img.paste(PIL.Image.fromarray(thresh),     (100, 0)) # tried detecting cloth
    display_img.paste(PIL.Image.fromarray(fake_image), (200, 0)) # for coverage detection
    display_img.paste(PIL.Image.fromarray(c_img_orig), (  0, 100)) # if resizing, original

    coverage = 1.0 - (np.sum(is_not_covered) / float(is_not_covered.size))

    if display:
        cv2.imshow("coverage: {:.3f}".format(coverage), np.array(display_img) )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return coverage


def load_mapping_table(row_board, column_board, file_name, cloth_height=0.005):
    """Load the mapping table which we need to map from neural net to action.

    The mapping table looks like this:

        nx,ny,rx,ry,rz

    Where `nx,ny` are coordinates w.r.t. the background plane, of which the
    cloth lies on top. Numbers range from (-1,1) and should be discretized in
    the mapping table. The `rx,ry,rz` are the x,y,z positions w.r.t. the robot's
    frame, and were derived by moving the robot gripper to that position over a
    checkerboard. Note that rotation is not included in the mapping table.

    :param row_board: number of rows.
    :param column_board: number of columns.
    :param file_name: name of the calibration file
    :param cloth_height: height offset, we add to the z values from the data.
    :return: data from calibration
    """
    assert os.path.exists(file_name), \
            'The file does not exist: {}'.format(file_name)
    data_default = np.loadtxt(file_name, delimiter=',')

    cnt = 0
    for i in range(row_board):
        for j in range(column_board):
            data_default[cnt, 0] = -1 + j * 0.4
            data_default[cnt, 1] = -1 + i * 0.4
            data_default[cnt, 4] = data_default[cnt, 4] + cloth_height
            cnt += 1
    data = data_default

    # Daniel: a bit confused about this, but it seems necessary to convert to
    # PSM space. See `transform_CB2PSM`.
    data_square = np.zeros((row_board + 1, column_board + 1, 5))
    for i in range(row_board):
        for j in range(column_board):
            data_square[i, j, :] = data[column_board * j + i, 0:5]

    for i in range(row_board):
        data_square[i, column_board, :] = data_square[i, column_board - 1, :]
    for j in range(column_board):
        data_square[row_board, j] = data_square[row_board - 1, j]

    return data_square


def transform_CB2PSM(x, y, row_board, col_board, data_square):
    """Minho's code, for calibation, figure out the PSM coordinates.

    Parameters (x,y) should be in [-1,1] (if not we clip it) and represent
    the coordinate range over the WHITE CLOTH BACKGROUND PLANE (or a
    'checkboard' plane). We then convert to a PSM coordinate.

    Uses bilinear interpolation.

    :param row_board: number of rows.
    :param col_board: number of columns.
    :param data_square: data from calibration.
    """
    if x>1: x=1.0
    if x<-1: x=-1.0
    if y>1:  y=1.0
    if y<-1: y=-1.0

    for i in range(row_board):
        for j in range(col_board):
            if x == data_square[row_board-1, j, 0] and y == data_square[i, col_board-1, 1]: # corner point (x=1,y=1)
                return data_square[row_board-1,col_board-1,2:5]
            else:
                if x == data_square[row_board-1, j, 0]:  # border line of x-axis
                    if data_square[i, j, 1] <= y and y < data_square[i, j + 1, 1]:
                        y1 = data_square[row_board-1, j, 1]
                        y2 = data_square[row_board-1, j+1, 1]
                        Q11 = data_square[row_board-1, j, 2:5]
                        Q12 = data_square[row_board-1, j+1, 2:5]
                        return (y2-y)/(y2-y1)*Q11 + (y-y1)/(y2-y1)*Q12
                elif y == data_square[i, col_board-1, 1]:  # border line of y-axis
                    if data_square[i, j, 0] <= x and x < data_square[i + 1, j, 0]:
                        x1 = data_square[i, col_board-1, 0]
                        x2 = data_square[i+1, col_board-1, 0]
                        Q11 = data_square[i, col_board-1, 2:5]
                        Q21 = data_square[i+1, col_board-1, 2:5]
                        return (x2-x)/(x2-x1)*Q11 + (x-x1)/(x2-x1)*Q21
                else:
                    if data_square[i,j,0] <= x and x < data_square[i+1,j,0]:
                        if data_square[i,j,1] <= y and y < data_square[i,j+1,1]:
                            x1 = data_square[i, j, 0]
                            x2 = data_square[i+1, j, 0]
                            y1 = data_square[i, j, 1]
                            y2 = data_square[i, j+1, 1]
                            Q11 = data_square[i, j, 2:5]
                            Q12 = data_square[i, j+1, 2:5]
                            Q21 = data_square[i+1, j, 2:5]
                            Q22 = data_square[i+1, j+1, 2:5]
                            if x1==x2 or y1==y2:
                                return []
                            else:
                                return 1/(x2-x1)/(y2-y1)*(Q11*(x2-x)*(y2-y) + Q21*(x-x1)*(y2-y) + Q12*(x2-x)*(y-y1) + Q22*(x-x1)*(y-y1))


def move_p_from_net_output(x, y, dx, dy, row_board, col_board, data_square, p,
                           debug=False, only_do_pick=False):
    """Minho's code, for calibration, processes policy network output.

    Be careful, the x,y coordinate from the neural net refers to a coordinate
    range of [-1,1] in the x and y directions. Thus, (x,y) = (-1,-1) is the
    BOTTOM LEFT CORNER.

    However, in simulation, we first converted (x,y) into the range [0,1] by
    dividing by two (to get values in [-0.5,0.5]) and then adding 0.5. Then,
    given x and y values that ranged from [0,1], we deduced a dx and dy such
    that ... when we apply the action, dx and dy independently 'adjust' x and y.
    So it is indeed (x+dx) and (y+dy). To convert this to our case, it should be
    as simple as doubling the dx and dy values.

    It's a bit trick to understand by reading gym-cloth code, because I first
    convert dx and dy into other values, and then I repeatdly do motions until
    the full length is achieved.

    :params (x, y, dx, dy): outputs from the neural network, all in [-1,1].
    :param row_board: number of rows.
    :param col_board: number of columns.
    :param data_square: data from calibration, from `utils.load_mapping_table`.
    :param p: An instance of `dvrkClothSim`.
    """
    assert -1 <= x <= 1, x
    assert -1 <= y <= 1, y
    assert -1 <= dx <= 1, dx
    assert -1 <= dy <= 1, dy

    # Find the targets, and then get pose w.r.t. PSM.
    targ_x = x + 2*dx
    targ_y = y + 2*dy
    pickup_pos = transform_CB2PSM(x,
                                  y,
                                  row_board,
                                  col_board,
                                  data_square)
    release_pos_temp = transform_CB2PSM(targ_x,
                                        targ_y,
                                        row_board,
                                        col_board,
                                        data_square)

    release_pos = np.array([release_pos_temp[0], release_pos_temp[1]])
    if debug:
        print('pickup position wrt PSM: {}'.format(pickup_pos))
        print('release position wrt PSM: {}'.format(release_pos))
    # just checking if the ROS input is fine
    # user_input = raw_input("Are you sure the values to input to the robot arm?(y or n)")
    # if user_input == "y":

    p.move_pose_pickup(pickup_pos, release_pos, 0, 'rad', only_do_pick=only_do_pick)


def print_means(images):
    average_img = np.zeros((100,100,3))
    for img in images:
        average_img += img
    a_img = average_img / len(images)
    print('ch 1: {:.1f} +/- {:.1f}'.format(np.mean(a_img[:,:,0]), np.std(a_img[:,:,0])))
    print('ch 2: {:.1f} +/- {:.1f}'.format(np.mean(a_img[:,:,1]), np.std(a_img[:,:,1])))
    print('ch 3: {:.1f} +/- {:.1f}'.format(np.mean(a_img[:,:,2]), np.std(a_img[:,:,2])))
    return a_img


def single_means(img, depth):
    if depth:
        print('Depth img:')
    else:
        print('Color img:')
    print('  ch 1: {:.1f} +/- {:.1f}'.format(np.mean(img[:,:,0]), np.std(img[:,:,0])))
    print('  ch 2: {:.1f} +/- {:.1f}'.format(np.mean(img[:,:,1]), np.std(img[:,:,1])))
    print('  ch 3: {:.1f} +/- {:.1f}'.format(np.mean(img[:,:,2]), np.std(img[:,:,2])))


def _adjust_gamma(image, gamma=1.0):
    """For darkening images.

    Builds a lookup table mapping the pixel values [0, 255] to their
    adjusted gamma values.

    https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 \
            for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


if __name__ == "__main__":
    """
    This should not be called normally! Just for testing and debugging.
    For example, use this to test the coverage code.  For this we should use
    the same virtualenv as we do for the camera script. Edit: no, getting
    problems with opencv, just do system python. Pray there are no substantive
    python 2 vs 3 changes with opencv code, if that issue arises ...
    """
    # Directory of images. Just run zivid camera script and copy images into 'tmp'.
    img_paths = sorted(
            [join('tmp',x) for x in os.listdir('tmp/') if 'c_img_crop_proc' in x and
                    '.png' in x and '_56' not in x]
    )
    images = [cv2.imread(x) for x in img_paths]
    img_paths_d = sorted(
            [join('tmp',x) for x in os.listdir('tmp/') if 'd_img_crop_proc' in x and
                    '.png' in x and '_56' not in x]
    )
    images_d = [cv2.imread(x) for x in img_paths_d]

    # Compute means, compare with simulated data.
    print('depth across all data in directory:')
    _ = print_means(images_d)
    print('color across all data in directory:')
    _ = print_means(images)

    nb_imgs = len(img_paths)
    print('num images: {}'.format(nb_imgs))

    # Test coverage.
    if False:
        for idx,(img,fname) in enumerate(zip(images,img_paths)):
            coverage = calculate_coverage(img, display=True)
            print('  image {} at {} has coverage {:.2f}'.format(idx, fname, coverage*100))

    # Test action mapping. Just with `python utils.py`. Only pick point matters.
    if True:
        #act = [-0.90, -0.90, 0.5, 0.5] # passes
        #act = [0.90, -0.90, 0.5, 0.5] # passes
        #act = [-0.90, 0.90, 0.5, 0.5] # passes
        act = [0.90, 0.90, 0.5, 0.5] # passes
        #act = [-0.0, -0.0, 0.5, 0.5] # passes
        for idx,(img,fname) in enumerate(zip(images,img_paths)):
            coverage = test_action_mapping(img, display=True, act=act)
            print('  image {} at {} has coverage {:.2f}'.format(idx, fname, coverage*100))

    # --------------------------------------------------------------------------
    # Daniel: I was using the following for debugging if we wanted to forcibly
    # adjust pixel values to get them in line with the training data.
    # Fortunately it seems close enough that we don't have to change anything.
    # --------------------------------------------------------------------------

    # Save any modified images. DEPTH here. Works well.
    if False:
        for idx,(img,fname) in enumerate(zip(images_d,img_paths_d)):
            print('  on image {}'.format(fname))
            single_means(img, depth=True)

            meanval = np.mean(img)
            target_meanval = 135.0
            img = np.minimum(np.maximum( img+(target_meanval-meanval), 0), 255)
            img = np.uint8(img)
            print('after correcting:')
            single_means(img, depth=True)

            savepath = fname.replace('tmp','tmp2')
            print('saving to: {}\n'.format(savepath))
            cv2.imwrite(savepath, img)

    # Save any modified images. RGB here. NOTE: doesn't work quite well
    if False:
        for idx,(img,fname) in enumerate(zip(images,img_paths)):
            print('  on image {}'.format(fname))
            single_means(img, depth=False)

            mean_0 = np.mean(img[:,:,0])
            mean_1 = np.mean(img[:,:,1])
            mean_2 = np.mean(img[:,:,2])

            means = np.array([mean_0, mean_1, mean_2])
            targets = np.array([155.0, 110.0, 85.0])
            img = np.minimum(np.maximum( img+(targets-means), 0), 255)
            img = np.uint8(img)
            print('after correcting:')
            single_means(img, depth=False)

            savepath = fname.replace('tmp','tmp2')
            print('saving to: {}\n'.format(savepath))
            cv2.imwrite(savepath, img)

    # Now for RGB gamma corrections.
    if False:
        for idx,(img,fname) in enumerate(zip(images,img_paths)):
            print('  on image {}'.format(fname))
            single_means(img, depth=False)

            img = _adjust_gamma(img, gamma=1.5)
            print('after correcting:')
            single_means(img, depth=False)

            savepath = fname.replace('tmp','tmp2')
            print('saving to: {}\n'.format(savepath))
            cv2.imwrite(savepath, img)

    # Try de-noising.
    if False:
        print('\n\nTRYING DE-NOISING\n')
        # Depth images are type uint8.
        for idx,(img,fname) in enumerate(zip(images_d,img_paths_d)):
            print('  on image {}'.format(fname))
            img = cv2.fastNlMeansDenoising(img,None,7,7,21)
            savepath = fname.replace('tmp','tmp2')
            print('saving to: {}\n'.format(savepath))
            cv2.imwrite(savepath, img)
        for idx,(img,fname) in enumerate(zip(images,img_paths)):
            print('  on image {}'.format(fname))
            img = cv2.fastNlMeansDenoisingColored(img,None,7,7,7,21)
            savepath = fname.replace('tmp','tmp2')
            print('saving to: {}\n'.format(savepath))
            cv2.imwrite(savepath, img)
