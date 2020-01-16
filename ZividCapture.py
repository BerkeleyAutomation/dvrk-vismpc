import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import datetime
import zivid
import cv2
import numpy as np
import argparse
import math
import os
import sys
from os.path import join
import utils as U

# ---------------------------------------------------------------------------- #
# Camera configuration, keep separate from internal `camera.py` testing.
# Tune cutoff carefully, it's in meters.
# ---------------------------------------------------------------------------- #
# Minho's new camera code returns depth in millimeters.
# ---------------------------------------------------------------------------- #
_CUTOFF_MIN = 780.0
_CUTOFF_MAX = 895.0

# Careful, make sure cropping and inpainting are consistent.
_IN_PAINT = True

# Last tuned: January 16, 2020. These are tuned for a specific camera
# configuration and a location of the fabric plane.
_IX = 365
_IY = 840

# The 'offset' is the size, in ORIGINAL pixels, that the image should span.
# Generally 480-500 works well.  We'd call a resize on this area AFTER we crop.
_OFFSET = 485

# Pretty simple, whatever the size of the actual image input to the neural net.
_FINAL_X = 56
_FINAL_Y = _FINAL_X
# ---------------------------------------------------------------------------- #
# End of camera configuration values.
# ---------------------------------------------------------------------------- #


class ZividCapture():

    def __init__(self):
        self.image = []
        self.depth = []
        app = zivid.Application()
        self.camera = app.connect_camera()
        self.t = 0.0
        self.t_prev = 0.0
        self.interval = 0.0
        self.fps = 0.0

        # 2D image setting
        self.settings_2d = zivid.Settings2D()
        self.settings_2d.iris = 26
        self.settings_2d.exposure_time = datetime.timedelta(microseconds=8333)

        # 3D capture setting
        with self.camera.update_settings() as updater:
            updater.settings.iris = 26
            updater.settings.exposure_time = datetime.timedelta(microseconds=8333)
            updater.settings.filters.reflection.enabled = True

    def measure_fps(self):
        self.t_prev = self.t
        self.t = time.clock()  # sec
        self.interval = self.t - self.t_prev
        self.fps = 1/self.interval
        # print(self.interval, self.fps)

    def capture_2Dimage(self):      # measured as 20~90 fps
        with self.camera.capture_2d(self.settings_2d) as frame_2d:
            np_array = frame_2d.image().to_array()
            # print(np_array.dtype.names)
            self.image = np.asarray([np_array["b"], np_array["g"], np_array["r"]])
            self.image = np.moveaxis(self.image, [0, 1, 2], [2, 0, 1])
            self.image = self.image.astype(np.uint8)
            # self.measure_fps()
            return self.image

    def capture_3Dimage(self):      # measured as 7~10 fps
        with self.camera.capture() as frame:
            np_array = frame.get_point_cloud().to_array()
            # print (np_array.dtype.names)
            self.image = np.asarray([np_array["b"], np_array["g"], np_array["r"]])
            self.image = np.moveaxis(self.image, [0, 1, 2], [2, 0, 1])
            self.image = self.image.astype(np.uint8)
            self.depth = np.asarray([np_array["z"]])    # unit = (mm)
            self.depth = np.moveaxis(self.depth, [0, 1, 2], [2, 0, 1])
            # self.measure_fps()
            return self.depth

    def get_c_d_img(self):
        """Get the rgb and depth we use for the fabric images.

        Careful: as usual, the depth will have lots of NaNs, so post-processing
        is necessary.
        """
        with self.camera.capture() as frame:
            np_array = frame.get_point_cloud().to_array()
            self.image = np.asarray([np_array["b"], np_array["g"], np_array["r"]])
            self.image = np.moveaxis(self.image, [0, 1, 2], [2, 0, 1])
            self.image = self.image.astype(np.uint8)
            self.depth = np.asarray([np_array["z"]])    # unit = (mm)
            self.depth = np.moveaxis(self.depth, [0, 1, 2], [2, 0, 1])
            return self.image, np.squeeze(self.depth)


def continuously_view_2d():
    # From Miho
    while True:
        image2D = zc.capture_2Dimage()
        # image3D = zc.capture_3Dimage()
        cv2.imshow("", image2D)
        cv2.waitKey(1)


def daniel_test(zc):
    """For real experiments, copy relevant code to the run script."""
    HEAD = "/home/davinci/seita/dvrk-vismpc/tmp"
    i = 0
    nb_images = 1

    while i < nb_images:
        print(os.listdir(HEAD))
        num = len([x for x in os.listdir(HEAD) if 'c_img_crop' in x])
        print('current index is at: {}'.format(num))
        d_img = None
        c_img = None
        while c_img is None or d_img is None:
            c_img, d_img = zc.get_c_d_img()

        # Check for NaNs.
        nb_items = np.prod(np.shape(c_img))
        nb_not_nan = np.count_nonzero(~np.isnan(c_img))
        print('RGB image shape {}, has {} items'.format(c_img.shape, nb_items))
        print('  num NOT nan: {}, or {:.2f}%'.format(nb_not_nan, nb_not_nan/float(nb_items)*100))
        nb_items = np.prod(np.shape(d_img))
        nb_not_nan = np.count_nonzero(~np.isnan(d_img))
        print('depth image shape {}, has {} items'.format(d_img.shape, nb_items))
        print('  num NOT nan: {}, or {:.2f}%'.format(nb_not_nan, nb_not_nan/float(nb_items)*100))

        # We fill in NaNs with zeros.
        c_img[np.isnan(c_img)] = 0
        d_img[np.isnan(d_img)] = 0
        print('\nAfter NaN filtering of the depth images:')
        U.debug_print_img(d_img)
        assert d_img.shape == (1200, 1920), d_img.shape
        assert c_img.shape == (1200, 1920, 3), c_img.shape

        # We can `inpaint` which fills in the zero pixels. WILL RESIZE IMAGE.
        if _IN_PAINT:
            d_img = U.inpaint_depth_image(d_img, ix=_IX, iy=_IY, offset=_OFFSET)
            d_img_crop = U.crop_then_resize(d_img, _IX, _IY, _OFFSET, _FINAL_X, _FINAL_Y,
                                            skip_crop=True)
        else:
            d_img_crop = U.crop_then_resize(d_img, _IX, _IY, _OFFSET, _FINAL_X, _FINAL_Y,
                                            skip_crop=False)
        c_img_crop = U.crop_then_resize(c_img, _IX, _IY, _OFFSET, _FINAL_X, _FINAL_Y)
        print('\nAfter NaN filtering of the depth images, now the RESIZED image:')
        U.debug_print_img(d_img_crop)

        # Let's process depth. Note that we do the cropped vs noncropped
        # separately, so the cropped one shouldn't have closer noisy values from
        # the dvrk arm affecting its calculations.
        d_img      = U.depth_to_3ch(d_img,      cutoff_min=_CUTOFF_MIN, cutoff_max=_CUTOFF_MAX)
        d_img_crop = U.depth_to_3ch(d_img_crop, cutoff_min=_CUTOFF_MIN, cutoff_max=_CUTOFF_MAX)
        d_img      = U.depth_3ch_to_255(d_img)
        d_img_crop = U.depth_3ch_to_255(d_img_crop)

        # Try blurring depth, bilateral recommends 9 for offline applications
        # that need heavy blurring. The two sigmas were 75 by default.
        d_img_crop_blur = cv2.bilateralFilter(d_img_crop, 9, 100, 100)
        #d_img_crop_blur = cv2.medianBlur(d_img_crop_blur, 5)

        c_tail           = "{}_c_img.png".format(str(num).zfill(2))
        d_tail           = "{}_d_img.png".format(str(num).zfill(2))
        c_tail_crop      = "{}_c_img_crop.png".format(str(num).zfill(2))
        d_tail_crop      = "{}_d_img_crop.png".format(str(num).zfill(2))
        d_tail_crop_blur = "{}_d_img_crop_blur.png".format(str(num).zfill(2))

        c_img_path           = join(HEAD, c_tail)
        d_img_path           = join(HEAD, d_tail)
        c_img_path_crop      = join(HEAD, c_tail_crop)
        d_img_path_crop      = join(HEAD, d_tail_crop)
        d_img_path_crop_blur = join(HEAD, d_tail_crop_blur)

        cv2.imwrite(c_img_path,           c_img)
        cv2.imwrite(d_img_path,           d_img)
        cv2.imwrite(c_img_path_crop,      c_img_crop)
        cv2.imwrite(d_img_path_crop,      d_img_crop)
        cv2.imwrite(d_img_path_crop_blur, d_img_crop_blur)

        print('\n  just saved: {}'.format(c_img_path))
        print('  just saved: {}'.format(d_img_path))
        print('  just saved: {}'.format(c_img_path_crop))
        print('  just saved: {}'.format(d_img_path_crop))
        print('  just saved: {}'.format(d_img_path_crop_blur))
        i += 1


if __name__ == "__main__":
    zc = ZividCapture()
    daniel_test(zc)
