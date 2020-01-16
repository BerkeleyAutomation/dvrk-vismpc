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
            return self.image, self.depth


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

    # For real physical robot experiments, use these values in `config.py`.
    CUTOFF_MIN = 0.800
    CUTOFF_MAX = 0.905
    IN_PAINT = True

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

        # Images are 1200 x 1920, with 3 channels (well, we force for depth).
        assert d_img.shape == (1200, 1920, 1), d_img.shape
        assert c_img.shape == (1200, 1920, 3), c_img.shape

        print(c_img.shape)
        print(d_img.shape)
        cv2.imwrite(join(HEAD,'c_img.png'), c_img)
        cv2.imwrite(join(HEAD,'d_img.png'), d_img)

        ## # BUT we can call `inpaint` which will fill in the zero pixels!
        ## # UPDATE: check if this works with Python3, but I think it does.
        ## if IN_PAINT:
        ##     d_img = U.inpaint_depth_image(d_img)

        ## # Check depth image. Also, we have to tune the cutoff.
        ## # The depth is clearly in METERS, but I think it's hard to get an
        ## # accurate cutoff, sadly.
        ## print('\nAfter NaN filtering of the depth images ...')
        ## print('  max: {:.3f}'.format(np.max(d_img)))
        ## print('  min: {:.3f}'.format(np.min(d_img)))
        ## print('  mean: {:.3f}'.format(np.mean(d_img)))
        ## print('  medi: {:.3f}'.format(np.median(d_img)))
        ## print('  std: {:.3f}'.format(np.std(d_img)))

        # I think we need a version with and without the cropped for depth.
        d_img_crop = U.process_img_for_net(d_img)
        print('\nAfter NaN filtering of the depth images ... now for the CROPPED image:')
        print('  max: {:.3f}'.format(np.max(d_img_crop)))
        print('  min: {:.3f}'.format(np.min(d_img_crop)))
        print('  mean: {:.3f}'.format(np.mean(d_img_crop)))
        print('  medi: {:.3f}'.format(np.median(d_img_crop)))
        print('  std: {:.3f}'.format(np.std(d_img_crop)))
        print('')



        i += 1


if __name__ == "__main__":
    zc = ZividCapture()
    daniel_test(zc)
