import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import datetime, time
import zivid
import cv2
import numpy as np

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

if __name__ == "__main__":
    zc = ZividCapture()
    while True:
        image2D = zc.capture_2Dimage()
        # image3D = zc.capture_3Dimage()

        cv2.imshow("", image2D)
        cv2.waitKey(1)