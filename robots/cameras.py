import threading
from abc import abstractmethod, abstractproperty
from typing import Dict, Optional, Union

import numpy as np

try:
    import cv2

    IMPORTED_CV2 = True
except ImportError:
    IMPORTED_CV2 = False

try:
    import pyrealsense2 as rs

    IMPORTED_PYREALSENSE = True
except ImportError:
    IMPORTED_PYREALSENSE = False


class Camera(object):
    def __init__(self, width: int, height: int, depth: bool = False):
        self.width = width
        self.height = height
        self.depth = depth

    @abstractproperty
    def has_depth(self):
        raise NotImplementedError

    @abstractmethod
    def get_frames(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def close(self):
        pass


class OpenCVCamera(Camera):
    def __init__(self, id: Optional[Union[int, str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self._cap = None

    @property
    def cap(self):
        assert IMPORTED_CV2, "cv2 not imported."
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.id)  # type: ignore
            # Values other than default 640x480 have not been tested yet
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self._cap

    def get_frames(self):
        _, image = self.cap.read()
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {"image": image}

    def close(self):
        self.cap.release()


class ThreadedOpenCVCamera(Camera):
    def __init__(self, id: Optional[Union[int, str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self._running = False

    @property
    def has_depth(self):
        return False

    def _init(self):
        assert IMPORTED_CV2, "cv2 not imported."
        self._running = True
        self._image = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        # We need to parse the CV Camera ID
        self._cap = cv2.VideoCapture(self.id)
        # Values other than default 640x480 have not been tested yet
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.height)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.width)

        while self._running:
            retval, img = self._cap.read()
            if retval:
                self.lock.acquire()
                self._image = img
                self.lock.release()

        self._cap.release()

    def get_frames(self) -> np.ndarray:
        if not self._running:
            self._init()
        image = None
        while image is None:
            self.lock.acquire()
            if self._image is not None:
                image = self._image
                self._image = None  # set back to None.
            self.lock.release()
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return dict(image=image)

    def close(self):
        if self._running:
            self._running = False
            self.thread.join()


class RealSenseCamera(Camera):
    def __init__(self, serial_number, **kwargs):
        super().__init__(**kwargs)
        self.serial_number = str(serial_number)
        self._pipeline = None
        self.align = None
        self.depth_filters = None

    @property
    def has_depth(self):
        return self.depth

    @property
    def pipeline(self):
        assert IMPORTED_PYREALSENSE, "pyrealsense2 not installed."
        if self._pipeline is None:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial_number)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            if self.depth:
                config.enable_stream(rs.stream.color, 640, 480, rs.format.z16, 30)
                self.depth_filters = [rs.spatial_filter(), rs.temporal_filter()]
            self._pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            # warmup cameras
            for _ in range(2):
                self._pipeline.wait_for_frames()
        return self._pipeline

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        image = np.asanyarray(aligned_frames.get_color_frame().get_data())
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frames = dict(image=image)
        if self.depth:
            depth = aligned_frames.get_depth_frame()
            for rs_filter in self.depth_filters:
                depth = rs_filter(depth)
            depth = np.asanyarray(depth.get_data())
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=-1)
            frames["depth"] = depth
        return frames

    def close(self):
        self.pipeline.stop()


class ThreadedRealSenseCamera(Camera):
    def __init__(self, serial_number, **kwargs):
        super().__init__(**kwargs)
        self.serial_number = str(serial_number)

    @property
    def has_depth(self):
        return self.depth

    def _init(self):
        assert IMPORTED_PYREALSENSE, "pyrealsense2 not installed."
        self._running = True
        self._image = None
        self._depth = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        if self.depth:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.z16, 30)
            depth_filters = [rs.spatial_filter(), rs.temporal_filter()]
            align = rs.align(rs.stream.color)
        pipeline.start(config)
        while self._running:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            if self.depth:
                depth = aligned_frames.get_depth_frame()
                for rs_filter in depth_filters:
                    depth = rs_filter(depth)
                depth = np.asanyarray(depth.get_data())
            else:
                self._depth = None
            image = np.asanyarray(aligned_frames.get_color_frame().get_data())

            self.lock.acquire()
            if self.depth:
                self._depth = depth
            self._image = image
            self.lock.release()

        # Close the thread
        pipeline.stop()

    def get_frames(self) -> np.ndarray:
        if not self._running:
            self._init()
        image = None
        while image is None:
            self.lock.acquire()
            if self._image is not None:
                image, depth = self._image, self._depth
                self._image, self._depth = None, None  # set back to None.
            self.lock.release()

        # Process the image
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frames = dict(image=image)
        if self.depth:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=-1)
            frames["depth"] = depth
        return frames

    def close(self):
        if self._running:
            self._running = False
            self.thread.join()
