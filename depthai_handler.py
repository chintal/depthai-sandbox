import depthai
import blobconverter
import numpy as np
import cv2
from aruco_detector import ArucoDetector
from scene_reconstruction import SceneReconstruction
from geometries import GeometriesManager
from spatial import SpatialCalculators
from roi import ROICalculators
from pprint import PrettyPrinter as pp
from dataclasses import dataclass


# TODO Change this config into something more manageable.
frame_specs = {
    'rgb': {
        'latest_frame': 'latest_frame_rgb',
        'calibkey': 'rgb',
        'is_grey': False,
        'seq': 2,
        'socket': depthai.CameraBoardSocket.CAM_A,
    },
    'left': {
        'latest_frame': 'latest_frame_l',
        'calibkey': 'left',
        'is_grey': True,
        'seq': 0,
        'socket': depthai.CameraBoardSocket.CAM_B,
    },
    'right': {
        'latest_frame': 'latest_frame_r',
        'calibkey': 'right',
        'is_grey': True,
        'seq': 1,
        'socket': depthai.CameraBoardSocket.CAM_C
    },
    'depth': {
        'latest_frame': 'latest_frame_depth',
        'is_grey': True,
        'seq': 4,
    },
    'disparity': {
        'latest_frame': 'latest_frame_disparity',
        'is_grey': False,
        'seq': 5,
    }
}


@dataclass
class FrameSet:
    left: any = None
    right: any = None
    rgb: any = None
    rgb_preview: any = None
    depth: any = None
    disparity: any = None
    nn: any = None
    aruco: any = None
    aruco_spatials: any = None
    pointcloud: any = None


class DepthAIHandler:
    def __init__(self, enable_nn=False, enable_aruco=True, 
                 enable_reconstruction=True, enable_depth=True, enable_pointcloud=True, 
                 rgb_nn_size=(300, 300), aruco_input='rgb', draw_aruco_on=None,
                 fps_mono=None, fps_rgb=None, 
                 depth_extended_disparity=False, depth_subpixel_disparity=False,
                 depth_align='left', depth_thresholds=None, depth_spatial_filter=False):
        self.enable_nn = enable_nn
        self.enable_aruco = enable_aruco
        self.enable_reconstruction = enable_reconstruction
        self.enable_depth = enable_depth
        self.enable_pointcloud = enable_pointcloud
        self.rgb_nn_size = rgb_nn_size
        self.aruco_input = aruco_input
        self.draw_aruco_on = draw_aruco_on or []
        self.fps_mono = fps_mono
        self.fps_rgb = fps_rgb
        self.depth_extended_disparity = depth_extended_disparity
        self.depth_subpixel_disparity = depth_subpixel_disparity
        self.depth_align = depth_align
        self.depth_thresholds = depth_thresholds or None
        self.depth_spatial_filter = depth_spatial_filter or False

        self.intrinsics = {}
        self.dist_coeffs = {}
        self.extrinsics = {}
        self.fov = {}

        self.pipeline = depthai.Pipeline()
        self.setupPipeline()
        self.startDevice()

        self.initGeometries()

        if self.enable_aruco:
            self.startAruco()
        
        if self.enable_reconstruction:
            self.startResconstruction()
            self.roi = ROICalculators()
            self.spatial = SpatialCalculators(self.geom, self.depth_thresholds)
    
    def initGeometries(self):
        self.geom = GeometriesManager()
        self.geom.frame_sizes = {
            'left': (640, 400),
            'right': (640, 400),
            'depth': (640, 400),
            'disparity': (640, 400),
            'rgb': (1920, 1080)
        }
        self.geom.intrinsics = self.intrinsics
        self.geom.dist_coeffs = self.dist_coeffs
        self.geom.extrinsics = self.extrinsics
        self.geom.fov = self.fov
        self.geom.print()

    def getRGBCamera(self, get_camera_only=False, preview_size=None):
        cam = self.pipeline.create(depthai.node.ColorCamera)
        cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        if preview_size:
            cam.setPreviewSize(*preview_size)
        cam.setInterleaved(False)
        
        if self.fps_rgb:
            cam.setFps(self.fps_rgb)

        if get_camera_only:
            return cam
        
        xout = self.pipeline.createXLinkOut()
        xout.setStreamName('rgb')
        cam.isp.link(xout.input)
        
        if preview_size:
            xout_preview = self.pipeline.createXLinkOut()
            xout_preview.setStreamName('rgb_preview')
            cam.preview.link(xout_preview.input)
            return cam, xout, xout_preview

        return cam, xout
    
    def getMonoCamera(self, side, get_camera_only=False):
        mono = self.pipeline.createMonoCamera()
        mono.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)

        if side not in ['left', 'right']: 
            raise ValueError(f"Expecting left or right. Got {side}")
        mono.setBoardSocket(frame_specs[side]['socket'])

        if self.fps_mono:
            mono.setFps(self.fps_mono)

        if get_camera_only:
            return mono
    
        xout = self.pipeline.createXLinkOut()
        xout.setStreamName(side)
        mono.out.link(xout.input)
        return mono, xout

    def getDepthCamera(self, get_camera_only=False):
        if not hasattr(self, 'cam_left') or not self.cam_left:
            self.cam_left = self.getMonoCamera('left', get_camera_only=True)
        if not hasattr(self, 'cam_right') or not self.cam_right:
            self.cam_right = self.getMonoCamera('right', get_camera_only=True)

        stereo = self.pipeline.createStereoDepth()
        stereo.setLeftRightCheck(True)
        stereo.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setExtendedDisparity(self.depth_extended_disparity)
        stereo.setSubpixel(self.depth_subpixel_disparity)
        config = stereo.initialConfig.get()
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 50
        config.postProcessing.spatialFilter.enable = self.depth_spatial_filter
        config.postProcessing.spatialFilter.alpha = 0.9
        config.postProcessing.spatialFilter.delta = 5
        config.postProcessing.spatialFilter.holeFillingRadius = 10
        config.postProcessing.spatialFilter.numIterations = 1
        if self.depth_thresholds:
            config.postProcessing.thresholdFilter.minRange = self.depth_thresholds[0]
            config.postProcessing.thresholdFilter.maxRange = self.depth_thresholds[1]
        self.cam_left.out.link(stereo.left)
        self.cam_right.out.link(stereo.right)
        
        if self.depth_align:
            stereo.setDepthAlign(frame_specs[self.depth_align]['socket'])

        if get_camera_only:
            return stereo
        
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)
        
        xoutDisparity = self.pipeline.createXLinkOut()
        xoutDisparity.setStreamName("disparity")
        stereo.disparity.link(xoutDisparity.input)

        xoutRectifiedLeft = self.pipeline.createXLinkOut()
        xoutRectifiedLeft.setStreamName("rectifiedLeft")
        stereo.rectifiedLeft.link(xoutRectifiedLeft.input)

        xoutRectifiedRight = self.pipeline.createXLinkOut()
        xoutRectifiedRight.setStreamName("rectifiedRight")
        stereo.rectifiedRight.link(xoutRectifiedRight.input)

        return stereo, xoutDepth, xoutDisparity, xoutRectifiedLeft, xoutRectifiedRight

    def setupPipeline(self):
        self.latest_frame_l = None
        self.latest_frame_r = None
        self.latest_frame_rgb = None
        self.latest_frame_rgb_preview = None

        if not self.enable_depth:
            self.cam_left, self.xout_left = self.getMonoCamera('left')
            self.cam_right, self.xout_right = self.getMonoCamera('right')
        else:
            self.latest_frame_depth = None
            self.latest_frame_disparity = None
            self.cam_depth, self.xout_depth, self.xout_disparity, self.xout_left, self.xout_right = self.getDepthCamera()

            if self.enable_pointcloud:
                self.latest_pointcloud = None
                self.pointcloud = self.pipeline.create(depthai.node.PointCloud)
                self.cam_depth.depth.link(self.pointcloud.inputDepth)
                xoutPointCloud = self.pipeline.createXLinkOut()
                xoutPointCloud.setStreamName("pointcloud")
                self.pointcloud.outputPointCloud.link(xoutPointCloud.input)
        
        self.cam_rgb, self.xout_rgb, self.xout_rgb_preview = self.getRGBCamera(preview_size=self.rgb_nn_size)

        if self.enable_nn:
            self.detection_nn = self.pipeline.create(depthai.node.MobileNetDetectionNetwork)
            self.detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
            self.detection_nn.setConfidenceThreshold(0.5)
            self.cam_rgb.preview.link(self.detection_nn.input)

            self.xout_nn = self.pipeline.create(depthai.node.XLinkOut)
            self.xout_nn.setStreamName("nn")
            self.detection_nn.out.link(self.xout_nn.input)

    def startDevice(self):
        self.device = depthai.Device(self.pipeline)
        self.loadCalibrationData()
        
        self.q_rgb = self.device.getOutputQueue("rgb", maxSize=1)
        self.q_rgb_preview = self.device.getOutputQueue("rgb_preview", maxSize=1, blocking=False)
        
        if self.enable_depth:
            self.q_depth = self.device.getOutputQueue("depth", maxSize=1, blocking=False)
            self.q_disparity = self.device.getOutputQueue("disparity", maxSize=1, blocking=False)
            self.q_ml = self.device.getOutputQueue("rectifiedLeft", maxSize=1, blocking=False)
            self.q_mr = self.device.getOutputQueue("rectifiedRight", maxSize=1, blocking=False)
            self.disparity_multiplier = 255 / self.cam_depth.getMaxDisparity()
            if self.enable_pointcloud:
                self.q_pointcloud = self.device.getOutputQueue("pointcloud", maxSize=1, blocking=False)
        else:
            self.q_ml = self.device.getOutputQueue("left", maxSize=1, blocking=False)
            self.q_mr = self.device.getOutputQueue("right", maxSize=1, blocking=False)

        if self.enable_nn:
            self.q_nn = self.device.getOutputQueue("nn")
            self.nn_frame = 'latest_frame_rgb_preview'
            self.detections = []
            self.latest_nn_frame = None

    def startAruco(self):
        self.aruco_frame = frame_specs[self.aruco_input]['latest_frame']
        self.aruco_detector = ArucoDetector(
            dict_type=cv2.aruco.DICT_6X6_50,
            marker_length=4.6,
            camera_matrix=self.intrinsics[frame_specs[self.aruco_input]['calibkey']], 
            dist_coeffs=self.dist_coeffs[frame_specs[self.aruco_input]['calibkey']]
        )
        self.latest_aruco_frame = None
        self.latest_aruco_corners = None
        self.latest_aruco_ids = None
        self.latest_aruco_rois = None

    def startResconstruction(self):
        self.reconstruction = SceneReconstruction()

    @property
    def reconstruction_target(self):
        return self.reconstruction.target
    
    @reconstruction_target.setter
    def reconstruction_target(self, target):
        self.reconstruction.target = target
    
    def loadCalibrationData(self):
        calib_data = self.device.readCalibration()
       
        # Intrinsics
        self.intrinsics['rgb'] = np.array(calib_data.getCameraIntrinsics(depthai.CameraBoardSocket.RGB))
        self.intrinsics['left'] = np.array(calib_data.getCameraIntrinsics(depthai.CameraBoardSocket.LEFT))
        self.intrinsics['right'] = np.array(calib_data.getCameraIntrinsics(depthai.CameraBoardSocket.RIGHT))
        
        # Distortion coefficients
        self.dist_coeffs['rgb'] = np.array(calib_data.getDistortionCoefficients(depthai.CameraBoardSocket.RGB))
        self.dist_coeffs['left'] = np.array(calib_data.getDistortionCoefficients(depthai.CameraBoardSocket.LEFT))
        self.dist_coeffs['right'] = np.array(calib_data.getDistortionCoefficients(depthai.CameraBoardSocket.RIGHT))
        
        # Assume rectified frames have the same intrinsics as the original left/right frames
        self.intrinsics['left_rectified'] = self.intrinsics['left']
        self.intrinsics['right_rectified'] = self.intrinsics['right']
        
        # No distortion in rectified images
        self.dist_coeffs['left_rectified'] = np.zeros_like(self.dist_coeffs['left'])
        self.dist_coeffs['right_rectified'] = np.zeros_like(self.dist_coeffs['right'])
        
        identity_extrinsic = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]

        identity_extrinsic_matrix = np.matrix(identity_extrinsic)

        # Add identity extrinsics for rectified transformations
        self.extrinsics['left_to_left_rectified'] = identity_extrinsic_matrix
        self.extrinsics['left_rectified_to_left'] = identity_extrinsic_matrix
        self.extrinsics['right_to_right_rectified'] = identity_extrinsic_matrix
        self.extrinsics['right_rectified_to_right'] = identity_extrinsic_matrix

        # Extrinsics
        self.extrinsics['rgb_to_left'] = np.matrix(calib_data.getCameraExtrinsics(depthai.CameraBoardSocket.RGB, depthai.CameraBoardSocket.LEFT))
        self.extrinsics['left_to_rgb'] = np.matrix(calib_data.getCameraExtrinsics(depthai.CameraBoardSocket.LEFT, depthai.CameraBoardSocket.RGB))
        self.extrinsics['left_to_right'] = np.matrix(calib_data.getCameraExtrinsics(depthai.CameraBoardSocket.LEFT, depthai.CameraBoardSocket.RIGHT))
        self.extrinsics['right_to_left'] = np.matrix(calib_data.getCameraExtrinsics(depthai.CameraBoardSocket.RIGHT, depthai.CameraBoardSocket.LEFT))
        self.extrinsics['right_rectified_to_left_rectified'] = self.extrinsics['left_to_right']  # Same as left to right

        # FoV
        self.fov["rgb"] = calib_data.getFov(depthai.CameraBoardSocket.CAM_A)
        self.fov["left"] = calib_data.getFov(depthai.CameraBoardSocket.CAM_B)
        self.fov["right"] = calib_data.getFov(depthai.CameraBoardSocket.CAM_C)
        # Baseline (distance between left and right cameras) and focal length for depth calculations
        # self.baseline = self.extrinsics['left_to_right'][1][0]
        # self.focal_length = self.intrinsics['left'][0, 0]

    def getFrame(self, queue):

        if queue.has():
            frame = queue.get()
            return frame.getCvFrame()
        else:
            return None
        
    def frameNorm(self, frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0 if bbox[1] < 1 else 1])
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

    def colorizeDisparity(self, frame):
        frame = (frame * self.disparity_multiplier).astype(np.uint8)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        return frame

    def update_frames(self):
        rv = FrameSet()

        frame_rgb = self.getFrame(self.q_rgb)
        frame_rgb_preview = self.getFrame(self.q_rgb_preview)
        frame_l = self.getFrame(self.q_ml)
        frame_r = self.getFrame(self.q_mr)

        if frame_rgb is not None:
            self.latest_frame_rgb = frame_rgb    
        if frame_l is not None:
            self.latest_frame_l = frame_l
        if frame_r is not None:
            self.latest_frame_r = frame_r
        if frame_rgb_preview is not None:
            self.latest_frame_rgb_preview = frame_rgb_preview

        rv.left = self.latest_frame_l
        rv.right = self.latest_frame_r
        rv.rgb = self.latest_frame_rgb 
        rv.rgb_preview = self.latest_frame_rgb_preview

        if self.enable_depth:
            frame_depth = self.getFrame(self.q_depth)
            frame_disparity = self.getFrame(self.q_disparity)
            if frame_depth is not None:
                self.latest_frame_depth = frame_depth
            if frame_disparity is not None:
                frame_disparity = self.colorizeDisparity(frame_disparity)
                self.latest_frame_disparity = frame_disparity
            rv.depth = self.latest_frame_depth
            rv.disparity = self.latest_frame_disparity

            if self.enable_pointcloud:
                pointcloud = None
                if self.q_pointcloud.has():
                    pointcloud = self.q_pointcloud.get()
                if pointcloud:
                    self.latest_pointcloud = pointcloud
                rv.pointcloud = self.latest_pointcloud

        if self.enable_nn:
            in_nn = self.q_nn.tryGet()
            
            if in_nn:
                self.detections = in_nn.detections

            if getattr(self, self.nn_frame) is not None:
                nn_frame = getattr(self, self.nn_frame).copy()
                for detection in self.detections:
                    bbox = self.frameNorm(nn_frame, 
                                    (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                                )
                    cv2.rectangle(nn_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                self.latest_nn_frame = nn_frame
            
            rv.nn = self.latest_nn_frame

        if self.enable_aruco:
            if getattr(self, self.aruco_frame) is not None:
                aruco_frame = getattr(self, self.aruco_frame)
                self.latest_aruco_corners, self.latest_aruco_ids = self.aruco_detector.detect_markers(
                    aruco_frame, is_grey=frame_specs[self.aruco_input]['is_grey']
                )

                if self.enable_depth:
                    aruco_spatials = {}
                    self.latest_aruco_rois = self.roi.get_square_rois(
                        self.latest_aruco_corners, fname=self.aruco_input, geom=self.geom
                    )
                    for idx, roi in enumerate(self.latest_aruco_rois):
                        spatial, _centroid = self.spatial.compute_spatial(self.latest_frame_depth, roi)
                        aruco_spatials[self.latest_aruco_ids.ravel().tolist()[idx]] = spatial
                    self.latest_aruco_spatials = aruco_spatials
                    rv.aruco_spatials = self.latest_aruco_spatials
                
                aruco_frame = getattr(self, self.aruco_frame).copy()
                self.latest_aruco_frame = self.aruco_detector.draw_markers(
                    aruco_frame, rois=self.latest_aruco_rois, 
                    is_grey=frame_specs[self.aruco_input]['is_grey']
                )
                
            rv.aruco = self.latest_aruco_frame
        
        if self.enable_reconstruction:
            if self.enable_aruco and not self.enable_depth:
                self.reconstruction.clear_plane(name='aruco_plane')
                self.reconstruction.clear_markers(name='aruco_marker')
                if self.latest_aruco_corners is not None and self.latest_aruco_ids is not None:
                    rvecs, tvecs = self.aruco_detector.estimate_poses()
                    centroid, normal = self.spatial.compute_average_plane(tvecs)
                    self.reconstruction.upsert_plane(centroid=centroid, normal=normal, name='aruco_plane', color='grey2')
                    for i, tvec in enumerate(tvecs):
                        self.reconstruction.upsert_marker(tvec=tvec, name=f'aruco_marker_{i}', color='r1')
            if self.enable_aruco and self.enable_depth:
                self.reconstruction.clear_plane(name='aruco_plane')
                self.reconstruction.clear_markers(name='aruco_markers')
                centroid, normal = self.spatial.compute_average_plane(list(self.latest_aruco_spatials.values()))
                self.reconstruction.upsert_plane(centroid=centroid, normal=normal, name='aruco_plane', color='orange5')
                for i, vec in self.latest_aruco_spatials.items():
                    self.reconstruction.upsert_marker(tvec=vec, name=f'aruco_marker_{i}', color='r7')
            self.reconstruction.upsert_camera(geom=self.geom)
        return self.postprocess_frames(rv)
    
    def postprocess_frames(self, frames: FrameSet):
        if self.enable_aruco and len(self.draw_aruco_on) and self.aruco_detector.ids is not None and len(self.aruco_detector.ids) > 0:
            for fname in self.draw_aruco_on:
                # TODO The transforms don't presently work. Additional frame drawing can only 
                #      be done on frames with a shared coordinate system.
                fidx = frame_specs[fname]['seq']
                if 'calibkey' in frame_specs[fname]:
                    print(f'Trying to draw aruco on {fname}')
                    self.aruco_detector.draw_markers_on(getattr(frames, fname), self.coordinates, self.aruco_input, fname)
                else:
                    self.aruco_detector.draw_markers(getattr(frames, fname), is_grey=frame_specs[fname]['is_grey'])
        return frames
    
    def close(self):
        if self.enable_reconstruction:
            self.reconstruction.close()
        if self.device:
            self.device.close()


def cleanup_depthai(depthai_handler):
    depthai_handler.close()
