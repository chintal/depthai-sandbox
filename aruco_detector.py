import cv2
import numpy as np

class ArucoDetector:
    def __init__(self, dict_type=cv2.aruco.DICT_6X6_50, marker_length=4.6, camera_matrix=None, dist_coeffs=None):
        self.marker_length = marker_length
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.corners = None
        self.ids = None
        self.tvecs = None
        self.rvecs = None
        
    def detect_markers(self, frame, is_grey=False):
        use_frame = frame.copy()
        if not is_grey:
            use_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.corners, self.ids, _ = self.detector.detectMarkers(use_frame)
        return self.corners, self.ids

    def draw_markers(self, frame, is_grey=False):
        if is_grey:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if self.ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, self.corners, self.ids)
        return frame
    
    def draw_markers_on(self, frame, coordinates, from_frame, to_frame):
        if self.ids is not None:
            transformed_corners = []
            
            print("ORIGINAL")
            print(self.corners)
            for corner_set in self.corners:
                transformed_corner_set = []
                for point in corner_set[0]:
                    point_transformed = coordinates.transformPoint(point, from_frame, to_frame)
                    transformed_corner_set.append(point_transformed)
                transformed_corners.append(np.array(transformed_corner_set, dtype=np.float32).reshape((1, -1, 2)))
            
            print("TRANSFORMED")
            print(tuple(transformed_corners))

            h, w = frame.shape[:2]
            for corner_set in transformed_corners:
                for point in corner_set[0]:
                    if not (0 <= point[0] < w and 0 <= point[1] < h):
                        print(f"Point {point} out of bounds for frame of size ({w}, {h})")
                        return frame

            # If frame is grayscale, convert it to BGR for drawing
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            cv2.aruco.drawDetectedMarkers(frame, transformed_corners, self.ids)
            cv2.imshow("ArucoL", frame)
        return frame

    def estimate_poses(self):
        if self.corners is not None and self.ids is not None:
            rvecs = []
            tvecs = []
            object_points = np.array([
                [-self.marker_length / 2, self.marker_length / 2, 0],
                [self.marker_length / 2, self.marker_length / 2, 0],
                [self.marker_length / 2, -self.marker_length / 2, 0],
                [-self.marker_length / 2, -self.marker_length / 2, 0]
            ], dtype=np.float32)

            for corner in self.corners:
                ret, rvec, tvec = cv2.solvePnP(object_points, corner, self.camera_matrix, self.dist_coeffs)
                if ret:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
            self.rvecs = np.array(rvecs, dtype=np.float32)
            self.tvecs = np.array(tvecs, dtype=np.float32)
            return self.rvecs, self.tvecs
        return None, None
