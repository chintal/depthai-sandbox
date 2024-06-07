
from typing import Dict
import numpy as np
import cv2
import logging


"""
# From ChatGPT. To crosscheck with docs.

intrinsics:
shape (3, 3)
This is a 3x3 matrix that represents the intrinsic properties of the camera. 
It contains parameters such as the focal length (in pixels) along the x and 
y axes, the optical center (principal point) coordinates, and skew factor. 
The camera matrix is denoted by K and has the following form:

fx s  cx
0  fy cy
0  0   1

- fx and fyfy are the focal lengths in pixels along the x and y axes respectively.
- cx and cy  are the coordinates of the optical center (principal point) in pixels.
- s is the skew factor, usually 0 for most cameras.

dist_coeffs: 
shape (1, 5)
These coefficients are used to correct for radial and tangential lens distortion. 
The distortion model typically includes five coefficients: k1,k2,p1,p2,k3. 
The distortion model can be represented as:

k1 k2 p1 p2 k3

- k1,k2,k3 are radial distortion coefficients.
- p1,p2 are tangential distortion coefficients.

extrinsics: 
shape (4, 4)
Representing transformation matrices.

| r00 r01 r02 Tx |
| r10 r11 r12 Ty |
| r20 r21 r22 Tz |
|  0   0   0   1 |

"""

class ReferenceFramesManager(object):
    def __init__(self) -> None:
        self.frame_sizes = {}
        self.intrinsics: Dict[str, np.ndarray] = {}
        self.dist_coeffs: Dict[str, np.ndarray] = {}
        self.extrinsics: Dict[str, np.ndarray] = {}

    def print(self):
        logging.info("#####   Frame Sizes  #####")
        self.printInfo(self.frame_sizes)
        logging.info("#####   Intrinsics  #####")
        self.printInfo(self.intrinsics)
        logging.info("#####   Dist Coeffs  #####")
        self.printInfo(self.dist_coeffs)
        logging.info("")
        logging.info("#####   Extrinsics  #####")
        self.printInfo(self.extrinsics)
        logging.info("")

    def printInfo(self, infodict):
        for key, value in infodict.items():
            logging.info (f"--- {key} ----")
            if isinstance(value, np.ndarray):
                logging.info(np.array_str(value, precision=4, suppress_small=False))
            else:
                logging.info(value)

    def transformPoint(self, point, from_frame, to_frame):
        M_from = self.intrinsics[from_frame]
        D_from = self.dist_coeffs[from_frame]
        M_to = self.intrinsics[to_frame]
        D_to = self.dist_coeffs[to_frame]
        extrinsics = self.extrinsics[f'{from_frame}_to_{to_frame}']
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]

        from_size = self.frame_sizes[from_frame]
        to_size = self.frame_sizes[to_frame]

        logging.debug(f"Transforming point {point} from {from_frame} to {to_frame}")
        logging.debug(f"Intrinsics from: {M_from}, Distortion from: {D_from}")
        logging.debug(f"Intrinsics to: {M_to}, Distortion to: {D_to}")
        logging.debug(f"Extrinsics: {extrinsics}")

        # Normalize the point to the range [0, 1] based on the from_frame size
        point_normalized = np.array([point[0] / from_size[0], point[1] / from_size[1]], dtype=np.float32)
        logging.debug(f"Normalized point: {point_normalized}")

        # Convert normalized 2D point to pixel coordinates in from_frame
        point_pixel = np.array([point_normalized[0] * from_size[0], point_normalized[1] * from_size[1]], dtype=np.float32)
        
        # Undistort the point
        point_undistorted = cv2.undistortPoints(np.array([[point_pixel]], dtype=np.float32), M_from, D_from, None, M_from)
        point_undistorted = np.append(point_undistorted[0][0], 1.0)
        logging.debug(f"Undistorted point: {point_undistorted}")

        # Convert to 3D point in from_frame coordinates
        point_3d = np.dot(np.linalg.inv(M_from), point_undistorted)
        logging.debug(f"Point in 3D (from_frame coords): {point_3d}")

        # Apply extrinsic transformation to get the point in to_frame coordinates
        point_transformed_3d = np.dot(R, point_3d) + t
        logging.debug(f"Point transformed in 3D (to_frame coords): {point_transformed_3d}")

        # Project transformed 3D point back to 2D in to_frame coordinates
        point_transformed_2d, _ = cv2.projectPoints(point_transformed_3d[:3], np.zeros(3), np.zeros(3), M_to, D_to)
        logging.debug(f"Point projected in 2D (to_frame): {point_transformed_2d}")

        # Convert projected 2D point to normalized coordinates in to_frame
        point_transformed_2d_normalized = np.array([point_transformed_2d[0][0][0] / to_size[0], point_transformed_2d[0][0][1] / to_size[1]], dtype=np.float32)
        
        # Denormalize the point back to the pixel coordinates of the to_frame
        point_denormalized = np.array([point_transformed_2d_normalized[0] * to_size[0], point_transformed_2d_normalized[1] * to_size[1]], dtype=np.float32)
        logging.debug(f"Denormalized point: {point_denormalized}")

        return point_denormalized

