

import math
import numpy as np
from geometries import GeometriesManager


class SpatialCalculators:
    def __init__ (self, geom: GeometriesManager, thresholds):
        self._geom = geom
        self._thresh_low = thresholds[0]
        self._thresh_high = thresholds[1]

    def _calc_angle(self, frame, offset, HFOV):
        return math.atan(math.tan(HFOV / 2.0) * offset / (frame.shape[1] / 2.0))
    
    def compute_spatial(self, depthFrame, roi, averaging_method=np.mean, fname='rgb'):
        xmin, ymin, xmax, ymax = roi
        depthRoI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self._thresh_low <= depthRoI) & (depthRoI <= self._thresh_high)
        HFOV = np.deg2rad(self._geom.fov['rgb'])
        averageDepth = averaging_method(depthRoI[inRange])
        
        centroid = { # Get centroid of the ROI
            'x': int((xmax + xmin) / 2),
            'y': int((ymax + ymin) / 2)
        }

        midW = int(depthFrame.shape[1] / 2) # middle of the depth img width
        midH = int(depthFrame.shape[0] / 2) # middle of the depth img height
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH

        angle_x = self._calc_angle(depthFrame, bb_x_pos, HFOV)
        angle_y = self._calc_angle(depthFrame, bb_y_pos, HFOV)

        spatials = {
            'z': averageDepth,
            'x': averageDepth * math.tan(angle_x),
            'y': -averageDepth * math.tan(angle_y)
        }
        return spatials, centroid
    
    def compute_average_plane(self, points):
        if not points:
            return None, None
        if isinstance(points[0], dict):
            points = np.array([[p['x'], p['y'], p['z']] for p in points])
        else:
            points = np.array(points).reshape(-1, 3)
        centroid = np.mean(points, axis=0)
        _, _, Vt = np.linalg.svd(points - centroid)
        plane_normal = Vt[2, :]
        return centroid, plane_normal

