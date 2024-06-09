
import numpy as np
from geometries import GeometriesManager

default_roi_size = 10


class ROICalculators:
    def _check_input(self, roi, fname=None, geom: GeometriesManager=None):
        if len(roi) == 4:
            return roi
        elif len(roi) != 2:
            raise ValueError("You have to pass either ROI (4 values) or point (2 values)!")
        else:
            # Convert point to ROI
            x = min(max(roi[0], default_roi_size/2), 
                    geom.frame_sizes[fname][0] - default_roi_size/2)
            y = min(max(roi[1], default_roi_size/2), 
                    geom.frame_sizes[fname][1] - default_roi_size/2)
            
            return [
                x - default_roi_size/2, 
                y - default_roi_size/2, 
                x + default_roi_size/2, 
                y + default_roi_size/2
            ]

    def get_square_roi(self, corners, fname=None, geom=None):
        # Find min and max coordinates of the marker
        min_x = np.min(corners[:,:,0])
        max_x = np.max(corners[:,:,0])
        min_y = np.min(corners[:,:,1])
        max_y = np.max(corners[:,:,1])

        # Calculate width and height of the marker
        width = max_x - min_x
        height = max_y - min_y

        # Define the size of the square ROI to fit within the marker
        roi_size = min(width, height)

        # Calculate center of the marker
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Define the square ROI
        roi = np.array([
            center_x - roi_size/2, 
            center_y - roi_size/2,
            center_x + roi_size/2, 
            center_y + roi_size/2
        ])

        return roi.astype(int)

    def get_square_rois(self, markers, fname, geom=None):
        rois = []
        for marker in markers:
            rois.append(self.get_square_roi(marker))
        return rois
