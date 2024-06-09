import sys
import cv2
import numpy as np
import logging
from datetime import datetime
from pprint import pp
from PyQt6.QtWidgets import QApplication, QSizePolicy
from PyQt6.QtWidgets import QVBoxLayout, QScrollArea, QGridLayout
from PyQt6.QtWidgets import \
    QWidget, QFrame, QLabel, QPushButton, QSpacerItem, \
    QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from depthai_handler import DepthAIHandler, cleanup_depthai
from vedo import Plotter
from widgets.qflowlayout import QFlowLayout
from widgets.qcustomsplitter import QCustomSplitter
from qt_material import apply_stylesheet

logging.basicConfig(level=logging.DEBUG)

enable_nn=True
enable_aruco=True
enable_depth=True
enable_pointcloud=False
enable_reconstruction=True

fps_rgb = 10
fps_mono = 10
depth_extended_disparity = False
depth_subpixel_disparity = True
depth_align = 'rgb'
depth_thresholds = (200, 2000)
depth_spatial_filter = False

# Left input should simplify the code during testing because frame translations 
# are not needed for getting to the depth frame. However, we probably want the 
# higher resolution of RGB in the deployable version. The translation of the 
# point from RGB to left is still pretty broken, though. 
aruco_input='rgb'
draw_aruco_on = ['disparity']


# Function to convert OpenCV images to QPixmap for display in PyQt6
def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap."""
    if cv_img is None:
        return QPixmap()
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    p = convert_to_Qt_format.scaled(640, 480, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
    return QPixmap.fromImage(p)


class AutoResizingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)

    def minimumSizeHint(self):
        return self.layout().minimumSize()

    def sizeHint(self):
        return self.layout().sizeHint()


class App(QWidget):
    def __init__(self, depthai_handler, dark_mode=False):
        super().__init__()
        self.dark_mode = dark_mode
        self.depthai_handler = depthai_handler
        self.initStats()
        self.initUI()

    def updateFrames(self):
        frames = self.depthai_handler.update_frames()
        self.l_label.setPixmap(convert_cv_qt(frames.left))
        self.r_label.setPixmap(convert_cv_qt(frames.right))
        self.rgb_label.setPixmap(convert_cv_qt(frames.rgb))
        _ = frames.rgb_preview
        if enable_depth:
            self.depth_label.setPixmap(convert_cv_qt(frames.depth))
            self.disparity_label.setPixmap(convert_cv_qt(frames.disparity))
        if enable_nn:
            self.nn_label.setPixmap(convert_cv_qt(frames.nn))
        if enable_aruco:
            self.aruco_label.setPixmap(convert_cv_qt(frames.aruco))
            if enable_depth:
                self.showArucoSpatials(frames.aruco_spatials)

        # TODO This FPS measurement is BS. FPS needs to be measured in 
        #      the depthai_handler on a per-stream level.
        self.frame_count += 1
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
            self.fps_label.setText(f"{fps:.2f}")

        # Reset counters every second to keep the display updated
        if elapsed_time >= 1.0:
            self.frame_count = 0
            self.start_time = datetime.now()

    def create_stream_frame_with_label(self, frame_name, label):
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.Box)
        frame.setFrameShadow(QFrame.Shadow.Raised)
        frame_layout = QVBoxLayout()
        frame_layout.addWidget(label)
        frame_name_label = QLabel(frame_name)
        frame_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(frame_name_label)
        frame.setLayout(frame_layout)
        frame.setMinimumWidth(label.sizeHint().width() + 20)
        return frame
    
    def initStreamHolders(self):
        self.rgb_label = QLabel()
        self.l_label = QLabel()
        self.r_label = QLabel()
        if enable_depth:
            self.depth_label = QLabel()
            self.disparity_label = QLabel()
        if enable_nn:
            self.nn_label = QLabel()
        if enable_aruco:
            self.aruco_label = QLabel()

    def initReconstruction(self):
        if enable_reconstruction:
            frame = QFrame()
            self.reconstruction_widget = QVTKRenderWindowInteractor(frame)
            self.reconstruction = Plotter(N=1, qt_widget=self.reconstruction_widget)
            self.depthai_handler.reconstruction_target = self.reconstruction
            update_button = QPushButton("Update Scene")
            update_button.clicked.connect(self.depthai_handler.reconstruction.update_scene)
            layout = QVBoxLayout()
            layout.addWidget(QLabel("Spatial Reconstruction"))
            layout.addWidget(self.reconstruction_widget)
            layout.addWidget(update_button)
            frame.setLayout(layout)
            frame.setMinimumWidth(400)
            self.depthai_handler.reconstruction.update_scene()
            return frame

    def buildResultsRow(self):
        layout = QFlowLayout()
        if enable_nn:
            nn_frame = self.create_stream_frame_with_label("NN Detections", self.nn_label)
            layout.addWidget(nn_frame)
        if enable_aruco:
            aruco_frame = self.create_stream_frame_with_label("Aruco Detections", self.aruco_label)
            layout.addWidget(aruco_frame)
        return layout
    
    def buildMonoSourcesRow(self):
        layout = QFlowLayout()
        if enable_depth:
            mono_type_name = "Rectified"
        else:
            mono_type_name = "Mono"
        left_frame = self.create_stream_frame_with_label(f"Left {mono_type_name}", self.l_label)
        layout.addWidget(left_frame)
        right_frame = self.create_stream_frame_with_label(f"Right {mono_type_name}", self.r_label)
        layout.addWidget(right_frame)
        return layout

    def buildComplexSourcesRow(self):
        layout = QFlowLayout()
        rgb_frame = self.create_stream_frame_with_label("RGB", self.rgb_label)
        layout.addWidget(rgb_frame)
        disparity_frame = self.create_stream_frame_with_label("Disparity", self.disparity_label)
        layout.addWidget(disparity_frame)
        return layout
    
    def showArucoSpatials(self, spatials):
        self.aruco_spatials_table.setRowCount(len(spatials))  # Adjust the number of rows

        for row, (key, values) in enumerate(spatials.items()):
            id_item = QTableWidgetItem(str(key))
            x_item = QTableWidgetItem(f"{values['x']:.2f}")
            y_item = QTableWidgetItem(f"{values['y']:.2f}")
            z_item = QTableWidgetItem(f"{values['z']:.2f}")

            # Center the text in each cell
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            x_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            y_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            z_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.aruco_spatials_table.setItem(row, 0, id_item)
            self.aruco_spatials_table.setItem(row, 1, x_item)
            self.aruco_spatials_table.setItem(row, 2, y_item)
            self.aruco_spatials_table.setItem(row, 3, z_item)

    def buildStats(self):
        stats_layout = QGridLayout()

        self.fps_label = QLabel("FPS: 0")

        if enable_aruco and enable_depth:
            self.aruco_spatials_table = QTableWidget()
            self.aruco_spatials_table.setColumnCount(4)
            self.aruco_spatials_table.setHorizontalHeaderLabels(["ID", "x", "y", "z"])
            self.aruco_spatials_table.horizontalHeader().setVisible(True)
            self.aruco_spatials_table.horizontalHeader().setMinimumHeight(20)
            self.aruco_spatials_table.verticalHeader().setVisible(False)
            self.aruco_spatials_table.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
            self.aruco_spatials_table.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            num_rows_to_display = 8
            row_height = self.aruco_spatials_table.verticalHeader().defaultSectionSize()
            header_height = self.aruco_spatials_table.horizontalHeader().height()
            total_height = header_height + (row_height * num_rows_to_display)
            self.aruco_spatials_table.setFixedHeight(total_height)

        stats_layout.addWidget(QLabel("FPS:"), 0, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        stats_layout.addWidget(self.fps_label, 0, 2, alignment=Qt.AlignmentFlag.AlignLeft)

        if enable_aruco and enable_depth:
            stats_layout.addWidget(QLabel("Aruco Spatials:"), 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
            stats_layout.addWidget(self.aruco_spatials_table, 1, 0, 10, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        # Set the column stretch to keep columns narrow and aligned to the left
        stats_layout.setColumnStretch(0, 0)
        stats_layout.setColumnStretch(1, 0)
        stats_layout.setColumnStretch(2, 1)

        stats_container = QWidget()
        stats_container.setLayout(stats_layout)
        stats_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        return stats_container

    def initStats(self):
        self.frame_count = 0
        self.start_time = datetime.now()

    def initUI(self):
        title = "DepthAI Sandbox"
        self.setWindowTitle(title)
    
        self.initStreamHolders()
    
        self.layout = QVBoxLayout(self)

        self.main_layout = QCustomSplitter(Qt.Orientation.Horizontal, 500, 300)
        self.streams_layout = QVBoxLayout()
        
        results_row = self.buildResultsRow()
        self.streams_layout.addLayout(results_row)

        complex_sources_row = self.buildComplexSourcesRow()
        self.streams_layout.addLayout(complex_sources_row)
        
        mono_sources_row = self.buildMonoSourcesRow()
        self.streams_layout.addLayout(mono_sources_row)

        self.streams_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        streams_scroll_area = QScrollArea()
        streams_scroll_area.setWidgetResizable(True)
        streams_scroll_content = AutoResizingWidget()
        streams_scroll_content.setLayout(self.streams_layout)
        streams_scroll_area.setWidget(streams_scroll_content)
        streams_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.adjust_scroll_area_width(streams_scroll_area, streams_scroll_content)

        self.main_layout.addWidget(streams_scroll_area)
        
        if enable_reconstruction:
            self.reconstruction_frame = self.initReconstruction()
            self.main_layout.addWidget(self.reconstruction_frame)
            
        self.layout.addWidget(self.main_layout)

        stats_container = self.buildStats()
        self.layout.addWidget(stats_container)

        self.resize(1000, 600)
        
        # Timer for updating frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateFrames)
        self.timer.start(100)

    def adjust_scroll_area_width(self, scroll_area, content):
        def adjust_width():
            scroll_area.setMinimumWidth(content.sizeHint().width() + 30)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(adjust_width)
        self.timer.start(30)
        adjust_width()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    depthai_handler = DepthAIHandler(
        enable_nn=enable_nn, 
        enable_aruco=enable_aruco,
        enable_reconstruction=enable_reconstruction,
        enable_depth=enable_depth,
        enable_pointcloud=enable_pointcloud,
        aruco_input=aruco_input,
        draw_aruco_on=draw_aruco_on,
        fps_rgb=fps_rgb, fps_mono=fps_mono,
        depth_extended_disparity=depth_extended_disparity,
        depth_subpixel_disparity=depth_subpixel_disparity,
        depth_align=depth_align,
        depth_thresholds=depth_thresholds,
        depth_spatial_filter=depth_spatial_filter
    )
    ui = App(depthai_handler, dark_mode=True)
    ui.show()

    apply_stylesheet(app, theme='dark_teal.xml')
    
    def close_app():
        cleanup_depthai(depthai_handler)
        app.quit()
        
    app.aboutToQuit.connect(close_app)
    sys.exit(app.exec())
