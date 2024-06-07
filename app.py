import sys
import cv2
import numpy as np
import logging
from PyQt6.QtWidgets import QApplication, QSizePolicy
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QScrollArea
from PyQt6.QtWidgets import QWidget, QFrame, QLabel, QPushButton, QSpacerItem
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from depthai_handler import DepthAIHandler, cleanup_depthai
from vedo import Plotter


logging.basicConfig(level=logging.DEBUG)

enable_nn=True
enable_aruco=True
enable_depth=True
enable_reconstruction=True

# Left input should simplify the code during testing because frame translations 
# are not needed for getting to the depth frame. However, we probably want the 
# higher resolution of RGB in the deployable version. The translation of the 
# point from RGB to left is still pretty broken, though. 
aruco_input='rgb'
draw_aruco_on = ['left']


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
        self.initUI()

    def updateFrames(self):
        frames = self.depthai_handler.update_frames()
        self.l_label.setPixmap(convert_cv_qt(frames.pop(0)))
        self.r_label.setPixmap(convert_cv_qt(frames.pop(0)))
        self.rgb_label.setPixmap(convert_cv_qt(frames.pop(0)))
        _ = frames.pop(0)
        if enable_depth:
            self.depth_label.setPixmap(convert_cv_qt(frames.pop(0)))
            self.disparity_label.setPixmap(convert_cv_qt(frames.pop(0)))
        if enable_nn:
            self.nn_label.setPixmap(convert_cv_qt(frames.pop(0)))
        if enable_aruco:
            self.aruco_label.setPixmap(convert_cv_qt(frames.pop(0)))

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
            layout.addWidget(QLabel("Scene Reconstruction"))
            layout.addWidget(self.reconstruction_widget)
            layout.addWidget(update_button)
            frame.setLayout(layout)
            frame.setMinimumWidth(400)
            self.depthai_handler.reconstruction.update_scene()
            return frame

    def buildResultsRow(self):
        layout = QHBoxLayout()
        if enable_nn:
            nn_frame = self.create_stream_frame_with_label("NN Detections", self.nn_label)
            layout.addWidget(nn_frame)
        if enable_aruco:
            aruco_frame = self.create_stream_frame_with_label("Aruco Detections", self.aruco_label)
            layout.addWidget(aruco_frame)
        return layout
    
    def buildMonoSourcesRow(self):
        layout = QHBoxLayout()
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
        layout = QHBoxLayout()
        rgb_frame = self.create_stream_frame_with_label("RGB", self.rgb_label)
        layout.addWidget(rgb_frame)
        disparity_frame = self.create_stream_frame_with_label("Disparity", self.disparity_label)
        layout.addWidget(disparity_frame)
        return layout

    def initUI(self):
        title = "DepthAI Sandbox"
        self.setWindowTitle(title)
    
        self.initStreamHolders()
    
        self.layout = QVBoxLayout(self)
        self.main_layout = QHBoxLayout()
        self.streams_layout = QVBoxLayout()
        
        self.results_row = self.buildResultsRow()
        self.streams_layout.addLayout(self.results_row)

        self.complex_sources_row = self.buildComplexSourcesRow()
        self.streams_layout.addLayout(self.complex_sources_row)
        
        self.mono_sources_row = self.buildMonoSourcesRow()
        self.streams_layout.addLayout(self.mono_sources_row)

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
            
        self.layout.addLayout(self.main_layout)

        self.applyTheme()

        # Timer for updating frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateFrames)
        self.timer.start(30)

    def adjust_scroll_area_width(self, scroll_area, content):
        def adjust_width():
            scroll_area.setMinimumWidth(content.sizeHint().width() + 30)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(adjust_width)
        self.timer.start(30)
        adjust_width()

    def applyTheme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QWidget {
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                    color: #f0f0f0;
                    background-color: #2e2e2e;
                }
                QFrame {
                    border: 1px solid #555;
                    border-radius: 4px;
                    margin: 4px;
                    padding: 4px;
                }
                QLabel {
                    color: #f0f0f0;
                    border: 0px;
                }
                QPushButton {
                    background-color: #444;
                    border: 1px solid #555;
                    color: #f0f0f0;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #555;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget {
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                    color: #333;
                    background-color: #f0f0f0;
                }
                QFrame {
                    border: 1px solid #dcdcdc;
                    border-radius: 4px;
                    margin: 4px;
                    padding: 4px;
                }
                QLabel {
                    color: #333;
                }
                QPushButton {
                    background-color: #eee;
                    border: 1px solid #dcdcdc;
                    color: #333;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #ddd;
                }
            """)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    depthai_handler = DepthAIHandler(
        enable_nn=enable_nn, 
        enable_aruco=enable_aruco,
        enable_reconstruction=enable_reconstruction,
        enable_depth=enable_depth,
        aruco_input=aruco_input,
        draw_aruco_on=draw_aruco_on
    )
    ex = App(depthai_handler, dark_mode=True)
    ex.show()
    
    def close_app():
        cleanup_depthai(depthai_handler)
        app.quit()
        
    app.aboutToQuit.connect(close_app)
    sys.exit(app.exec())
