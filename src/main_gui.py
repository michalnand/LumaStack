import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout,
    QPushButton, QWidget, QHBoxLayout, QScrollArea, QGridLayout
)
from PySide6.QtGui import QPixmap, QImage, QAction  # QAction moved here in Qt6
from PySide6.QtCore import Qt

from app_core import *

import numpy


def numpy_to_qpixmap(img_np):
    img = numpy.clip(img_np * 255, 0, 255).astype(numpy.uint8)
    h, w, _ = img.shape
    qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)  


class MainWindow(QMainWindow):
    def __init__(self, app_backend):
        super().__init__()
        self.app = app_backend
        self.setWindowTitle("LumaStack")

        self.setAcceptDrops(True)


        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Result and control panel layout
        center_layout = QHBoxLayout()
        main_layout.addLayout(center_layout)

        # Result viewer
        self.result_label = QLabel("Result Preview")
        self.result_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.result_label, stretch=3)

        # Control panel
        controls = QVBoxLayout()
        center_layout.addLayout(controls, stretch=1)

        self.stack_btn = QPushButton("Stack Images")
        self.stack_btn.clicked.connect(self.stack_images)
        controls.addWidget(self.stack_btn)

        controls.addSpacing(50)

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export)
        controls.addWidget(self.export_btn)

        self.export_cropped_btn = QPushButton("Export Cropped")
        self.export_cropped_btn.clicked.connect(self.export_cropped)
        controls.addWidget(self.export_cropped_btn)
        controls.addStretch()

        # Thumbnail area
        self.thumb_area = QScrollArea()
        self.thumb_area.setMaximumHeight(150)
        self.thumb_container = QWidget()
        self.thumb_layout = QGridLayout(self.thumb_container)
        self.thumb_area.setWidgetResizable(True)
        self.thumb_area.setWidget(self.thumb_container)
        main_layout.addWidget(self.thumb_area)

        # Menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_action)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.app.load_files(folder)
            self.show_thumbnails()

    def show_thumbnails(self):
        for i in reversed(range(self.thumb_layout.count())):
            widget = self.thumb_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        thumbs = self.app.get_thumbnails()
        for idx, thumb_np in enumerate(thumbs):
            pixmap = numpy_to_qpixmap(thumb_np)
            label = QLabel()
            label.setPixmap(pixmap.scaledToWidth(150, Qt.SmoothTransformation))
            #label.setPixmap(pixmap.scaledToHeight(150, Qt.SmoothTransformation))
            self.thumb_layout.addWidget(label, idx // 5, idx % 5)

    def stack_images(self):
        self.app.process()
        result = self.app.get_result()
        if result is not None:
            pixmap = numpy_to_qpixmap(result)
            self.result_label.setPixmap(pixmap.scaled(
                self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def export(self):
        self.app.export()

    def export_cropped(self):
        self.app.export_cropped()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        
        if urls:
            files = []
            for n in range(len(urls)):
                files.append(str(urls[n].toLocalFile()))

            self.app.load_files(files)
            self.show_thumbnails()


if __name__ == "__main__":

    app_core = APPCore()

    qt_app = QApplication(sys.argv)
    
    window = MainWindow(app_core)
    window.resize(1000, 600)
    window.show()

    sys.exit(qt_app.exec_())