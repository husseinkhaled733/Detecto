import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ultralytics import YOLO
from Detection import Ui_MainWindow
import os


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # path of images
        self.fname = []
        # path of results
        self.results = []
        # self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        self.pixmap = None
        # to indicate the number of the image which is shown now
        self.counter = 0
        self.showResults = False
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ROOT_DIR = os.path.abspath("")

        self.ui.Select_btn.clicked.connect(self.choose_photos)
        self.ui.Front_btn.clicked.connect(self.forward)
        self.ui.Back_btn.clicked.connect(self.back)
        self.ui.Run_btn.clicked.connect(self.detect)
        self.ui.Original_btn.clicked.connect(self.change_state)

        self.ui.Back_btn.setIcon(QIcon('back-button.png'))
        self.ui.Front_btn.setIcon((QIcon('right.png')))
        self.ui.Run_btn.setIcon(QIcon('run.png'))
        self.ui.Select_btn.setIcon(QIcon('select.png'))
        self.ui.Original_btn.setIcon(QIcon('undo.png'))

    def choose_photos(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose Images", "", "All Files (*)")

        for file in files:
            self.fname.append(file)

        print(str(self.fname))
        self.fname.sort()
        self.show_photos()

    def forward(self):
        if self.showResults:
            if self.counter < len(self.results) - 1:
                self.counter = self.counter + 1
                print(self.counter)
        else:
            if self.counter < len(self.fname) - 1:
                self.counter = self.counter + 1
                print(self.counter)

        self.show_photos()

    def back(self):
        if self.counter > 0:
            self.counter = self.counter - 1
            self.show_photos()

    def detect(self):
        model = YOLO(self.ROOT_DIR+"/train/weights/best.pt")
        results = model.predict(self.fname, save=True, device='cpu')
        files = os.listdir(results[0].save_dir)
        # images = [file for file in files if any(file.endswith(ext) for ext in self.image_extensions)]
        self.results = []
        for file in files:
            self.results.append(results[0].save_dir + '/' + file)

        self.results.sort()
        self.show_results()

    def show_results(self):
        self.showResults = True
        self.show_photos()

    def show_photos(self):
        if self.showResults and len(self.results) > 0:
            self.pixmap = QPixmap(self.results[self.counter])
            # to scale the image to fit the label
            self.pixmap = self.pixmap.scaled(self.ui.photo.size().width(), self.ui.photo.size().height(),
                                             Qt.KeepAspectRatio, Qt.FastTransformation)
            self.ui.photo.setPixmap(self.pixmap)
            self.ui.Original_btn.setIcon(QIcon('undo.png'))
        elif len(self.fname) > 0:
            self.pixmap = QPixmap(self.fname[self.counter])
            # to scale the image to fit the label
            self.pixmap = self.pixmap.scaled(self.ui.photo.size().width(), self.ui.photo.size().height(),
                                             Qt.KeepAspectRatio, Qt.FastTransformation)
            self.ui.photo.setPixmap(self.pixmap)
            self.ui.Original_btn.setIcon(QIcon('redo.png'))

    def change_state(self):
        if (not self.showResults) and (self.counter >= len(self.results)):
            dlg = QMessageBox(self)
            dlg.setWindowTitle("No Results Found")
            dlg.setText("Press the Run button to see the Results")
            dlg.exec()
        else:
            self.showResults = not self.showResults
            self.show_photos()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
