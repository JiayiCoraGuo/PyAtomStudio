# AmpMap_GUI
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QToolButton,QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from QupyRibbon.GUI.RibbonWidget import RibbonWidget
from AmpMap_Widgets import myWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AmpMap")
        self.setGeometry(100, 100, 1590, 1000)

        self.myWidget = myWidget(self)

        self.ribbon = RibbonWidget(self)
        self.addToolBar(self.ribbon)
        self.myWidget.init_ribbon(self.ribbon)

        self.initUI()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        grid = QGridLayout()
        grid.setSpacing(5)

        # Row 0: Original Image, Phase - A, Phase - B
        grid.addWidget(self.myWidget.initInput(), 0, 0)
        grid.addWidget(self.myWidget.initPhaseA(), 0, 1)
        grid.addWidget(self.myWidget.initPhaseB(), 0, 2)

        # Row 1: FFT, Strain Field - xx, Strain Field - yy
        grid.addWidget(self.myWidget.initFFT(), 1, 0)
        grid.addWidget(self.myWidget.initStrainXX(), 1, 1)
        grid.addWidget(self.myWidget.initStrainYY(), 1, 2)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        central_widget.setLayout(grid)




def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
