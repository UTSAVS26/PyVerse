import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QStackedWidget, QMessageBox
from application import Ui_MainWindow
from info import Ui_DetailsWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(803, 616)
        self.setWindowTitle("Movie Finder")
        
        # Create stacked widget
        self.stacked_widget = QStackedWidget()
        
        # Create widgets for each window
        self.window1 = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.window1)
        button1 = self.ui.getMovieBtn
        button1.clicked.connect(self.show_window2)
        
        self.window2 = QMainWindow()
        self.ui_info = Ui_DetailsWindow()
        self.ui_info.setupUi(self.window2)
        button2 = self.ui_info.backBtn
        button2.clicked.connect(self.show_window1)
        
        # Add windows to stacked widget
        self.stacked_widget.addWidget(self.window1)
        self.stacked_widget.addWidget(self.window2)
        
        self.setCentralWidget(self.stacked_widget)
    
    def show_window1(self):
        self.stacked_widget.setCurrentWidget(self.window1)
    
    def show_window2(self):
        movies = self.ui.getMovie()
        #check if movies is empty, if so show a message box with text No Movies Found
        if not movies:
            # Create QMessageBox instance
            message = QMessageBox()

            # Set window title
            message.setWindowTitle("Error")

            # Set message text
            message.setText("No Movies Found")

            # Set message box icon to critical (error)
            message.setIcon(QMessageBox.Critical)

            # Display the message box
            message.exec_()
            return
        self.ui_info.setMovie(movies)
        self.stacked_widget.setCurrentWidget(self.window2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())