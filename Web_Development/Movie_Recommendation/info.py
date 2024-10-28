from PyQt5 import QtCore, QtGui, QtWidgets

class HorizontalLineDelegate(QtWidgets.QStyledItemDelegate):
    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        
        # Draw a horizontal line at the bottom of the item if it's not a child item
        if not index.parent().isValid():
            line_width = 200  # Adjust the width of the line
            line_y = option.rect.bottom()  # Y-coordinate of the line
            line_x_left = option.rect.left()  # X-coordinate of the left end of the line
            line_x_right = line_x_left + line_width  # X-coordinate of the right end of the line

            painter.setPen(QtGui.QPen(QtGui.QColor('black')))
            painter.drawLine(line_x_left, line_y, line_x_right, line_y)

class Ui_DetailsWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(803, 616)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.moviesDetails = QtWidgets.QTreeView(self.centralwidget)
        self.moviesDetails.setGeometry(QtCore.QRect(10, 90, 781, 431))
        self.moviesDetails.setObjectName("moviesDetails")
        self.backBtn = QtWidgets.QPushButton(self.centralwidget)
        self.backBtn.setGeometry(QtCore.QRect(690, 550, 75, 23))
        self.backBtn.setObjectName("backBtn")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(130, 10, 381, 61))
        font = QtGui.QFont()
        font.setPointSize(36)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 70, 811, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(-23, 530, 831, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 10, 61, 61))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("logoMovieFinder.svg"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 803, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.treeInit()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    
    def setMovie(self, movies):
        model = QtGui.QStandardItemModel()
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(16)  # Set the font size to 16
        for movie, details in movies.items():
            movie_item = QtGui.QStandardItem(movie)
            movie_item.setFont(font)  # Set the font to the item
            model.appendRow(movie_item)
            for detail in details:
                detail_item = QtGui.QStandardItem(detail)
                movie_item.appendRow(detail_item)

        self.moviesDetails.setModel(model)
        delegate = HorizontalLineDelegate(self.moviesDetails)
        self.moviesDetails.setItemDelegate(delegate)

    def treeInit(self):
        self.moviesDetails.setHeaderHidden(True)
        #self.moviesDetails.setAlternatingRowColors(True)
        self.moviesDetails.setSortingEnabled(True)
        self.moviesDetails.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.moviesDetails.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.moviesDetails.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        #self.model = QtGui.QStandardItemModel()
        #self.moviesDetails.setModel(self.model)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MovieInfo"))
        self.backBtn.setText(_translate("MainWindow", "Back"))
        self.label.setText(_translate("MainWindow", "The Movie Finder"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_DetailsWindow()
    ui.setupUi(MainWindow)
    movies = {
        "Movie 1": "Details for Movie 1",
        "Movie 2": "Details for Movie 2",
        "Movie 3": "Details for Movie 3"
    }
    ui.setMovie(movies)
    MainWindow.show()
    sys.exit(app.exec_())
