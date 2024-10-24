from PyQt5 import QtCore, QtGui, QtWidgets
from owlready2 import *
from test import get_movies
from display_movie import get_movie_details

class MovieFinder():
    # Function to get the full "namespace#name" form of a class or property
    def get_full_name(self, entity):
        return "%s%s" % (entity.namespace.base_iri, entity.name)

    def __init__(self):
        # Load the ontology
        onto = get_ontology("data/Movies.rdf").load()

        # Get the namespace of the ontology
        namespace = onto.get_namespace("http://www.semanticweb.org/owl/owlapi/turtle#")
        # Define the rules
        with onto:
            hasGenre = self.get_full_name(namespace.hasGenre)
            hasActor = self.get_full_name(namespace.hasActor)
            movie = self.get_full_name(namespace.Movie)
            actor = self.get_full_name(namespace.Actor)
            genre = self.get_full_name(namespace.Genre)
            director = self.get_full_name(namespace.Director)
            
            class Movie(Thing):  # Specify Movie as a subclass of Thing
                pass
            class Director(Thing):  # Specify Director as a subclass of Thing
                pass
            class Actor(Thing):  # Specify MaleActors as a subclass of Thing
                pass
            class Genre(Thing): # Specify Genres as a subclass of Thing
                pass
            
            rule_genres = Imp()
            rule_genres.set_as_rule(genre + "(?g) -> Genre(?g)")
            
            rule_directors = Imp()
            rule_directors.set_as_rule(director + "(?d) -> Director(?d)")

            rule_all_movies = Imp()
            rule_all_movies.set_as_rule(movie + "(?m) -> Movie(?m)")

            rule_actors = Imp()
            rule_actors.set_as_rule(actor + "(?p) -> Actor(?p)")
        
        # Run the reasoner
        sync_reasoner_pellet(infer_property_values=True)

        # Retrieve inferred individuals
        self.Director = list(onto.Director.instances())
        self.Actor = list(onto.Actor.instances())
        self.Genre = list(onto.Genre.instances())
        self.Movie = list(onto.Movie.instances())


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(803, 616)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.treeView_all = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_all.setGeometry(QtCore.QRect(10, 90, 281, 431))
        self.treeView_all.setObjectName("treeView_all")
        self.getMovieBtn = QtWidgets.QPushButton(self.centralwidget)
        self.getMovieBtn.setGeometry(QtCore.QRect(690, 550, 75, 23))
        self.getMovieBtn.setObjectName("getMovieBtn")
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
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(320, 310, 471, 211))
        self.groupBox.setObjectName("groupBox")
        self.listView_exclude = QtWidgets.QListView(self.groupBox)
        self.listView_exclude.setGeometry(QtCore.QRect(10, 80, 441, 121))
        self.listView_exclude.setObjectName("listView_exclude")
        self.addBtn_exclude = QtWidgets.QPushButton(self.groupBox)
        self.addBtn_exclude.setGeometry(QtCore.QRect(260, 40, 75, 23))
        self.addBtn_exclude.setObjectName("addBtn_exclude")
        self.removeBtn_exclude = QtWidgets.QPushButton(self.groupBox)
        self.removeBtn_exclude.setGeometry(QtCore.QRect(370, 40, 75, 23))
        self.removeBtn_exclude.setObjectName("removeBtn_exclude")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(320, 100, 471, 191))
        self.groupBox_2.setObjectName("groupBox_2")
        self.listView_include = QtWidgets.QListView(self.groupBox_2)
        self.listView_include.setGeometry(QtCore.QRect(10, 60, 441, 121))
        self.listView_include.setObjectName("listView_include")
        self.addBtn_include = QtWidgets.QPushButton(self.groupBox_2)
        self.addBtn_include.setGeometry(QtCore.QRect(260, 20, 75, 23))
        self.addBtn_include.setObjectName("addBtn_include")
        self.removeBtn_include = QtWidgets.QPushButton(self.groupBox_2)
        self.removeBtn_include.setGeometry(QtCore.QRect(370, 20, 75, 23))
        self.removeBtn_include.setObjectName("removeBtn_include")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 10, 61, 61))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("logoMovieFinder.svg"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.groupBox_2.raise_()
        self.groupBox.raise_()
        self.treeView_all.raise_()
        self.getMovieBtn.raise_()
        self.label.raise_()
        self.line.raise_()
        self.line_2.raise_()
        self.label_2.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 803, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.movieFinderInit()
        self.treeInit()
        print("Tree initialized")

        # Initialize QStandardItemModel for listView_include and listView_exclude
        self.model_include = QtGui.QStandardItemModel(self.listView_include)
        self.listView_include.setModel(self.model_include)
        self.model_exclude = QtGui.QStandardItemModel(self.listView_exclude)
        self.listView_exclude.setModel(self.model_exclude)

        self.setOnclicks()
        print("Onclicks set")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    
    def movieFinderInit(self):
        self.movieFinder = MovieFinder()
        print("Movie Finder initialized")
    
    def setOnclicks(self):
        #self.getMovieBtn.clicked.connect(self.getMovie)
        self.addBtn_exclude.clicked.connect(self.addExclude)
        self.removeBtn_exclude.clicked.connect(self.removeExclude)
        self.addBtn_include.clicked.connect(self.addInclude)
        self.removeBtn_include.clicked.connect(self.removeInclude)

    def getMovie(self):
        included_actors = []
        excluded_actors = []
        included_directors = []
        excluded_directors = []
        included_genres = []
        excluded_genres = []

        for i in range(self.model_include.rowCount()):
            item = self.model_include.item(i).text()
            if item in self.lists_dict["Director"]:
                item = item.replace(" ", "_")
                included_directors.append(item)
            elif item in self.lists_dict["Actor"]:
                item = item.replace(" ", "_")
                included_actors.append(item)
            elif item in self.lists_dict["Genre"]:
                included_genres.append(item)
        for i in range(self.model_exclude.rowCount()):
            item = self.model_exclude.item(i).text()
            if item in self.lists_dict["Director"]:
                item = item.replace(" ", "_")
                excluded_directors.append(item)
            elif item in self.lists_dict["Actor"]:
                item = item.replace(" ", "_")
                excluded_actors.append(item)
            elif item in self.lists_dict["Genre"]:
                excluded_genres.append(item)
        # message = QtWidgets.QMessageBox()
        # message.setWindowTitle("Results")
        file_path = "data/Movies.rdf"

        results = get_movies(file_path, included_actors=included_actors, excluded_actors=excluded_actors,
                    included_directors=included_directors, excluded_directors=excluded_directors,
                    included_genres=included_genres, excluded_genres=excluded_genres)
        movies = {}
        for movie in results:
            movies[movie[0].value] = get_movie_details(movie[0].value)
        # results_str = ""
        # for movie in results:
        #     results_str += movie[0].value +", "
        #     get_movie_details(movie[0].value)
        #     print("="*20)
        # message.setText("Movies found: "+results_str[:-2]+".")
        # message.exec_()
        
        return movies
            

    def addExclude(self):
        #get the selected item in the tree and add to listView_exclude
        item = QtGui.QStandardItem(self.treeView_all.selectedIndexes()[0].data())
        # Check if item already exists in listView_include
        for i in range(self.model_include.rowCount()):
            if self.model_include.item(i).text() == item.text():
                self.model_include.removeRow(i)  # Remove item from include list
                break
        # Check if item already exists in listView_exclude
        for i in range(self.model_exclude.rowCount()):
            if self.model_exclude.item(i).text() == item.text():
                return  # Item already exists, do not add
        self.model_exclude.appendRow(item)
    
    def removeExclude(self):
        #remove the selected item in listView_exclude
        self.model_exclude.removeRow(self.listView_exclude.currentIndex().row())
    
    def addInclude(self):
        #get the selected item in the tree and add to listView_include
        item = QtGui.QStandardItem(self.treeView_all.selectedIndexes()[0].data())
        # Check if item already exists in listView_exclude
        for i in range(self.model_exclude.rowCount()):
            if self.model_exclude.item(i).text() == item.text():
                self.model_exclude.removeRow(i)  # Remove item from exclude list
                break
        # Check if item already exists in listView_include
        for i in range(self.model_include.rowCount()):
            if self.model_include.item(i).text() == item.text():
                return  # Item already exists, do not add
        self.model_include.appendRow(item)
    
    def removeInclude(self):
        #remove the selected item in listView_include
        self.model_include.removeRow(self.listView_include.currentIndex().row())

    def treeInit(self):
        self.treeView_all.setHeaderHidden(True)
        #self.treeView_all.setAlternatingRowColors(True)
        self.treeView_all.setSortingEnabled(True)
        self.treeView_all.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.treeView_all.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.treeView_all.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.model = QtGui.QStandardItemModel()
        self.treeView_all.setModel(self.model)

        lists_names = ["Director","Actor","Genre"]
        self.lists_dict = {}

        for name in lists_names:
            root_item = QtGui.QStandardItem(name)
            self.model.appendRow(root_item)
            items = getattr(self.movieFinder, name, [])
            self.lists_dict[name] = []
            
            for item in items:
                item_name = item.name
                #if there is _ in item.name, replace it with space
                if "_" in item.name and name != "Genre":
                    item_name = item.name.replace("_", " ")
                self.lists_dict[name].append(item_name)
                child_item = QtGui.QStandardItem(item_name)
                root_item.appendRow(child_item)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Movie Finder"))
        self.getMovieBtn.setText(_translate("MainWindow", "Get Movie"))
        self.label.setText(_translate("MainWindow", "The Movie Finder"))
        self.groupBox.setTitle(_translate("MainWindow", "Excluded Actors, Directors, and Genres:"))
        self.addBtn_exclude.setText(_translate("MainWindow", "Add"))
        self.removeBtn_exclude.setText(_translate("MainWindow", "Remove"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Included Actors, Directors, and Genres "))
        self.addBtn_include.setText(_translate("MainWindow", "Add"))
        self.removeBtn_include.setText(_translate("MainWindow", "Remove"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
