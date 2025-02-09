import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLineEdit, QLabel,
    QMessageBox, QGroupBox, QGridLayout
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from KMeans import K_Means


class MplCanvas(FigureCanvas):
    """Canvas Matplotlib do rysowania wykresów."""
    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class KMeansApp(QMainWindow):
    """Główna klasa aplikacji."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("K-Means Visualizer")
        self.resize(800, 600)

        self.canvas = MplCanvas(self)
        self.canvas2 = MplCanvas(self)
        self.file_path = None
        self.kmeans = None

        self.init_ui()

    def init_ui(self):
        """Tworzenie interfejsu użytkownika."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        box = QGroupBox()
        box_layout = QVBoxLayout()
        box.setLayout(box_layout)

        inside_layout = QGridLayout()

        self.file_label = QLabel("File not selected")
        self.file_label.setAlignment(Qt.AlignCenter)
        file_button = QPushButton("Select file")
        file_button.clicked.connect(self.select_file)
        file_button.setFixedWidth(200)
        inside_layout.addWidget(self.file_label,0,0,1,1)
        inside_layout.addWidget(file_button, 1,0,1,1)

        cluster_label = QLabel("Number of clusters:")
        cluster_label.setAlignment(Qt.AlignCenter)
        self.cluster_input = QLineEdit("3")
        self.cluster_input.setFixedWidth(200)
        inside_layout.addWidget(cluster_label, 0,1,1,1)
        inside_layout.addWidget(self.cluster_input,1,1,1,1)

        iteration_label = QLabel("Max number of iteration (min.10):")
        iteration_label.setAlignment(Qt.AlignCenter)
        self.iteration_input = QLineEdit("25")
        self.iteration_input.setFixedWidth(200)
        inside_layout.addWidget(iteration_label, 0, 2, 1, 1)
        inside_layout.addWidget(self.iteration_input, 1, 2, 1, 1)

        run_button = QPushButton("Run K-Means")
        run_button.clicked.connect(self.run_kmeans)
        inside_layout.addWidget(run_button,2,0,1,3)

        box_layout.addLayout(inside_layout)

        main_layout.addWidget(box)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.canvas2)



    def select_file(self):
        """Wybór pliku."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select file with data.")
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"Selected file: {file_path.split('/')[-1]}")

    def run_kmeans(self):
        """Uruchom algorytm K-Means i wizualizuj wyniki."""
        if not self.file_path:
            QMessageBox.critical(self, "ERROR", "Select file!")
            return

        try:
            cluster_num = int(self.cluster_input.text())
            if cluster_num <= 0:
                QMessageBox.critical(self, "ERROR", "Specify the number of clusters greater than 0!")
                return
        except ValueError:
            QMessageBox.critical(self, "ERROR", "Specify the number of clusters!")
            return

        try:
            self.iteration_num = int(self.iteration_input.text())
            if self.iteration_num < 2:
                QMessageBox.critical(self, "ERROR", "Specify the number of iteration greater or equal 10!")
                return
        except ValueError:
            QMessageBox.critical(self, "ERROR", "Specify maximum number of iteration!")
            return

        # Uruchomienie algorytmu
        self.kmeans = K_Means(cluster_num)
        self.kmeans.data_download(self.file_path)
        self.kmeans.data_validation()
        self.kmeans.data_normalization()
        self.kmeans.centroids_draw()
        self.kmeans.finding_clusters()
        self.kmeans.visualisation_original_data(self.canvas2.axes)
        self.canvas2.draw()
        self.iter = 0

        # Tworzenie animacji
        self.anim = FuncAnimation(self.canvas.figure, self.animate, frames=self.iteration_num, interval=1000, repeat=False)
        self.canvas.draw()

    def animate(self, i):
        """Aktualizacja dla każdej klatki animacji."""
        if i > 0:
            self.kmeans.finding_clusters()
            self.kmeans.finding_new_centroids()
            self.kmeans.converged()
        self.kmeans.visualisation(self.canvas.axes)

        if self.kmeans.conv:
            self.anim.pause()
            QMessageBox.information(self,"Finish", "Centroid converged!")

        if self.iteration_num - self.iter == 0:
            self.anim.pause()
            QMessageBox.information(self, "Finish", "Reached maximum iteration!")
        self.iter += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KMeansApp()
    window.show()
    sys.exit(app.exec())
