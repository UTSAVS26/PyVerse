import sys

def run_dashboard(mode='qt'):
    if mode == 'qt':
        try:
            from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel
        except ImportError:
            print("PyQt5 is not installed. Please install it with 'pip install pyqt5'.")
            sys.exit(1)

        class DashboardWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("Dynamic Hardware Resource Monitor")
                self.setGeometry(100, 100, 1000, 700)
                self.tabs = QTabWidget()
                self.setCentralWidget(self.tabs)
                self.init_tabs()

            def init_tabs(self):
                self.tabs.addTab(self._make_tab("CPU Usage"), "CPU")
                self.tabs.addTab(self._make_tab("Memory Usage"), "Memory")
                self.tabs.addTab(self._make_tab("GPU Usage"), "GPU")
                self.tabs.addTab(self._make_tab("Disk I/O"), "Disk")
                self.tabs.addTab(self._make_tab("Network I/O"), "Network")
                self.tabs.addTab(self._make_tab("Prediction"), "Prediction")

            def _make_tab(self, label_text):
                tab = QWidget()
                layout = QVBoxLayout()
                label = QLabel(label_text)
                layout.addWidget(label)
                tab.setLayout(layout)
                return tab

        app = QApplication(sys.argv)
        window = DashboardWindow()
        window.show()
        sys.exit(app.exec_())
    elif mode == 'web':
        try:
            import streamlit as st
        except ImportError:
            print("Streamlit is not installed. Please install it with 'pip install streamlit'.")
            sys.exit(1)
        st.title("Dynamic Hardware Resource Monitor")
        st.write("[Streamlit dashboard coming soon]")
    else:
        print(f"Unknown mode: {mode}")
