import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QListWidget, QTextEdit, QProgressBar, QHBoxLayout, QCheckBox, QComboBox, QMessageBox, QSlider, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from utils.pdf_utils import pdf_to_images
from ocr.llama_ocr import ocr_images
from ocr.postprocess import clean_text
import torch

OUTPUT_DIR = 'output'

class Worker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, list)  # add errors list

    def __init__(self, pdf_paths, options, preprocess_opts):
        super().__init__()
        self.pdf_paths = pdf_paths
        self.options = options
        self.preprocess_opts = preprocess_opts

    def run(self):
        results = []
        errors = []
        total = len(self.pdf_paths)
        for idx, pdf_path in enumerate(self.pdf_paths):
            try:
                images = pdf_to_images(pdf_path, dpi=self.options.get('dpi', 300), preprocess_opts=self.preprocess_opts)
                if not images:
                    raise Exception('No images extracted from PDF')
                raw_text = ocr_images(
                    images,
                    model=self.options.get('ocr_model', 'easyocr'),
                    gpu=self.options.get('gpu', False),
                    preserve_layout=self.options.get('preserve_layout', False)
                )
                cleaned = clean_text(
                    raw_text,
                    fix_hyphens=self.options.get('fix_hyphens', True),
                    merge_paragraphs=self.options.get('merge_paragraphs', True),
                    preserve_layout=self.options.get('preserve_layout', False),
                    spellcheck=self.options.get('spellcheck', False),
                    column_detection=self.options.get('column_detection', False),
                    ai_correction=self.options.get('ai_correction', False),
                    ollama_ai=self.options.get('ollama_ai', False),
                    ollama_model=self.options.get('ollama_model', 'llama2')
                )
                results.append((pdf_path, cleaned))
            except Exception as e:
                errors.append((pdf_path, str(e)))
            self.progress.emit(int((idx+1)/total*100))
        self.finished.emit(results, errors)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PDFTextGenie')
        self.setGeometry(100, 100, 800, 600)
        self.pdf_list = []
        self.settings = QSettings('PDFTextGenie', 'App')
        self.output_dir = self.settings.value('output_dir', OUTPUT_DIR)
        self.last_dpi = self.settings.value('dpi', '300')
        self.last_fix_hyphens = self.settings.value('fix_hyphens', 'true') == 'true'
        self.last_merge_paragraphs = self.settings.value('merge_paragraphs', 'true') == 'true'
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_widget.setAcceptDrops(True)
        self.list_widget.dragEnterEvent = self.dragEnterEvent
        self.list_widget.dropEvent = self.dropEvent
        layout.addWidget(QLabel('Selected PDFs:'))
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton('Add PDFs')
        self.add_btn.clicked.connect(self.add_pdfs)
        btn_layout.addWidget(self.add_btn)
        self.remove_btn = QPushButton('Remove Selected')
        self.remove_btn.clicked.connect(self.remove_selected)
        btn_layout.addWidget(self.remove_btn)
        self.clear_btn = QPushButton('Clear List')
        self.clear_btn.clicked.connect(self.clear_list)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

        self.output_btn = QPushButton('Choose Output Folder')
        self.output_btn.clicked.connect(self.choose_output)
        layout.addWidget(self.output_btn)
        self.output_label = QLabel(f'Output: {self.output_dir}')
        layout.addWidget(self.output_label)

        # Options
        options_layout = QHBoxLayout()
        self.ocr_model_label = QLabel('OCR Model:')
        options_layout.addWidget(self.ocr_model_label)
        self.ocr_model_combo = QComboBox()
        self.ocr_model_combo.addItems(['EasyOCR', 'TrOCR'])
        self.ocr_model_combo.setCurrentText('EasyOCR')
        options_layout.addWidget(self.ocr_model_combo)
        self.hyphen_cb = QCheckBox('Fix hyphens')
        self.hyphen_cb.setChecked(self.last_fix_hyphens)
        self.hyphen_cb.stateChanged.connect(self.save_settings)
        options_layout.addWidget(self.hyphen_cb)
        self.paragraph_cb = QCheckBox('Merge paragraphs')
        self.paragraph_cb.setChecked(self.last_merge_paragraphs)
        self.paragraph_cb.stateChanged.connect(self.save_settings)
        options_layout.addWidget(self.paragraph_cb)
        self.preserve_layout_cb = QCheckBox('Preserve Layout')
        self.preserve_layout_cb.setChecked(False)  # Default: do not preserve structure
        options_layout.addWidget(self.preserve_layout_cb)
        self.use_gpu_cb = QCheckBox('Use GPU')
        self.use_gpu_cb.setChecked(torch.cuda.is_available())
        options_layout.addWidget(self.use_gpu_cb)
        self.spellcheck_cb = QCheckBox('Spellcheck/Grammar Correction')
        self.spellcheck_cb.setToolTip('Run spellcheck and grammar correction on the output text.')
        options_layout.addWidget(self.spellcheck_cb)
        self.column_cb = QCheckBox('Column Detection')
        self.column_cb.setToolTip('Try to detect and preserve columns in the PDF.')
        options_layout.addWidget(self.column_cb)
        self.ai_correction_cb = QCheckBox('AI Correction')
        self.ai_correction_cb.setToolTip('Use an AI model to correct and enhance the output text.')
        options_layout.addWidget(self.ai_correction_cb)
        self.ollama_ai_cb = QCheckBox('Ollama AI Correction')
        self.ollama_ai_cb.setToolTip('Use a local Llama/Mistral/Mixtral model via Ollama for advanced formatting.')
        options_layout.addWidget(self.ollama_ai_cb)
        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.addItems(['llama3', 'llama2', 'mistral', 'mixtral'])
        self.ollama_model_combo.setCurrentText('llama3')
        self.ollama_model_combo.setEnabled(False)
        options_layout.addWidget(self.ollama_model_combo)
        self.ollama_ai_cb.stateChanged.connect(lambda s: self.ollama_model_combo.setEnabled(self.ollama_ai_cb.isChecked()))
        self.dpi_label = QLabel('DPI:')
        options_layout.addWidget(self.dpi_label)
        self.dpi_combo = QComboBox()
        self.dpi_combo.addItems(['200', '300', '400'])
        self.dpi_combo.setCurrentText(self.last_dpi)
        self.dpi_combo.currentTextChanged.connect(self.save_settings)
        options_layout.addWidget(self.dpi_combo)
        layout.addLayout(options_layout)

        # Image Preprocessing Options
        pre_layout = QHBoxLayout()
        self.binarize_cb = QCheckBox('Binarize')
        pre_layout.addWidget(self.binarize_cb)
        self.denoise_cb = QCheckBox('Denoise')
        pre_layout.addWidget(self.denoise_cb)
        self.contrast_cb = QCheckBox('Enhance Contrast')
        pre_layout.addWidget(self.contrast_cb)
        self.contrast_spin = QSpinBox()
        self.contrast_spin.setRange(1, 5)
        self.contrast_spin.setValue(2)
        self.contrast_spin.setPrefix('x')
        self.contrast_spin.setEnabled(False)
        pre_layout.addWidget(QLabel('Contrast Factor:'))
        pre_layout.addWidget(self.contrast_spin)
        self.contrast_cb.stateChanged.connect(lambda s: self.contrast_spin.setEnabled(self.contrast_cb.isChecked()))
        layout.addLayout(pre_layout)

        self.convert_btn = QPushButton('Start Conversion')
        self.convert_btn.clicked.connect(self.start_conversion)
        layout.addWidget(self.convert_btn)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        layout.addWidget(QLabel('Preview:'))
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        layout.addWidget(self.preview)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Menu bar for Help/About
        menubar = self.menuBar()
        help_menu = menubar.addMenu('Help')
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)

    def add_pdfs(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Select PDF files', '', 'PDF Files (*.pdf)')
        for f in files:
            if f not in self.pdf_list:
                self.pdf_list.append(f)
                self.list_widget.addItem(f)

    def remove_selected(self):
        # Remove items in reverse order to avoid index shifting
        items = [(self.list_widget.row(item), item) for item in self.list_widget.selectedItems()]
        for idx, item in sorted(items, reverse=True):
            self.list_widget.takeItem(idx)
            del self.pdf_list[idx]
    def clear_list(self):
        self.list_widget.clear()
        self.pdf_list = []

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder', self.output_dir)
        if folder:
            self.output_dir = folder
            self.output_label.setText(f'Output: {self.output_dir}')
            self.save_settings()

    def save_settings(self):
        self.settings.setValue('output_dir', self.output_dir)
        self.settings.setValue('dpi', self.dpi_combo.currentText())
        self.settings.setValue('fix_hyphens', 'true' if self.hyphen_cb.isChecked() else 'false')
        self.settings.setValue('merge_paragraphs', 'true' if self.paragraph_cb.isChecked() else 'false')

    def load_settings(self):
        self.output_label.setText(f'Output: {self.output_dir}')
        self.dpi_combo.setCurrentText(self.last_dpi)
        self.hyphen_cb.setChecked(self.last_fix_hyphens)
        self.paragraph_cb.setChecked(self.last_merge_paragraphs)
        self.use_gpu_cb.setChecked(torch.cuda.is_available())

    def show_about(self):
        QMessageBox.information(self, 'About PDFTextGenie',
            'PDFTextGenie\n\nSmart GUI PDF-to-Text Converter\n\nBatch convert scanned/image PDFs to clean text using OCR.\n\nDeveloped with PyQt5.\n\nhttps://github.com/UTSAVS26/PyVerse')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith('.pdf') and path not in self.pdf_list:
                self.pdf_list.append(path)
                self.list_widget.addItem(path)

    def start_conversion(self):
        if not self.pdf_list:
            self.statusBar().showMessage('No PDFs selected.')
            return
        os.makedirs(self.output_dir, exist_ok=True)
        ocr_model = self.ocr_model_combo.currentText().lower()
        use_gpu = self.use_gpu_cb.isChecked()
        options = {
            'fix_hyphens': self.hyphen_cb.isChecked(),
            'merge_paragraphs': self.paragraph_cb.isChecked(),
            'preserve_layout': self.preserve_layout_cb.isChecked(),
            'spellcheck': self.spellcheck_cb.isChecked(),
            'column_detection': self.column_cb.isChecked(),
            'ai_correction': self.ai_correction_cb.isChecked(),
            'ollama_ai': self.ollama_ai_cb.isChecked(),
            'ollama_model': self.ollama_model_combo.currentText(),
            'dpi': int(self.dpi_combo.currentText()),
            'ocr_model': ocr_model,
            'gpu': use_gpu,
        }
        preprocess_opts = {
            'binarize': self.binarize_cb.isChecked(),
            'denoise': self.denoise_cb.isChecked(),
            'contrast': self.contrast_cb.isChecked(),
            'contrast_factor': float(self.contrast_spin.value()) if self.contrast_cb.isChecked() else 1.0,
        }
        self.worker = Worker(self.pdf_list, options, preprocess_opts)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.handle_results)
        self.progress.setValue(0)
        self.worker.start()
        self.statusBar().showMessage('Converting...')

    def handle_results(self, results, errors):
        preview_text = ''
        for pdf_path, text in results:
            base = os.path.splitext(os.path.basename(pdf_path))[0]
            pdf_dir = os.path.dirname(pdf_path)
            # Re-run clean_text with filename for prompt header
            from ocr.postprocess import clean_text as clean_text_with_filename
            text = clean_text_with_filename(
                text,
                fix_hyphens=self.hyphen_cb.isChecked(),
                merge_paragraphs=self.paragraph_cb.isChecked(),
                preserve_layout=self.preserve_layout_cb.isChecked(),
                spellcheck=self.spellcheck_cb.isChecked(),
                column_detection=self.column_cb.isChecked(),
                ai_correction=self.ai_correction_cb.isChecked(),
                ollama_ai=self.ollama_ai_cb.isChecked(),
                ollama_model=self.ollama_model_combo.currentText(),
                filename=base
            )
            out_path = os.path.join(pdf_dir, base + '.txt')
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(text)
            preview_text += f'--- {base}.txt ---\n{text}\n\n'
        if errors:
            error_msg = '\n'.join([f"{os.path.basename(p)}: {msg}" for p, msg in errors])
            preview_text += f'\n--- ERRORS ---\n{error_msg}\n'
            QMessageBox.critical(self, 'Conversion Errors', f'Some PDFs failed to convert:\n\n{error_msg}')
            self.statusBar().showMessage(f'Done with errors: {len(errors)} failed.')
        else:
            self.statusBar().showMessage('Done!')
        self.preview.setPlainText(preview_text)
        self.progress.setValue(100)

def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 