"""
GUI applications for PyFlowViz.
"""

import os
import tempfile
from typing import Optional

# Optional imports for GUI dependencies
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QTextEdit, QLabel, 
                                 QFileDialog, QMessageBox, QComboBox)
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

from ..parser import ASTParser, BytecodeParser
from ..visualizer import GraphvizGenerator, HTMLRenderer


class GradioGUI:
    """Web-based GUI using Gradio."""
    
    def __init__(self):
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is not installed. Install it with: pip install gradio")
        
        self.ast_parser = ASTParser()
        self.bytecode_parser = BytecodeParser()
        self.graphviz_gen = GraphvizGenerator()
        self.html_renderer = HTMLRenderer()
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(title="PyFlowViz - Code to Flowchart Generator") as interface:
            gr.Markdown("# 🔁 PyFlowViz: Code-to-Flowchart Auto Generator")
            gr.Markdown("Instantly visualize Python logic with dynamic flowcharts.")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📝 Input")
                    
                    # File upload
                    file_input = gr.File(
                        label="Upload Python File",
                        file_types=[".py"],
                        type="filepath"
                    )
                    
                    # Text input
                    code_input = gr.Textbox(
                        label="Or paste Python code here:",
                        placeholder="def example():\n    if x > 0:\n        print('positive')\n    else:\n        print('negative')",
                        lines=10
                    )
                    
                    # Parser selection
                    parser_type = gr.Radio(
                        choices=["AST Parser", "Bytecode Parser"],
                        value="AST Parser",
                        label="Parser Type"
                    )
                    
                    # Output format
                    output_format = gr.Radio(
                        choices=["SVG", "PNG", "Interactive HTML", "Mermaid HTML"],
                        value="SVG",
                        label="Output Format"
                    )
                    
                    # Generate button
                    generate_btn = gr.Button("🚀 Generate Flowchart", variant="primary")
                
                with gr.Column():
                    gr.Markdown("## 📊 Output")
                    
                    # Output display
                    output_display = gr.HTML(label="Generated Flowchart")
                    
                    # Download link
                    download_link = gr.File(label="Download")
            
            # Event handlers
            generate_btn.click(
                fn=self._generate_flowchart,
                inputs=[file_input, code_input, parser_type, output_format],
                outputs=[output_display, download_link]
            )
        
        return interface
    
    def _generate_flowchart(self, file_path, code_text, parser_type, output_format):
        """Generate flowchart from input."""
        try:
            # Determine input source
            if file_path:
                source_code = self._read_file(file_path)
                source_name = os.path.basename(file_path)
            elif code_text.strip():
                source_code = code_text
                source_name = "input_code"
            else:
                return "<p style='color: red;'>Please provide either a file or code text.</p>", None
            
            # Parse code
            if parser_type == "AST Parser":
                parser = self.ast_parser
            else:
                parser = self.bytecode_parser
            
            flowchart_data = parser.parse_string(source_code)
            
            # Generate output
            if output_format == "SVG":
                output_path = self.graphviz_gen.generate_svg(flowchart_data)
                return f"<p>✅ Generated SVG flowchart: <a href='file://{output_path}' target='_blank'>{output_path}</a></p>", output_path
            
            elif output_format == "PNG":
                output_path = self.graphviz_gen.generate_png(flowchart_data)
                return f"<p>✅ Generated PNG flowchart: <a href='file://{output_path}' target='_blank'>{output_path}</a></p>", output_path
            
            elif output_format == "Interactive HTML":
                output_path = self.html_renderer.generate_html(flowchart_data)
                return f"<p>✅ Generated interactive HTML: <a href='file://{output_path}' target='_blank'>{output_path}</a></p>", output_path
            
            else:  # Mermaid HTML
                output_path = self.html_renderer.generate_mermaid_html(flowchart_data)
                return f"<p>✅ Generated Mermaid HTML: <a href='file://{output_path}' target='_blank'>{output_path}</a></p>", output_path
                
        except Exception as e:
            return f"<p style='color: red;'>❌ Error: {str(e)}</p>", None
    
    def _read_file(self, file_path):
        """Read file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def launch(self, share=False):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(share=share)


class PyQt5GUI:
    """Desktop GUI using PyQt5."""
    
    def __init__(self):
        if not PYQT5_AVAILABLE:
            raise ImportError("PyQt5 is not installed. Install it with: pip install PyQt5")
        
        self.ast_parser = ASTParser()
        self.bytecode_parser = BytecodeParser()
        self.graphviz_gen = GraphvizGenerator()
        self.html_renderer = HTMLRenderer()
        self.app = None
        self.window = None
    
    def create_window(self):
        """Create the main window."""
        self.window = QMainWindow()
        self.window.setWindowTitle("PyFlowViz - Code to Flowchart Generator")
        self.window.setGeometry(100, 100, 1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)
        
        # Main layout
        layout = QHBoxLayout(central_widget)
        
        # Left panel - Input
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Title
        title_label = QLabel("🔁 PyFlowViz: Code-to-Flowchart Generator")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(title_label)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        file_btn = QPushButton("Browse File")
        file_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(file_btn)
        left_layout.addLayout(file_layout)
        
        # Code input
        code_label = QLabel("Or paste Python code:")
        left_layout.addWidget(code_label)
        
        self.code_input = QTextEdit()
        self.code_input.setPlaceholderText("def example():\n    if x > 0:\n        print('positive')\n    else:\n        print('negative')")
        left_layout.addWidget(self.code_input)
        
        # Parser selection
        parser_layout = QHBoxLayout()
        parser_layout.addWidget(QLabel("Parser:"))
        self.parser_combo = QComboBox()
        self.parser_combo.addItems(["AST Parser", "Bytecode Parser"])
        parser_layout.addWidget(self.parser_combo)
        left_layout.addLayout(parser_layout)
        
        # Output format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["SVG", "PNG", "Interactive HTML", "Mermaid HTML"])
        format_layout.addWidget(self.format_combo)
        left_layout.addLayout(format_layout)
        
        # Generate button
        generate_btn = QPushButton("🚀 Generate Flowchart")
        generate_btn.clicked.connect(self._generate_flowchart)
        generate_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        left_layout.addWidget(generate_btn)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green; margin: 10px;")
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        layout.addWidget(left_panel, 1)
        
        # Right panel - Output
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        output_label = QLabel("Generated Flowchart:")
        right_layout.addWidget(output_label)
        
        self.output_display = QLabel()
        self.output_display.setAlignment(Qt.AlignCenter)
        self.output_display.setStyleSheet("border: 1px solid #ccc; background: white;")
        right_layout.addWidget(self.output_display)
        
        layout.addWidget(right_panel, 2)
    
    def _browse_file(self):
        """Browse for Python file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Select Python File",
            "",
            "Python Files (*.py);;All Files (*)"
        )
        
        if file_path:
            self.file_label.setText(os.path.basename(file_path))
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.code_input.setPlainText(f.read())
            except Exception as e:
                QMessageBox.warning(self.window, "Error", f"Failed to read file: {str(e)}")
    
    def _generate_flowchart(self):
        """Generate flowchart from input."""
        try:
            # Get input
            code_text = self.code_input.toPlainText().strip()
            if not code_text:
                QMessageBox.warning(self.window, "Error", "Please provide Python code.")
                return
            
            # Select parser
            if self.parser_combo.currentText() == "AST Parser":
                parser = self.ast_parser
            else:
                parser = self.bytecode_parser
            
            # Parse code
            flowchart_data = parser.parse_string(code_text)
            
            # Generate output
            output_format = self.format_combo.currentText()
            
            if output_format == "SVG":
                output_path = self.graphviz_gen.generate_svg(flowchart_data)
            elif output_format == "PNG":
                output_path = self.graphviz_gen.generate_png(flowchart_data)
            elif output_format == "Interactive HTML":
                output_path = self.html_renderer.generate_html(flowchart_data)
            else:  # Mermaid HTML
                output_path = self.html_renderer.generate_mermaid_html(flowchart_data)
            
            # Update status
            self.status_label.setText(f"✅ Generated: {os.path.basename(output_path)}")
            self.status_label.setStyleSheet("color: green; margin: 10px;")
            
            # Show preview if PNG
            if output_format == "PNG":
                pixmap = QPixmap(output_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        self.output_display.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.output_display.setPixmap(scaled_pixmap)
                else:
                    self.output_display.setText("Generated PNG file")
            else:
                self.output_display.setText(f"Generated {output_format} file:\n{output_path}")
            
            QMessageBox.information(
                self.window,
                "Success",
                f"Flowchart generated successfully!\nSaved to: {output_path}"
            )
            
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet("color: red; margin: 10px;")
            QMessageBox.critical(self.window, "Error", f"Failed to generate flowchart: {str(e)}")
    
    def launch(self):
        """Launch the PyQt5 GUI."""
        self.app = QApplication([])
        self.create_window()
        self.window.show()
        self.app.exec_() 