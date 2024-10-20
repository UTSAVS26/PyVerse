# Interactive Python Playground

This is an interactive web-based Python playground built using **HTML**, **JavaScript**, and **CodeMirror**. It allows users to write, execute, and save Python code directly in the browser. The interface includes a code editor, buttons for running and saving code, and a panel to display the output. Additionally, it features a list of saved code snippets for easy retrieval.

## Features

- **Code Editor**: Powered by CodeMirror, the editor supports syntax highlighting and line numbering for Python code.
- **Run Python Code**: Execute your Python code directly in the browser and see the results in real-time.
- **Save & Load Code Snippets**: Save your favorite code snippets and reload them anytime for quick access.
- **Output Panel**: Displays the output or errors of executed Python code.
- **Image Display**: Includes two images aligned with the editor for a visually enhanced layout.


## Getting Started

### Prerequisites

- A **web server** or **local development environment** for running HTML, such as:
  - Python's built-in HTTP server (`python -m http.server`)
  - Apache or Nginx
- **Node.js** (for backend setup if using server-side Python execution)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/shivi13102/interactive-python-playground.git
    cd interactive-python-playground
    ```

2. Set up a local HTTP server (e.g., using Python):
    ```bash
    python -m http.server 8000
    ```

3. Open your web browser and go to:
    ```
    http://localhost:8000
    ```

4. Start writing and running your Python code!


### Configuration

- **Editor Settings**: The CodeMirror editor can be customized for different themes, modes, and line number display.
- **Images**: To change the images in the interface, replace the `src` attribute in the `img` tags with the URLs of your desired images.

## Usage

1. **Write Python Code**: Use the editor to write Python code. The editor supports syntax highlighting, making it easier to read and debug.
2. **Run Code**: Click the `Run Code` button to execute your Python script. The output will be displayed in the output panel below.
3. **Save Snippets**: Click `Save Code` to store your current script. The snippet will be available in the snippets list.
4. **Load Snippets**: Click `Load Saved Snippets` to retrieve previously saved code snippets and select any snippet to load it into the editor.

## Technologies Used

- **HTML**: For structure and layout.
- **CSS**: For styling the playground.
- **JavaScript**: For handling user interactions and making HTTP requests.
- **CodeMirror**: A versatile text editor implemented in JavaScript for the code editor.
- **Python**: For executing code snippets (via a backend server).

## Customization

- **Editor Customization**: Modify CodeMirror settings to change the appearance and functionality of the code editor.
- **Output Panel Styling**: Adjust CSS for the `#output` element to change colors, font size, and other display properties.
- **Add More Buttons**: You can add additional buttons (e.g., `Clear Output`, `Reset Editor`) by extending the `button-container` in the HTML.


## Acknowledgements

- **[CodeMirror](https://codemirror.net/)** for the code editor.
- **Flask** for providing a simple server-side Python execution environment.
- Icons made by [author](https://www.flaticon.com/authors/author) from [www.flaticon.com](https://www.flaticon.com/).

## Contact

For any inquiries or suggestions, please contact:

- **Name**: CH Shivangi
- **Email**: chshivangi1@gmail.com
