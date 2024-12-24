# Grade Auto-Filler Setup Guide

## Overview
The `grade-auto-filler` application automates the process of grading using machine learning and image processing techniques. This guide will help you set up the necessary environment to run the application seamlessly.

---

## Prerequisites

### System Requirements
- **Python Version**: Python 3.11
- **Operating System**: Compatible with Windows, macOS, or Linux

### Tools
- Ensure you have `pip`, Python’s package manager, installed.

---

## Installation Instructions

### Step 1: Clone or Download the Repository
```bash
# Clone the repository
git clone https://github.com/3mr-mahmoud/grades-auto-filler.git
cd grade-auto-filler
```

### Step 2: Install Python Dependencies

1. Create a `requirements.txt` file and copy the following content into it:
   ```plaintext
   streamlit
   pandas
   openpyxl
   opencv-python
   opencv-contrib-python
   numpy
   matplotlib
   pytesseract
   scikit-learn
   joblib
   ```

2. Install all dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Additional Setup for `pytesseract`
`pytesseract` requires Tesseract OCR to be installed on your system:

- **Windows**:
  - Download and install Tesseract from [here](https://github.com/UB-Mannheim/tesseract/wiki).
  - Add the installation path (e.g., `C:\Program Files\Tesseract-OCR`) to your system's PATH environment variable.

- **Linux**:
  ```bash
  sudo apt-get install tesseract-ocr
  ```

- **macOS**:
  ```bash
  brew install tesseract
  ```

- Verify the installation by running:
  ```bash
  tesseract --version
  ```

### Step 4: Verify All Dependencies
Run the following command to check if all dependencies are installed correctly:
```bash
python -m pip freeze
```

---

## Running the Application

1. Launch the application:
   ```bash
   cd ./src
   streamlit run app.py
   ```

2. Ensure that all necessary files (such as input data, models, or configurations) are in the working directory.

---

## Troubleshooting
- If you encounter any issues, ensure:
  1. All libraries are correctly installed via `pip`.
  2. Tesseract OCR is installed and accessible in your environment.
  3. Python version is 3.11.

- For further assistance, refer to the documentation or open an issue in the repository.

---

## License
This project is distributed under the [MIT License](LICENSE).

---

## Contact
For any questions or issues, please reach out to the project maintainer.

