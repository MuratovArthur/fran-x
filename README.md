# FRaN-X Entity Recognition Web Interface

A Streamlit-based web interface for entity recognition using DeBERTa model.

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or 3.10 (recommended)
- pip or conda for package management

### Installation

1. Create and activate a virtual environment (recommended):
   ```bash
   # Using conda (recommended)
   conda create -n franx python=3.10
   conda activate franx

   # OR using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

   Note: This will automatically install all dependencies, including pytorch-crf from GitHub.

3. Download the model files:
   - Ensure you have the model files in `models/24_june_x1_large_mult/`
   - Required files:
     - config.json
     - model.safetensors
     - special_tokens_map.json
     - tokenizer.json
     - tokenizer_config.json
     - spm.model

### Running the App

1. Start the Streamlit app:
   ```bash
   streamlit run Home.py
   ```

2. Open your browser and navigate to:
   - Local: http://localhost:8501
   - Network: Check the terminal output for the network URL

## 🔧 Troubleshooting

If you encounter any issues:

1. Make sure you're using Python 3.9 or 3.10
2. Try reinstalling the requirements:
   ```bash
   pip uninstall -y pytorch-crf
   pip install -r requirements.txt
   ```
3. Verify that all model files are present in the correct directory

## 📦 Project Structure

```
FRaN-X_Streamlit_Website/
├── Home.py                 # Main Streamlit app
├── pages/                  # Additional pages
├── models/                 # Model files
├── src/                    # Core model code
└── utils/                  # Utility functions
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the terms of the LICENSE file included in the repository.
