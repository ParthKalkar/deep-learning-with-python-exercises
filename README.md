# Deep Learning with Python Exercises

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/ParthKalkar/deep-learning-with-python-exercises)

This repository contains complete Python implementations of all exercises from the book **"Deep Learning with Python"** by François Chollet. It's designed for students and practitioners to learn deep learning hands-on using TensorFlow/Keras.

## 📚 What's Included

- **Complete Code**: Python scripts for every exercise, organized by chapter.
- **Task Descriptions**: Detailed explanations of each task in `results.md`.
- **Sample Outputs**: Example results from running the code.
- **Setup Guide**: Easy installation and environment setup.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ParthKalkar/deep-learning-with-python-exercises.git
   cd deep-learning-with-python-exercises
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
deep-learning-with-python-exercises/
├── chapter2/          # Getting started with neural networks
│   ├── mnist.py       # MNIST classification
│   ├── naive_ops.py   # Vector operations
│   └── display_digit.py
├── chapter3/          # Introduction to Keras and TensorFlow
│   ├── imdb.py        # IMDB sentiment analysis
│   ├── boston.py      # Boston housing regression
│   └── reuters.py     # Reuters news classification
├── chapter5/          # Deep learning for computer vision
│   ├── convnet_mnist.py
│   ├── dogs_vs_cats.py
│   └── feature_extraction.py
├── chapter6/          # Deep learning for text and sequences
│   ├── rnn.py
│   ├── embedding.py
│   └── conv_rnn.py
├── chapter7/          # Advanced deep-learning best practices
│   ├── functional_api.py
│   └── callbacks.py
├── chapter8/          # Generative deep learning
│   ├── text_generation.py
│   ├── vae.py
│   └── gan.py
├── results.md         # Task descriptions and sample outputs
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── .gitignore
```

## 🏃 Running the Exercises

Each chapter contains Python files for the exercises. Run them like this:

```bash
python chapter2/mnist.py
```

**Notes**:
- Datasets (MNIST, IMDB, etc.) are downloaded automatically by TensorFlow/Keras.
- For exercises requiring external data (e.g., Dogs vs. Cats), update file paths in the code.
- On macOS with Apple Silicon, add MPS support by including this at the top of scripts:
  ```python
  import os
  os.environ['TF_ENABLE_MPS'] = '1'
  import tensorflow as tf
  ```

## 📊 Results

Check `results.md` for:
- Detailed task descriptions
- Sample code outputs
- Performance metrics

## 🤝 Contributing

Contributions are welcome! Please:
- Fork the repository
- Create a feature branch
- Submit a pull request

## 📄 License

This project is for educational purposes. Please refer to the original book "Deep Learning with Python" by François Chollet for licensing information.

## 📖 Book Reference

Chollet, François. *Deep Learning with Python*. Manning Publications, 2018.

---

**Happy Learning!** If you find this helpful, give the repo a ⭐ on GitHub.
