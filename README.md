# Deep Learning with Python Exercises

This repository contains implementations of exercises from the book "Deep Learning with Python" by François Chollet.

## Setup

1. Ensure Python 3.8+ is installed.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `source .venv/bin/activate` (on macOS/Linux)
4. Install dependencies: `pip install -r requirements.txt`

## Structure

- `chapter2/`: Getting started with neural networks
- `chapter3/`: Introduction to Keras and TensorFlow
- `chapter5/`: Deep learning for computer vision
- `chapter6/`: Deep learning for text and sequences
- `chapter7/`: Advanced deep-learning best practices
- `chapter8/`: Generative deep learning

## Running the Code

Each chapter has Python files implementing the tasks. Run them with `python chapterX/task.py`.

Note: Some tasks require downloading datasets (e.g., MNIST, IMDB) which TensorFlow/Keras handles automatically. For others like Dogs vs. Cats or Jena weather, update paths to local files.

## Requirements

- tensorflow
- numpy
- matplotlib
- scikit-learn