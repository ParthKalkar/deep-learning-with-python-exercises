# Deep Learning with Python Exercises

This repository contains implementations of exercises from the book "Deep Learning with Python" by François Chollet.

## Setup

1. Ensure Python 3.8+ is installed.
2. Clone this repository: `git clone <your-repo-url>`
3. Navigate to the directory: `cd deep_learning_with_python`
4. Create a virtual environment: `python -m venv .venv`
5. Activate it: `source .venv/bin/activate` (on macOS/Linux) or `.venv\Scripts\activate` (on Windows)
6. Install dependencies: `pip install -r requirements.txt`

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

For MPS optimization on macOS (Apple Silicon), add this at the top of scripts:

```python
import os
os.environ['TF_ENABLE_MPS'] = '1'
import tensorflow as tf
```

## Results

See `results.md` for sample outputs from running key tasks.

## Contributing

Feel free to submit issues or pull requests for improvements.

## License

This project is for educational purposes. Refer to the book for licensing.# Chapter 2 completed
# Chapter 3 completed
# Chapter 5 completed
# Final updates
