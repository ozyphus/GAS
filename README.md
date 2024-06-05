# Detectron2 Object Detection Toolkit

This repository contains a collection of Python scripts designed to facilitate the training, testing, and evaluation of object detection models using Detectron2, along with utilities for handling datasets. The scripts cover a range of functionalities from dataset conversion to performance logging.

## Contents

- `convertyolotococo.py`: Converts YOLO format datasets to COCO format to make them compatible with Detectron2.
- `GOSLAMDetectron_TensorBoard.py`: Extends Detectron2's capabilities to include TensorBoard for training visualization.
- `GOSLAMDetectron_Weights_Biases.py`: Integrates Detectron2 with Weights & Biases for advanced training metrics tracking and visualization.
- `mergedatasets.py`: Combines multiple datasets into a single dataset, facilitating broader training scenarios.
- `test_detectron.py`: Provides functionality to test a trained Detectron2 model on new data, outputting performance metrics and visualizations.
