# Multi-Agent-LSTM
 paper'Scalable Anomaly Detection in IIoT Networks Using Multi-Agent LSTM Framework' code
## Overview
This project implements a **multi-agent system** based on Long Short-Term Memory (LSTM) neural networks. Each agent trains a separate LSTM model for binary classification tasks using data from different silos, mimicking a federated learning setup. The framework evaluates the performance of these agents and saves the results.

## Features
- Supports multiple silos, each with its own training and test datasets.
- LSTM-based binary classification for each silo.
- Early stopping mechanism to prevent overfitting.
- Comprehensive evaluation metrics: Accuracy, Precision, Recall, and F1-score.
- Results are saved to a text file for easy review.

## Requirements
- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Multi-Agent-LSTM
