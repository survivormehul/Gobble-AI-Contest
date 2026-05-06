# Gobble AI Contest Submissions

This repository contains my submissions for the Gobble AI challenges.

## Challenges

### 1. Pedestrian Crossing Intent Challenge
Located in the `Crossing Challenge/` directory.

The objective of this challenge is to predict pedestrian crossing intent and future trajectory. 
This solution implements a machine learning system to predict pedestrian crossing behavior, optimizing for both accuracy and low-latency inference within a restricted Docker container size.

**Files included:**
- `predict.py`: Inference script
- `train_advanced.py`: Advanced training script
- `model.pkl`: Serialized trained model
- `grade.py`: Evaluation script
- `Dockerfile`: Container definition for reproducible evaluation
- `requirements.txt`: Python dependencies

### 2. Ride-Hailing ETA Predictor
Located in the `ETA Challenge/my-folder/` directory.

The objective of this challenge is to build a high-performance machine learning system to predict the Estimated Time of Arrival (ETA) for ride-hailing trips. 
The solution utilizes an advanced XGBoost model with target encoding and categorical features, optimizing for strict latency constraints and low prediction error (MAE).

**Files included:**
- `predict.py`: Inference script
- `train_advanced.py`: Model training script
- `model.pkl`: Serialized trained model
- `grade.py`: Evaluation script
- `Dockerfile`: Container definition for reproducible evaluation
- `requirements.txt`: Python dependencies

## Setup and Evaluation
Each challenge directory contains its own `Readme.md` and `Dockerfile` detailing the specific setup, requirements, and instructions to build the image and evaluate the model using `grade.py`.

Please refer to the respective directories for more detailed instructions on running inference or re-training the models.

## Contact
- **LinkedIn:** [Mehul](https://www.linkedin.com/in/survivor-mehul/)