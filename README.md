# Facial Expression Recognition Project

This project is being developed for the **Data Mining** course of the **Master’s in Data Science** program at the **University of Naples Federico II**.

It focuses on **facial expression recognition (FER)** using a pre-processed subset of the **AffectNet dataset**, specifically the **"Facial Expression Image Data AFFECTNET YOLO Format"** available on Kaggle. This dataset provides:
* Face-cropped images, uniformly resized to **96x96 pixels**.
* Images from 8 emotion categories: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
* Pre-defined train, validation, and test splits (70/20/10 ratio).
* Labels provided via accompanying `.txt` files in YOLO format, which will be parsed to retrieve emotion class IDs for each image.

The project aims to leverage deep learning techniques (using Keras/TensorFlow) to accurately classify these facial expressions.
