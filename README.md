# Chart Classifier

![Chart Classifier](https://img.shields.io/badge/status-ready-brightgreen)

## Overview

**Chart Classifier** is a Python application that predicts the type of chart from an input image.  
It uses a pre-trained Vision Transformer (ViT) model to classify chart images offline and outputs the predicted class along with a confidence score.  
This project provides an easy way to classify charts using Python and supports multiple chart types.

## Features

- Classifies chart images (bar chart, line chart, pie chart, etc.)
- Offline inference using a pre-trained ViT model
- Outputs results in JSON format
- Easy-to-use CLI command `chart <image_path>` after setup
- Works on Windows, Linux, and macOS

## Setup & Usage

Follow these steps to set up and run the application:

1. **Clone the repository**:

```bash
git clone <repository_url>
cd <repository_folder>

pip install -r requirements.txt

3. **Run the setup script**  

This will:  
- Create `config.json` storing the model path  
- Download the pre-trained ViT model for offline use  
- Create a global CLI command `chart` to run predictions  

```bash
python set_up.py


4. **Predict chart type**

- **Using the CLI command**:

```bash
chart /path/to/chart_image.png


{
    "predicted_class": "bar",
    "confidence_score": 0.987
}
