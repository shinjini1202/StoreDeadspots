# Store Deadspot Detection

## Overview
This project is a Streamlit application that analyzes store video footage to detect customer movement patterns and identify "deadspots" - areas with low customer traffic. It uses computer vision techniques to detect people in the video and generates a heatmap of customer footfall. The app also provides a detailed report of items located in deadspot areas.

## Features
- Real-time people detection in store video footage
- Generation of customer footfall heatmap
- Identification of store deadspots
- Detailed report of products in deadspot areas, including category, brand, and weekly sales

## Requirements
- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Installation
1. Clone this repository:
   https://github.com/shinjini1202/StoreDeadspots.git
2. Install the required packages:
   pip install -r requirements.txt
## Usage
1. Ensure you have the following files in the project directory:
- `store.mp4`: Video file of the store
- `deploy.prototxt`: Network architecture file for the detection model
- `mobilenet_iter_73000.caffemodel`: Pre-trained weights for the detection model
- `store.csv`: CSV file containing store product data

2. Run the Streamlit app:
   streamlit run app.py
   
3. The app will open in your default web browser. You will see:
- The store video feed with bounding boxes around detected people
- A heatmap showing customer footfall
- A table displaying deadspot locations and associated product information
