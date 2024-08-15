import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Load the necessary files
cwd = os.getcwd()
video_path = os.path.abspath(os.path.join(cwd, 'store.mp4'))
prototxt_path = os.path.abspath(os.path.join(cwd, 'deploy.prototxt'))
model_path = os.path.abspath(os.path.join(cwd, 'mobilenet_iter_73000.caffemodel'))

# Load the video and initialize the detection model
cap = cv2.VideoCapture(video_path)
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load the store data
store_data = pd.read_csv('store.csv')

def detect_people(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    people = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Class label for person
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                people.append((startX, startY, endX, endY))
    return people

def draw_bounding_boxes(frame, people):
    for (startX, startY, endX, endY) in people:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return frame

def calculate_footfall(frame, people):
    tracks = defaultdict(list)
    track_id = 0

    for (startX, startY, endX, endY) in people:
        centroid = (int((startX + endX) / 2), int((startY + endY) / 2))
        tracks[track_id].append(centroid)
        track_id += 1

    grid_size = (5, 5)  # Adjust grid size as needed
    footfall = np.zeros(grid_size)

    for track in tracks.values():
        for (x, y) in track:
            grid_x = x // (frame.shape[1] // grid_size[0])
            grid_y = y // (frame.shape[0] // grid_size[1])
            footfall[grid_y, grid_x] += 1

    return footfall

def plot_heatmap(footfall):
    plt.figure(figsize=(8, 6))
    sns.heatmap(footfall, cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'Foot Traffic'})
    plt.title('Customer Footfall Heatmap')
    plt.xlabel('Store Width')
    plt.ylabel('Store Length')
    return plt.gcf()

def generate_report(footfall):
    deadspots = []
    threshold = np.percentile(footfall, 25)  # Define a threshold for low footfall areas

    grid_size = footfall.shape
    items_per_grid = len(store_data) // (grid_size[0] * grid_size[1])

    report_data = []

    for i in range(footfall.shape[0]):
        for j in range(footfall.shape[1]):
            if footfall[i, j] <= threshold:
                deadspots.append((i, j))
                start_index = (i * grid_size[1] + j) * items_per_grid
                end_index = start_index + items_per_grid
                items = store_data.iloc[start_index:end_index]
                for _, item in items.iterrows():
                    report_data.append({
                        "Grid Location": f"Y: {i+1}, X: {j+1}",
                        "Product": item['Product'],
                        "Brand": item['Brand'],
                        "Weekly Sales": f"${item['Weekly Sales']}"
                    })

    report_df = pd.DataFrame(report_data)
    
    summary = f"Total Grids: {footfall.size}\n"
    summary += f"Deadspots Identified: {len(deadspots)}"

    return summary, report_df

def get_best_selling_products_with_brands(category):
    category_data = store_data[store_data['Category'] == category]
    best_selling = category_data.groupby(['Product', 'Brand'])['Weekly Sales'].sum().sort_values(ascending=False).head(5)
    return best_selling

def plot_best_selling_products_with_brands(data):
    plt.figure(figsize=(12, 6))
    ax = data.plot(kind='bar', color='skyblue', edgecolor='navy')
    plt.title('Top 5 Best-Selling Products with Brands', fontsize=16)
    plt.xlabel('Product (Brand)', fontsize=12)
    plt.ylabel('Total Weekly Sales ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Modify x-axis labels to show Product (Brand)
    labels = [f"{product} ({brand})" for product, brand in data.index]
    ax.set_xticklabels(labels)
    
    # Add value labels on top of each bar
    for i, v in enumerate(data):
        ax.text(i, v, f'${v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt.gcf()

def run_app():
    st.set_page_config(layout="wide", page_title="Store Analysis Dashboard")
    
    # Apply custom CSS for layout and styling
    st.markdown("""
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .stPlotlyChart {
        height: 400px;
    }
    .stDataFrame {
        height: 400px;
        overflow-y: auto;
    }
    .title {
        color: #2ca02c;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">Store Analysis Dashboard</div>', unsafe_allow_html=True)

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Live Feed & Heatmap", "Category Analysis", "Deadspots Report"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Live Video Feed")
            frame_placeholder = st.empty()
        
        with col2:
            st.subheader("Customer Footfall Heatmap")
            heatmap_placeholder = st.empty()

    with tab2:
        st.subheader("Category Analysis")
        categories = store_data['Category'].unique()
        selected_category = st.selectbox("Select a category", categories)
        best_selling = get_best_selling_products_with_brands(selected_category)
        
        st.pyplot(plot_best_selling_products_with_brands(best_selling))

    with tab3:
        st.subheader("Deadspots Report")
        col3, col4 = st.columns([1, 3])
        
        with col3:
            summary_placeholder = st.empty()
        
        with col4:
            table_placeholder = st.empty()

    # Start the video processing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people and draw bounding boxes
        people = detect_people(frame)
        frame_with_boxes = draw_bounding_boxes(frame, people)

        # Update the video feed
        frame_placeholder.image(frame_with_boxes, channels="BGR", use_column_width=True)

        # Calculate and display the heatmap
        footfall = calculate_footfall(frame, people)
        heatmap_placeholder.pyplot(plot_heatmap(footfall))

        # Generate and display the report
        summary, report_df = generate_report(footfall)
        summary_placeholder.text(summary)
        table_placeholder.dataframe(report_df)

        # Add a short pause to reduce CPU usage
        plt.pause(0.1)

    # Release the video capture
    cap.release()

if __name__ == "__main__":
    run_app()