import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os

# Load the saved model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r'C:\Users\Student\Downloads\deepfake_detector_model.keras')
    return model

model = load_model()

# Placeholder for deepfake detection function (for images)
def detect_deepfake(image):
    # Preprocess the image
    image = image.resize((128, 128))  # Resize to match the input shape of the model
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Perform model prediction
    prediction = model.predict(image_array)
    
    # Convert prediction to label and confidence
    result = "Fake" if prediction >= 0.5 else "Real"
    confidence = float(prediction[0][0]) if result == "Fake" else 1 - float(prediction[0][0])
    
    return result, confidence

# Function to detect deepfake in a video
def detect_deepfake_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Sample every 10th frame to reduce computation time
        if frame_count % 10 == 0:
            # Convert frame to PIL Image and resize it
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame)
            frame_image = frame_image.resize((128, 128))  # Resize frame
            
            # Append preprocessed frame to list
            frame_array = np.array(frame_image) / 255.0
            frames.append(frame_array)
    
    cap.release()
    
    # Convert frames list to a numpy array and predict on frames
    frames_array = np.array(frames)
    predictions = model.predict(frames_array)
    
    # Average the predictions for the video
    avg_prediction = np.mean(predictions)
    result = "Fake" if avg_prediction >= 0.5 else "Real"
    confidence = avg_prediction if result == "Fake" else 1 - avg_prediction
    
    return result, confidence

# Streamlit Dashboard Layout
st.title("Deepfake Detection Dashboard")
st.write("Upload an image or video to check for deepfake content.")

# File upload
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Analyzing...")
        
        # Perform deepfake detection on the image
        result, confidence = detect_deepfake(image)
        
        # Display results
        st.subheader("Detection Results")
        st.write(f"Result: **{result}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
    
    elif uploaded_file.type == "video/mp4":
        # Process video
        st.video(uploaded_file)
        st.write("Analyzing video... This may take a while.")
        
        # Save uploaded video to disk temporarily
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Perform deepfake detection on the video
        result, confidence = detect_deepfake_video(temp_video_path)
        
        # Clean up temporary file
        os.remove(temp_video_path)
        
        # Display results
        st.subheader("Detection Results")
        st.write(f"Result: **{result}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
else:
    st.info("Please upload an image or video for deepfake detection.")

# Sidebar information
st.sidebar.header("About")
st.sidebar.write("""
This tool uses a CNN machine learning model to detect deepfakes in images or videos.
The model analyzes the uploaded file and provides a confidence score for the detection.
""")
