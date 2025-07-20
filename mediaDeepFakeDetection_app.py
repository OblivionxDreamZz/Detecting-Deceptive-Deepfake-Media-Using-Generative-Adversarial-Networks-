import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
import librosa

# Load the saved image/video deepfake detection model
@st.cache_resource
def load_image_video_model():
    model = tf.keras.models.load_model(r'C:\Users\Student\Downloads\deepfake_detector_model2.keras')
    return model

image_video_model = load_image_video_model()

# Load the saved voice deepfake detection model
@st.cache_resource
def load_voice_model():
    voice_model = tf.keras.models.load_model(r'C:\Users\Student\Downloads\Deepfake_VoiceDetection_Model3.keras')
    return voice_model

voice_model = load_voice_model()

# Placeholder for deepfake detection function (for image)
def detect_deepfake_image(image):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = image_video_model.predict(image_array)
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
        if frame_count % 10 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame)
            frame_image = frame_image.resize((128, 128))
            frame_array = np.array(frame_image) / 255.0
            frames.append(frame_array)
    
    cap.release()
    frames_array = np.array(frames)
    predictions = image_video_model.predict(frames_array)
    avg_prediction = np.mean(predictions)
    result = "Fake" if avg_prediction >= 0.5 else "Real"
    confidence = avg_prediction if result == "Fake" else 1 - avg_prediction
    
    return result, confidence

# Function to extract features from an audio file and detect deepfake
def detect_deepfake_audio(audio_path):
    # Load audio file and extract features (MFCCs)
    audio, sr = librosa.load(audio_path, sr=22050)  # Ensure sample rate is consistent
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)  # Average MFCC features over time

    # Reshape for model input
    mfccs_scaled = np.expand_dims(mfccs_scaled, axis=0)
    mfccs_scaled = np.expand_dims(mfccs_scaled, axis=2)

    # Perform prediction
    prediction = voice_model.predict(mfccs_scaled)
    print(f"Raw prediction: {prediction}")  # Debugging statement
    
    # Ensure prediction is in the expected format
    result = "Fake" if prediction[0][0] >= 0.5 else "Real"
    confidence = float(prediction[0][0]) if result == "Fake" else 1 - float(prediction[0][0])
    
    return result, confidence

# Streamlit Dashboard Layout
st.title("Deepfake Detection Dashboard")
st.write("Upload an image, video, or audio file to check for deepfake content.")

# File upload
uploaded_file = st.file_uploader("Choose an image, video, or audio file...", type=["jpg", "jpeg", "png", "mp4", "wav", "mp3"])

if uploaded_file is not None:
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Analyzing...")
        
        # Perform deepfake detection on the image
        result, confidence = detect_deepfake_image(image)
        
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
    
    elif uploaded_file.type in ["audio/wav", "audio/mp3"]:
        # Process audio
        st.audio(uploaded_file)
        st.write("Analyzing audio...")

        # Save uploaded audio to disk temporarily
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Perform deepfake detection on the audio
        result, confidence = detect_deepfake_audio(temp_audio_path)
        
        # Clean up temporary file
        os.remove(temp_audio_path)
        
        # Display results
        st.subheader("Detection Results")
        st.write(f"Result: **{result}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
else:
    st.info("Please upload an image, video, or audio file for deepfake detection.")

# Sidebar information
st.sidebar.header("About")
st.sidebar.write("""
This tool uses machine learning models to detect deepfakes in images, videos, and voice recordings.
The models analyze the uploaded file and provide a confidence score for the detection.
""")
