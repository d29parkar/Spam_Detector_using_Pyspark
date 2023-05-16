#!/usr/bin/env python
# coding: utf-8

# Install required packages
# !pip install pyspark
# !pip install opencv-python

from pyspark.sql import SparkSession
import cv2
from pyspark.streaming import StreamingContext

# Set up PySpark streaming context
spark = SparkSession.builder \
    .appName("ESRGANVideoStreaming") \
    .getOrCreate()

sc = spark.sparkContext
ssc = StreamingContext(sc, batchDuration=1)  # Specify the batch duration

# Read the input video using TextFileStream
input_video_stream = ssc.textFileStream("./Data/trimmed.mp4")

# Import required libraries
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

# Load the pre-trained ESRGAN model
esrgan_model = torch.load('/content/RRDB_ESRGAN_x4.pth')

# Create the image preprocessing and postprocessing transforms
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
postprocess = transforms.Compose([
    transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444]),
    transforms.ToPILImage()
])

# Function to apply ESRGAN to a single frame
def apply_esrgan(frame):
    # Preprocess the frame
    input_tensor = preprocess(frame)

    # Add batch dimension to the input tensor
    input_tensor = input_tensor.unsqueeze(0)

    # Convert the input tensor to a CUDA tensor if available
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Apply the ESRGAN model to the input tensor
    with torch.no_grad():
        enhanced_tensor = esrgan_model(input_tensor)

    # Convert the enhanced tensor to a PIL image
    enhanced_image = postprocess(enhanced_tensor.squeeze().cpu())

    # Convert the PIL image to an OpenCV image
    enhanced_frame = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

    # Return the enhanced frame
    return enhanced_frame

# Apply ESRGAN to each frame using map transformation
enhanced_frames_stream = input_video_stream.map(apply_esrgan)

# Define the output video parameters
output_width = 640
output_height = 480
output_fps = 30

# Create a VideoWriter object
video_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (output_width, output_height))

# Function to write the processed frames to an output video
def write_video_output(rdd):
    frames = rdd.collect()
    for frame in frames:
        # Resize the frame to the desired output dimensions
        frame = cv2.resize(frame, (output_width, output_height))

        # Write the frame to the output video
        video_writer.write(frame)

# Stop the video writer and release the output video file
def stop_video_writer():
    video_writer.release()

# Write the processed frames to the output video
enhanced_frames_stream.foreachRDD(write_video_output)

# Start the streaming context
ssc.start()
ssc.awaitTermination()
