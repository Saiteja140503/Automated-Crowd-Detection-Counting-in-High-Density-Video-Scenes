#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os


video_path = 'strrer.mp4'
output_folder = 'strrer_frames'
os.makedirs(output_folder, exist_ok=True)

# Desired frame rate for extraction
target_fps = 5

# Load video
cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)

if original_fps == 0:
    print(" Error: Could not read FPS from video.")
else:
    frame_interval = int(original_fps / target_fps)
    print(f"Processing '{video_path}' | Original FPS: {original_fps:.2f} | Extracting every {frame_interval} frames...")

    frame_num = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_num += 1

    cap.release()
    print(f" Done! Saved {saved_frame_count} frames to '{output_folder}/'")


# In[3]:


import cv2
import os

# Your video file
video_path = 'bparty.mp4'
output_folder = 'bparty_frames'
os.makedirs(output_folder, exist_ok=True)

# Desired frame rate for extraction
target_fps = 5

# Load video
cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)

if original_fps == 0:
    print(" Error: Could not read FPS from video.")
else:
    frame_interval = int(original_fps / target_fps)
    print(f" Processing '{video_path}' | Original FPS: {original_fps:.2f} | Extracting every {frame_interval} frames...")

    frame_num = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_num += 1

    cap.release()
    print(f" Done! Saved {saved_frame_count} frames to '{output_folder}/'")


# In[5]:


import cv2
import os

# Your video file
video_path = 'crowd2.mp4'
output_folder = 'crowd_frames'
os.makedirs(output_folder, exist_ok=True)

# Desired frame rate for extraction
target_fps = 5

# Load video
cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)

if original_fps == 0:
    print(" Error: Could not read FPS from video.")
else:
    frame_interval = int(original_fps / target_fps)
    print(f"Processing '{video_path}' | Original FPS: {original_fps:.2f} | Extracting every {frame_interval} frames...")

    frame_num = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_num += 1

    cap.release()
    print(f" Done! Saved {saved_frame_count} frames to '{output_folder}/'")


# In[7]:


import cv2
import os

# Your video file
video_path = 'rally.mp4'
output_folder = 'rally_frames'
os.makedirs(output_folder, exist_ok=True)

# Desired frame rate for extraction
target_fps = 5

# Load video
cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)

if original_fps == 0:
    print("Error: Could not read FPS from video.")
else:
    frame_interval = int(original_fps / target_fps)
    print(f" Processing '{video_path}' | Original FPS: {original_fps:.2f} | Extracting every {frame_interval} frames...")

    frame_num = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_num += 1

    cap.release()
    print(f"Done! Saved {saved_frame_count} frames to '{output_folder}/'")


# In[9]:


import cv2
import os

# Your video file
video_path = 'war.mp4'
output_folder = 'war_frames'
os.makedirs(output_folder, exist_ok=True)

# Desired frame rate for extraction
target_fps = 5

# Load video
cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)

if original_fps == 0:
    print("Error: Could not read FPS from video.")
else:
    frame_interval = int(original_fps / target_fps)
    print(f" Processing '{video_path}' | Original FPS: {original_fps:.2f} | Extracting every {frame_interval} frames...")

    frame_num = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_num += 1

    cap.release()
    print(f"Done! Saved {saved_frame_count} frames to '{output_folder}/'")


# In[11]:


import cv2
import os

# Your video file
video_path = 'flagh.mp4'
output_folder = 'flag_frames'
os.makedirs(output_folder, exist_ok=True)

# Desired frame rate for extraction
target_fps = 5

# Load video
cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)

if original_fps == 0:
    print("Error: Could not read FPS from video.")
else:
    frame_interval = int(original_fps / target_fps)
    print(f" Processing '{video_path}' | Original FPS: {original_fps:.2f} | Extracting every {frame_interval} frames...")

    frame_num = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_num += 1

    cap.release()
    print(f" Done! Saved {saved_frame_count} frames to '{output_folder}/'")


# In[13]:


import cv2
import os

# Your video file
video_path = 'fast.mp4'
output_folder = 'fast_frames'
os.makedirs(output_folder, exist_ok=True)

# Desired frame rate for extraction
target_fps = 5

# Load video
cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)

if original_fps == 0:
    print(" Error: Could not read FPS from video.")
else:
    frame_interval = int(original_fps / target_fps)
    print(f" Processing '{video_path}' | Original FPS: {original_fps:.2f} | Extracting every {frame_interval} frames...")

    frame_num = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_num += 1

    cap.release()
    print(f" Done! Saved {saved_frame_count} frames to '{output_folder}/'")


# In[15]:


import cv2
import os

# Your video file
video_path = 'fest2.mp4'
output_folder = 'fest2_frames'
os.makedirs(output_folder, exist_ok=True)

# Desired frame rate for extraction
target_fps = 5

# Load video
cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)

if original_fps == 0:
    print("Error: Could not read FPS from video.")
else:
    frame_interval = int(original_fps / target_fps)
    print(f" Processing '{video_path}' | Original FPS: {original_fps:.2f} | Extracting every {frame_interval} frames...")

    frame_num = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_num += 1

    cap.release()
    print(f" Done! Saved {saved_frame_count} frames to '{output_folder}/'")


# In[17]:


import cv2
import os

# Your video file
video_path = '856135-hd_1920_1080_30fps (1).mp4'
output_folder = 'cc_frames'
os.makedirs(output_folder, exist_ok=True)

# Desired frame rate for extraction
target_fps = 5

# Load video
cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)

if original_fps == 0:
    print(" Error: Could not read FPS from video.")
else:
    frame_interval = int(original_fps / target_fps)
    print(f" Processing '{video_path}' | Original FPS: {original_fps:.2f} | Extracting every {frame_interval} frames...")

    frame_num = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_num += 1

    cap.release()
    print(f" Done! Saved {saved_frame_count} frames to '{output_folder}/'")


# In[19]:


import cv2
import matplotlib.pyplot as plt

# List of video file paths
video_paths = [
    '856135-hd_1920_1080_30fps (1).mp4',
    'fest2.mp4',
    'fast.mp4',
    'strrer.mp4',
    'flagh.mp4',
    'war.mp4',
    'rally.mp4',
    'bparty.mp4',
    'crowd2.mp4'
]

# Parameters
max_frames_per_video = 20  # Limit frames per video
cols = 5  # Number of columns when plotting

# Process each video
for video_idx, video_path in enumerate(video_paths):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        continue

    frames = []
    frame_count = 0

    while frame_count < max_frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1

    cap.release()

    # Display frames
    rows = (len(frames) + cols - 1) // cols
    plt.figure(figsize=(20, rows * 4))
    for i, frame in enumerate(frames):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(frame)
        plt.title(f'Video {video_idx + 1} - Frame {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# In[22]:


get_ipython().system('pip install labelImg')


# In[20]:


import cv2
import os

# List of video file paths
video_paths = [
    '856135-hd_1920_1080_30fps (1).mp4',
    'fest2.mp4',
    'fast.mp4',
    'strrer.mp4',
    'flagh.mp4',
    'war.mp4',
    'rally.mp4',
    'bparty.mp4',
    'crowd2.mp4'
]

# Base directory to save frames
base_save_dir = r'C:/video_frames'  # Save directly under C drive
os.makedirs(base_save_dir, exist_ok=True)

# Frame extraction settings
frame_rate = 5  # Save every 5th frame

for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get file name without extension
    save_dir = os.path.join(base_save_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        continue

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            filename = f"frame_{saved_count:04}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f" Done! Saved {saved_count} frames from '{video_name}' to '{save_dir}'")

print(" All videos processed!")


# In[49]:


import shutil
import os

# Source base path (where your annotated folders are)
source_base = r'C:/Users/saite/OneDrive/Documents'

# Destination base path (where you want to copy them)
destination_base = r'C:/video_frames'

# List of folders to copy (same names as videos without .mp4)
folders_to_copy = [
    '856135_frames',
    'fest2_frames',
    'fast_frames',
    'strrer_frames',
    'flagh_frames',
    'war_frames',
    'rally_frames',
    'bparty_frames',
    'crowd2 (2)_frames'
]

for folder_name in folders_to_copy:
    source_folder = os.path.join(source_base, folder_name)
    destination_folder = os.path.join(destination_base, folder_name)
    
    if os.path.exists(source_folder):
        shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
        print(f"Copied '{folder_name}' to '{destination_folder}'")
    else:
        print(f" Folder '{folder_name}' not found at source!")

print(" All folders copied!")


# In[51]:


import os
import shutil
import random

# Path where your current annotated folders are
source_base = r'C:/video_frames'

# New dataset base path
dataset_base = r'C:/video_frames_yolov8'
images_train_dir = os.path.join(dataset_base, 'images', 'train')
images_val_dir = os.path.join(dataset_base, 'images', 'val')
labels_train_dir = os.path.join(dataset_base, 'labels', 'train')
labels_val_dir = os.path.join(dataset_base, 'labels', 'val')

# Create directories
for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Collect all (image, label) pairs
image_label_pairs = []
for folder_name in os.listdir(source_base):
    folder_path = os.path.join(source_base, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg'):
                img_path = os.path.join(folder_path, file_name)
                label_path = img_path.replace('.jpg', '.txt')
                if os.path.exists(label_path):
                    image_label_pairs.append((img_path, label_path))

# Shuffle and split into train and val
random.shuffle(image_label_pairs)
split_idx = int(0.8 * len(image_label_pairs))  # 80% train, 20% val
train_pairs = image_label_pairs[:split_idx]
val_pairs = image_label_pairs[split_idx:]

# Copy files
def copy_pairs(pairs, img_dest, label_dest):
    for img_path, label_path in pairs:
        shutil.copy(img_path, img_dest)
        shutil.copy(label_path, label_dest)

copy_pairs(train_pairs, images_train_dir, labels_train_dir)
copy_pairs(val_pairs, images_val_dir, labels_val_dir)

print("Dataset organized for YOLOv8 training at:", dataset_base)


# In[53]:


data_yaml_content = """
path: C:/video_frames_yolov8
train: images/train
val: images/val

names:
  0: car
  1: person
  2: bike
  3: umbrella
"""  

# Save it
with open('video_frames_data.yaml', 'w') as f:
    f.write(data_yaml_content)

print(" Created 'video_frames_data.yaml'")


# In[55]:


data_yaml_content = """
path: C:/video_frames_yolov8
train: images/train
val: images/val

names:
  0: car
  1: person
  2: bike
  3: umbrella
"""  

# Save it to the specific path
with open('C:/video_frames_yolov8/video_frames_data.yaml', 'w') as f:
    f.write(data_yaml_content)

print("Created 'video_frames_data.yaml' at C:/video_frames_yolov8")


# In[42]:


get_ipython().system('pip install ultralytics')


# In[57]:


get_ipython().system('yolo task=detect mode=train model=yolov8n.pt data=video_frames_data.yaml epochs=5 imgsz=120')


# In[65]:


from ultralytics import YOLO
# Load a YOLOv8 model (YOLOv8n is small; you can also use YOLOv8s, m, l, x)
model = YOLO('yolov8n.pt')  # pre-trained weights

# Train it
model.train(
    data='C:\\video_frames_yolov8\\video_frames_data.yaml',
    epochs=10,
    imgsz=120,
    batch=5,
    name='video_frames_yolov8_train'
)


# In[67]:


# Step 1: Import YOLOv8
from ultralytics import YOLO

# Step 2: Load your trained model
model = YOLO('runs/detect/video_frames_yolov8_train10/weights/best.pt')  # make sure path is correct

# Step 3: Predict on validation images (or any images)
results = model.predict(
    source='C:/video_frames_yolov8(1)/images/val/',  # path to val folder (or use 'train' folder if you want)
    imgsz=640,         # input image size for better results
    conf=0.25,         # minimum confidence threshold
    save=True,         # save results with bounding boxes
    show=True          # show predictions immediately
)

# Step 4: Check output images
print("Predicted images are saved inside: 'runs/detect/predict/' folder!")


# In[73]:


import os
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Set the path to your predicted images
output_folder = 'runs/detect/predict/'

# Step 2: List all the image files in the output folder
predicted_images = os.listdir(output_folder)

# Step 3: Show the first few images
for i, image_file in enumerate(predicted_images[:20]):  # Show first 5 images
    img_path = os.path.join(output_folder, image_file)
    img = Image.open(img_path)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(f'Predicted Image {i+1}')
    plt.axis('off')  # Hide axes
    plt.show()


# In[87]:


from ultralytics import YOLO
# Load a YOLOv8 model (YOLOv8n is small; you can also use YOLOv8s, m, l, x)
model = YOLO('yolov8n.pt')  # pre-trained weights

# Train it
model.train(
    data='C:\\video_frames_yolov8\\video_frames_data.yaml',
    epochs=50,
    imgsz=620,
    batch=10,
    name='video_frames_yolov8_train'
)


# In[91]:


results_dict = {
    'metrics/precision(B)': 0.755966743583531,
    'metrics/recall(B)': 0.7353383458646616,
    'metrics/mAP50(B)': 0.8000839687914348,
    'metrics/mAP50-95(B)': 0.41588214466967105,
    'fitness': 0.4543023270818475
}


print("Evaluation Metrics:")
for key, value in results_dict.items():
    print(f"{key}: {value:.4f}")


# In[93]:


import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image

# Function to process frames and count persons
def count_persons_in_frames(frame_folder):
    # Get all the frames from the folder
    frames = os.listdir(frame_folder)
    frame_counts = []

    for frame_name in frames:
        # Construct the full frame path
        frame_path = os.path.join(frame_folder, frame_name)

        # Read the frame (image)
        img = cv2.imread(frame_path)

        # Perform detection on the frame
        results = model(img)  # YOLOv8 inference

        # Filter detections for class 'person' (ID = 1)
        persons = results[0].boxes.cls == 1  # Filter only 'person' class

        # Count the number of people detected
        person_count = persons.sum().item()

        # Save or display results
        frame_counts.append((frame_name, person_count))

        # Display the frame with the number of detected persons
        annotated_img = results[0].plot()  # Annotated image with bounding boxes
        annotated_pil_img = Image.fromarray(annotated_img)  # Convert to PIL image for display
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_pil_img)
        plt.title(f"Frame: {frame_name} - Person Count: {person_count}")
        plt.axis('off')  # Hide axes
        plt.show()

    return frame_counts


frame_folder = 'C:/video_frames_yolov8(1)/images/new_video_frames/'

# Count persons in each frame and show the results
frame_person_counts = count_persons_in_frames(frame_folder)

# Print the person counts for each frame
print("Person counts per frame:")
for frame_name, count in frame_person_counts:
    print(f"{frame_name}: {count} persons")


# In[99]:


import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image

# Function to process frames and count persons
def count_persons_in_frames(frame_folder):
    # Get all the frames from the folder
    frames = os.listdir(frame_folder)
    frame_counts = []

    for frame_name in frames:
        # Construct the full frame path
        frame_path = os.path.join(frame_folder, frame_name)

        # Read the frame (image)
        img = cv2.imread(frame_path)

        # Perform detection on the frame
        results = model(img)  # YOLOv8 inference

        # Filter detections for class 'person' (ID = 1)
        persons = results[0].boxes.cls == 1  # Filter only 'person' class

        # Count the number of people detected
        person_count = persons.sum().item()

        # Save or display results
        frame_counts.append((frame_name, person_count))

        # Display the frame with the number of detected persons
        annotated_img = results[0].plot()  # Annotated image with bounding boxes
        annotated_pil_img = Image.fromarray(annotated_img)  # Convert to PIL image for display
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_pil_img)
        plt.title(f"Frame: {frame_name} - Person Count: {person_count}")
        plt.axis('off')  # Hide axes
        plt.show()

    return frame_counts

frame_folder = 'C:\\video_frames_yolov8(1)\\images\\train'

# Count persons in each frame and show the results
frame_person_counts = count_persons_in_frames(frame_folder)

# Print the person counts for each frame
print("Person counts per frame:")
for frame_name, count in frame_person_counts:
    print(f"{frame_name}: {count} persons")


# Evaluation Metrics:
# metrics/precision(B): 0.7560
# metrics/recall(B): 0.7353
# metrics/mAP50(B): 0.8001
# metrics/mAP50-95(B): 0.4159
# fitness: 0.4543

# In[ ]:




