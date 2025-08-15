# Automated-Crowd-Detection-Counting-in-High-Density-Video-Scenes
Developed a computer vision system to detect and count people in high-density video scenes. Handled occlusions, overlaps, and perspective distortions to provide accurate frame-wise crowd counts and visual annotations for real-time crowd monitoring and analysis

## Features
- Detects people in dense crowds with high accuracy.
- Counts the number of people in each frame of a video.
- Handles varying crowd densities and camera perspectives.
- Supports video input and frame-wise output for analysis.

## Technologies Used
- **Python** – Main programming language.
- **OpenCV** – Video processing and frame extraction.
- **YOLOv8** – Object detection model for people detection.
- **NumPy / Pandas** – Data manipulation and processing.
- **Matplotlib / Seaborn** – Visualization of crowd counts.

## Installation
1. Clone the repository:  
   ```bash
   git clone <your-repo-link>

Install dependencies:

    pip install -r requirements.txt

Usage

Prepare your video or frames in the designated folder.

Run the detection script:

    python detect_people.py --source path/to/video
Results

Outputs annotated frames with detected people.

Provides a CSV summary of people count per frame.

Enables visualization of crowd density trends over time.

Contributing

Fork the repository and create a new branch for features or bug fixes.

Submit a pull request with a clear description of changes.

License

This project is licensed under the MIT License – see the LICENSE file for details.

Contact

Author: Saiteja Nandanakari

Email: saitejanandanakari@gmail.com



