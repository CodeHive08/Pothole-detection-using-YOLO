# Pothole Detection

This project uses a YOLO-based deep learning model to detect potholes in road videos. It leverages the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) library for object detection and segmentation, and OpenCV for video processing and visualization.

## Features

- Detects potholes in video frames using a trained YOLO model (`best.pt`)
- Draws contours and bounding boxes around detected potholes
- Displays class names and confidence scores on each detection
- Filters overlapping detections to avoid duplicate predictions per frame
- Processes video input and displays results in real-time

## Screenshots
<img width="1920" height="1026" alt="image" src="https://github.com/user-attachments/assets/b04479d8-eea1-4dca-a161-24b2d5cdd9f6" />

## Folder Structure

```
Pothole detection/
├── best.pt                # Trained YOLO model weights
├── main.py                # Main detection and visualization script
├── video.mp4              # Input video for pothole detection
├── README.md              # Project documentation
└── (other files)
```

## Requirements

- Python 3.8+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV (`cv2`)
- NumPy
- cvzone (optional, not used in current code)

Install dependencies:
```bash
pip install ultralytics opencv-python numpy cvzone
```

## Usage

1. Place your trained YOLO model (`best.pt`) and input video (`video.mp4`) in the project folder.
2. Run the detection script:
    ```bash
    python main.py
    ```
3. The script will display the video with detected potholes highlighted.

## Output

- Detected potholes are outlined with contours and labeled with class names and confidence scores.
- The processed video is displayed in a window. Press `q` to exit.

## Author

Made by [Sandeep Singh Mehta](https://github.com/CodeHive08)

---
