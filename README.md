# Smart Attendance System

A face recognition-based attendance system using OpenCV's LBPH (Local Binary Patterns Histograms) algorithm for face recognition, Haar Cascade for face detection, and YOLOv8 for person detection. The system captures face datasets, trains a recognition model, and automatically marks attendance through real-time face recognition.

## Features

- **Face Registration**: Capture face samples per person with automatic dataset organization
- **LBPH Model Training**: Train face recognition models that work well with varying lighting conditions
- **Real-time Recognition**: Automatic attendance marking via camera with confidence threshold
- **Multi-Stage Detection**: YOLO person detection → Haar face detection → LBPH recognition
- **Duplicate Prevention**: Same person cannot be marked twice per day
- **Modern Dashboard**: CustomTkinter GUI with dark theme and statistics
- **Attendance Logging**: CSV-based attendance records with timestamps

## How It Works

### Face Recognition (LBPH)

LBPH analyzes face images by dividing them into small cells and creating histograms of local binary patterns. These histograms are concatenated into a feature vector that uniquely identifies each person. The algorithm is robust to lighting changes and works well for frontal face recognition.

Key aspects:
- Trains on captured face samples from each person
- Generates a feature vector for each face image
- Uses Chi-Square distance to compare faces during recognition
- Lower confidence scores indicate better matches

### Face Detection (Haar Cascade)

Uses OpenCV's pre-trained Haar Cascade classifier to detect faces in images. This classifier uses a cascade of simple features to quickly identify face regions.

Parameters:
- Scale factor: Controls detection speed vs accuracy trade-off
- Minimum neighbors: Higher values reduce false positives
- Minimum face size: Ensures only sufficiently large faces are detected

### Person Detection (YOLOv8)

YOLOv8 Nano provides person detection to improve the recognition pipeline by focusing processing on regions containing people. This reduces false positives and improves overall accuracy.

Characteristics:
- ~6.5MB model size
- Fast single-pass detection
- Optional GPU acceleration
- Filters for person class only

## Model Training

Training analyzes captured face images, extracts facial features using LBPH, and creates:
- `trainer.yml`: The trained recognition model
- `labels.npy`: ID-to-name mapping for translating predictions

### Training Process

1. **Data Loading**: Scans the dataset directory for user folders, loads all face images, detects and crops faces
2. **Model Training**: Creates an LBPH recognizer and trains it on all face images with their corresponding person IDs
3. **Model Saving**: Saves the trained model and label mapping to disk

### Training Data Guidelines

**Per Person:**
- Minimum: 10 images
- Recommended: 45 images (default)
- Optimal: 60-80 images

**Image Variety:**
- Different lighting conditions
- Slight head rotations
- Various expressions (neutral, smile)
- Different distances from camera

**Image Specifications:**
- Format: Grayscale JPEG
- Face size: At least 50×50 pixels
- Captured at 100×100 to 200×200 pixels after crop

### Training Quality

**Good Training Indicators:**
- Low confidence scores (0-50) for recognized faces
- High recognition rate (>90%)
- Model size proportional to number of samples

**Warning Signs:**
- High confidence scores (50-100+)
- Low recognition rate
- Unusually small model file

## Recognition Pipeline

The system processes video frames through multiple stages:

1. **YOLO Person Detection**: Identifies regions containing people
2. **Haar Face Detection**: Locates faces within person regions
3. **LBPH Recognition**: Identifies the person and returns a confidence score
4. **Attendance Logging**: Records attendance if confidence is below threshold

### Confidence Threshold

The system uses a confidence threshold (default: 85) to determine valid recognitions:
- **Lower scores (0-85)**: Better match, attendance marked
- **Higher scores (85+)**: Poor match, treated as unknown face

Note: LBPH uses Chi-Square distance where lower values indicate better matches.

### Performance Considerations

**Factors affecting accuracy:**
- Lighting conditions (even lighting works best)
- Face angle (frontal faces preferred)
- Face occlusions (masks significantly reduce accuracy)
- Number of training samples per person

**Scalability:**
- Works well for 1-100 registered people
- Performance degrades with very large datasets (>500 people)
- Recognition time increases slightly with more registered faces

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or IP camera
- 2GB+ RAM
- 1GB free storage

### Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- `opencv-contrib-python` - Face recognition (includes face module)
- `ultralytics` - YOLOv8 person detection
- `customtkinter` - Modern GUI framework
- `numpy`, `pandas`, `pillow` - Data handling

The YOLO model (`yolov8n.pt`, ~6.5MB) downloads automatically on first run.

## Usage

### GUI Application (Recommended)

```bash
python main.py
```

**Workflow:**

1. **Register Students**:
   - Enter student name in "Full Name" field
   - Click "Capture Dataset" (captures 45 samples by default)
   - Position face in camera window, vary angle/lighting during capture
   - Wait for "Dataset capture finished" message

2. **Train Model**:
   - Click "Train Model" button
   - System analyzes all captured faces and creates the recognition model
   - Model saved to `data/trainer/trainer.yml`

3. **Start Recognition**:
   - Click "Start Recognition"
   - Show face to camera
   - Green box = recognized, Red box = unknown
   - Attendance marked automatically for recognized faces
   - Press 'q' to exit

4. **View Dashboard**:
   - "Registered Students" shows all captured students
   - "Attendance Records" shows today's attendance
   - Statistics show counts and model status

### Command Line Tools

**Capture faces:**
```bash
python -m src.core.capture --id 1 --name "Student Name" --samples 45 --camera 0
```

**Train model:**
```bash
python -m src.core.train
```

**Run recognition:**
```bash
python -m src.core.recognize --camera 0 --backend dshow
```

## Project Structure

```
.
├── main.py              # Entry point - launches GUI application
├── src/
│   ├── core/            # Core business logic
│   │   ├── capture.py   # Face capture module
│   │   ├── recognize.py # Face recognition & attendance
│   │   └── train.py     # LBPH model training
│   ├── detection/       # Detection algorithms
│   │   ├── face_detector.py    # Haar Cascade face detection
│   │   └── person_detector.py  # YOLOv8 person detection
│   ├── gui/             # User interface
│   │   └── dashboard.py # CustomTkinter GUI
│   └── utils/           # Utility modules
│       ├── attendance.py # Attendance CSV operations
│       ├── camera.py     # Video source handling
│       ├── config.py     # Paths & constants
│       └── dataset.py    # Dataset management
├── models/              # ML model files
│   └── yolov8n.pt       # YOLOv8 Nano person detection model
├── data/                # Data storage (auto-created)
│   ├── dataset/         # Face images
│   ├── trainer/         # Trained LBPH models
│   └── attendance/      # Attendance CSV records
├── requirements.txt     # Python dependencies
├── README.md           # Documentation
└── .gitignore          # Git ignore rules
```

## Configuration

Key parameters can be adjusted in the source files:

| Parameter | File | Description |
|-----------|------|-------------|
| Confidence Threshold | `src/core/recognize.py` | Recognition threshold (lower = stricter) |
| Haar Scale Factor | `src/core/capture.py` | Detection speed vs accuracy trade-off |
| Default Samples | `src/gui/dashboard.py` | Images captured per person |
| Data Directory | `src/utils/config.py` | Base path for all data storage |

## Data Storage

### Directory Structure (Auto-Created)

```
data/
├── dataset/              # Face images
│   └── user_1_name/      # Person folder (ID extracted from name)
│       ├── 001.jpg
│       ├── 002.jpg
│       └── ...
├── trainer/              # Trained models
│   ├── trainer.yml       # LBPH recognition model
│   └── labels.npy        # ID-to-name mapping
└── attendance/           # Attendance records
    └── attendance.csv    # CSV format: person_id,name,timestamp
```

**File formats:**
- `trainer.yml`: OpenCV LBPH model storage
- `labels.npy`: NumPy dictionary mapping IDs to names
- `attendance.csv`: CSV with headers `person_id,name,timestamp`

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "No training data found" | Capture images first with the GUI or CLI capture tool |
| "Model file not found" | Click "Train Model" or run training via CLI |
| High confidence scores | Recapture with better, even lighting |
| Recognition fails | Verify training data quality and retrain if needed |
| Camera not detected | Try different camera index (0, 1, 2) or backend |
| Attendance not recording | Check confidence threshold; may need recapture |
| "Already marked today" | Normal duplicate prevention behavior |

### Tips for Better Results

**Accuracy:**
- Use even, bright lighting (avoid shadows and backlighting)
- Capture 45+ samples per person with variety in angle/expression
- Position camera at face level, 1-1.5 meters distance
- Use plain backgrounds when possible

**Speed:**
- Lower camera resolution if FPS is insufficient
- Adjust Haar scale factor for faster (but less accurate) detection
- Skip frames during recognition if needed

## System Requirements

**Minimum:**
- CPU: Intel i3 / AMD Ryzen 3
- RAM: 4GB
- Storage: 1GB
- Camera: 720p USB webcam
- OS: Windows 10 / Ubuntu 20.04 / macOS 10.14+

**Recommended:**
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 8GB
- Storage: 5GB
- Camera: 1080p USB webcam
- GPU: Optional for faster YOLO inference

## License

MIT License

## References

- **LBPH**: Ahonen, T., Hadid, A., & Pietikäinen, M. (2006). Face description with local binary patterns: Application to face recognition. IEEE TPAMI.
- **Haar Cascade**: Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. IEEE CVPR.
- **YOLO**: Redmon, J., et al. (2016). You only look once: Unified, real-time object detection. IEEE CVPR.
