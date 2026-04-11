# Face Monitor 🔍

**Face Monitor** is a desktop application for real-time face detection and watchlist-based face recognition. Built with OpenCV for video processing, DeepFace for face recognition, and PySide6 for a modern GUI interface.

## 📸 Screenshots

### Main Application Window

![Face Monitor Application](face_monitor.png)

## ✨ Features

- 🎥 **Multiple Video Sources**: Camera, screen capture, or video file
- 👤 **Real-time Face Detection**: Detect faces from live video streams
- 🎯 **Watchlist Matching**: Match detected faces against a watchlist database
- 📸 **Auto-Save Snapshots**: Automatically save full-frame images with bounding boxes
- 📜 **Detection History**: View complete history with timestamps and images
- 🗂️ **Watchlist Management**: Easy-to-manage watchlist with nested folder support
- 🎨 **Modern Dark UI**: Professional dark-themed interface with PySide6
- 🔧 **Configurable Settings**: Adjust models, detectors, thresholds, and performance
- 💾 **SQLite Storage**: Local database for detection history
- 🔊 **Audio Alerts**: Sound notifications for new detections

## 📋 Requirements

- **OS**: Windows (recommended for current scripts)
- **Python**: 3.12
- **pip**: Latest version

### Key Dependencies

- **PySide6** 6.6.2 - Qt for Python GUI framework
- **OpenCV** 4.10.0 - Computer vision and video processing
- **DeepFace** 0.0.79 - Face recognition library
- **TensorFlow** 2.19.0 - Machine learning backend
- **NumPy** 1.26.4 - Numerical computing
- **Pandas** 2.2.3 - Data manipulation
- **mss** - Screen capture
- **MTCNN**, **RetinaFace** - Face detectors

## 🚀 Quick Start

### Installation (Windows)

1. **Clone the repository**:
   ```powershell
   git clone <repository-url>
   cd face_monitor
   ```

2. **Run the installer**:
   ```powershell
   .\clear_and_install.bat
   ```

   This script will:
   - Create a fresh Python virtual environment
   - Install all required packages
   - Set environment variables (`TF_USE_LEGACY_KERAS=1`)
   - Generate helper launch scripts (`launcher.bat`, `diagnostic.bat`)

3. **Verify installation** (optional):
   ```powershell
   .\diagnostic.bat
   ```

### Running the Application

#### Option 1: Batch Launcher (Recommended)
```powershell
.\launcher.bat
```

#### Option 2: Python Module
```powershell
.\venv\Scripts\python -m app.launcher
```

#### Option 3: Activated Virtual Environment
```powershell
venv\Scripts\activate
python -m app.launcher
```

## 📁 Project Structure

```
face_monitor/
├── app/
│   ├── config/
│   │   ├── config.py              # Main configuration file
│   │   ├── settings_store.py      # Settings persistence
│   │   └── __init__.py
│   ├── recognition/
│   │   ├── processor.py           # Face detection/recognition pipeline
│   │   └── __init__.py
│   ├── storage/
│   │   ├── async_saver.py         # Async image saving
│   │   ├── db.py                  # SQLite database operations
│   │   ├── files.py               # File I/O utilities
│   │   └── __init__.py
│   ├── ui/
│   │   ├── detected_face_widget.py    # Detected face card widget
│   │   ├── history_panel.py           # History tab UI
│   │   ├── logging.py                 # Stdout-to-Qt log bridge
│   │   ├── overlay.py                 # UI drawing helpers
│   │   ├── settings_panel.py          # Settings tab UI
│   │   ├── video_display_label.py     # Resizable video widget
│   │   ├── watchlist_panel.py         # Watchlist tab UI
│   │   └── __init__.py
│   ├── video/
│   │   ├── thread.py              # Video capture/processing thread
│   │   ├── win_capture.py         # Windows screen capture
│   │   └── __init__.py
│   ├── launcher.py                # Application entry point
│   ├── main_window.py             # Main window class
│   └── __init__.py
├── watchlist/                     # Known faces directory (supports nested folders)
│   └── [Person Name]/
│       ├── photo1.jpg
│       ├── photo2.jpg
│       └── [Person Name].txt      # Optional: person description
├── detections/                    # Auto-saved detection snapshots
├── face_crop/                     # Face cropping utility
├── detections.db                  # SQLite detection history database
├── settings.json                  # Application settings
├── clear_and_install.bat          # Installation script
├── launcher.bat                   # Application launcher
├── diagnostic.bat                 # Installation diagnostic tool
├── icon.png                       # Application icon
└── README.md                      # This file
```

## 🔧 Configuration

### Settings File

Application settings are stored in `settings.json` and can be modified through the GUI Settings panel.

### Configuration Presets

The application includes predefined configuration presets in `app/config/config.py`:

- **Fast**: Fastest processing, lower accuracy (Facenet + opencv)
- **Streaming**: Optimized for live feeds (Facenet512 + ssd)
- **Balanced**: Good accuracy and speed (Facenet512 + mtcnn) - *Default*

### Key Configuration Options

| Setting | Description | Default |
|---------|-------------|----------|
| `MODEL_NAME` | Recognition model | `Facenet512` |
| `DETECTOR` | Face detector | `mtcnn` |
| `SIMILARITY_THRESHOLD` | Match threshold (0-1) | `0.4` |
| `DISTANCE_METRIC` | Distance metric | `cosine` |
| `PROCESS_INTERVAL` | Process every Nth frame | `3` |
| `DETECTION_SCALE` | Frame resize scale | `0.75` |
| `ASYNC_PROCESSING` | Enable async mode | `False` |
| `EXPAND_FACE_BOX` | Expand face box | `True` |
| `FACE_BOX_EXPAND_FACTOR` | Box expansion factor | `1.8` |

### Available Models

- **VGG-Face**: Classic CNN-based model
- **Facenet**: Google's FaceNet model
- **Facenet512**: Higher dimensional FaceNet (most accurate)
- **OpenFace**: Lightweight model
- **DeepFace**: Facebook's model
- **ArcFace**: State-of-the-art accuracy
- **Dlib**: Traditional computer vision (requires dlib package)

### Available Detectors

- **opencv**: Fast, basic Haar cascades
- **ssd**: Single Shot Detector (good balance)
- **mtcnn**: Multi-task CNN (accurate, recommended)
- **retinaface**: State-of-the-art detector
- **mediapipe**: Google's solution
- **yolov8**: YOLO-based detector
- **yunet**: OpenCV's DNN-based detector

## 📖 Usage Guide

### Setting Up Watchlist

1. Create folders in the `watchlist/` directory
2. Name each folder after the person (e.g., `watchlist/John Doe/`)
3. Add clear, front-facing photos of each person
4. Optionally, add a `.txt` file with the person's description:
   - `watchlist/John Doe/John Doe.txt`

**Example Structure:**
```
watchlist/
├── John Doe/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── John Doe.txt
└── Jane Smith/
    ├── photo1.jpg
    └── Jane Smith.txt
```

### Video Source Options

1. **Camera**: Capture from default webcam or USB camera
2. **Screen**: Capture entire screen or specific window
3. **Video File**: Load and process video files (MP4, AVI, MOV, MKV)

### Keyboard Shortcuts

- **Ctrl+D**: Clear all detected faces
- **Ctrl+Shift+D**: Clear all history (database + photos)
- **Ctrl+L**: Clear logs
- **Ctrl+X**: Exit application

## 🗂️ Data Storage

| Type | Location | Description |
|------|----------|-------------|
| **Settings** | `settings.json` | Application configuration |
| **Watchlist** | `watchlist/` | Known faces and descriptions |
| **Detections** | `detections/` | Saved detection snapshots |
| **History DB** | `detections.db` | SQLite database with detection records |

## 🛠️ Troubleshooting

### Installation Issues

**Problem**: Virtual environment creation fails
- **Solution**: Ensure Python 3.12 is installed and in PATH
- Run: `python --version` to verify

**Problem**: TensorFlow import errors
- **Solution**: Verify `TF_USE_LEGACY_KERAS=1` is set
- Run `diagnostic.bat` to check

**Problem**: Dlib-related errors
- **Solution**: Switch to another detector (mtcnn, ssd) in Settings
- Dlib requires CMake and C++ compiler on Windows

### Runtime Issues

**Problem**: Low FPS / Performance issues
- Increase `PROCESS_INTERVAL` (e.g., 5-7)
- Decrease `DETECTION_SCALE` (e.g., 0.5)
- Reduce `MAX_FACES_TO_CHECK`
- Switch to "Fast" preset

**Problem**: No faces detected
- Ensure watchlist photos are clear and front-facing
- Adjust `SIMILARITY_THRESHOLD` (increase for stricter matching)
- Try different detector (mtcnn recommended)

**Problem**: Camera/Screen not working
- Check device permissions
- Verify camera is not in use by another application
- For screen capture, ensure window is visible

**Problem**: History/Detections not saving
- Check write permissions for `detections/` folder
- Verify `detections.db` is not locked

### Git Issues

**Problem**: `Repository not found` on push
- Verify remote URL: `git remote -v`
- Check GitHub account permissions

## 🎯 Performance Optimization

### For Weak Hardware
- Use "Fast" preset
- Set `PROCESS_INTERVAL` to 5-7
- Set `DETECTION_SCALE` to 0.5
- Set `MAX_FACES_TO_CHECK` to 1-2
- Use `opencv` detector

### For Best Accuracy
- Use "Balanced" preset
- Set `SIMILARITY_THRESHOLD` to 0.35-0.4
- Use `mtcnn` or `retinaface` detector
- Use `Facenet512` or `ArcFace` model

### For Live Streaming
- Use "Streaming" preset
- Enable `ASYNC_PROCESSING`
- Set `PROCESS_INTERVAL` to 2-3

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **DeepFace**: Face recognition library by Serengil
- **OpenCV**: Computer vision library
- **PySide6**: Qt for Python
- **TensorFlow**: Machine learning framework

## 📞 Support

For issues and questions:
1. Check the Troubleshooting section
2. Run `diagnostic.bat` to verify installation
3. Review application logs in the Logs tab
4. Check `detections.db` and file permissions

---

**Version**: 1.01  
**Last Updated**: 2024

