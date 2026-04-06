# Face Monitor

## Entry Points

- Primary launcher: `face_monitor.py`
- Unified app launcher: `app/launcher.py`
- Main window module: `app/main_window.py`

Both launchers call the same `main()` function from `app/main_window.py`.

## Current Module Layout

- `app/main_window.py` - main `MainWindow` class and app startup function.
- `app/launcher.py` - unified launcher for module-based start.
- `app/config/settings_store.py` - load/save app settings (`settings.json`).
- `app/storage/db.py` - detection database operations.
- `app/storage/files.py` - file utilities (`save_frame`, `imread_utf8`, transliteration).
- `app/storage/async_saver.py` - background saving pipeline for detections.
- `app/recognition/processor.py` - face detection/recognition pipeline (`FaceProcessor`).
- `app/video/thread.py` - video capture and frame processing thread (`VideoThread`).
- `app/ui/history_panel.py` - history tab UI.
- `app/ui/detected_face_widget.py` - detected-face card widget.
- `app/ui/watchlist_panel.py` - watchlist tab UI.
- `app/ui/settings_panel.py` - settings tab UI.
- `app/ui/video_display_label.py` - resizable video label widget.
- `app/ui/overlay.py` - UI drawing/overlay helpers.
- `app/ui/logging.py` - stdout-to-Qt log bridge (`LogStream`).

## Run

```powershell
python face_monitor.py
```

Alternative launcher:

```powershell
python -m app.launcher
```

## Face Crop Utility

Legacy usage examples are kept in `readme.txt`.

