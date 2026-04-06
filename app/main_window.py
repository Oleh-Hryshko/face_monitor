"""
Face Monitor - GUI Version using PySide6
OpenCV only for video processing, PySide6 for GUI
Uses virtual environment for dependencies
"""

# Import configuration
from app.config import config

import os
import sys
import warnings
import time
import threading
from collections import OrderedDict
from datetime import datetime
import winsound  # for Windows

# Add venv site-packages to path if needed
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    # We're in a virtual environment
    pass

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # Important for compatibility
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

import cv2
from deepface import DeepFace

from PySide6.QtCore import (
    Qt, QTimer, Slot
)
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QAction, QIcon
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QFrame,
    QScrollArea, QSizePolicy, QSplitter,
    QMessageBox, QComboBox, QGroupBox, QTabWidget,
    QStatusBar, QProgressBar, QFileDialog, QTextEdit, QPushButton,
)

from app.ui.logging import LogStream
from app.ui.history_panel import HistoryPanel
from app.ui.detected_face_widget import DetectedFaceWidget
from app.ui.watchlist_panel import StopListPanel
from app.ui.settings_panel import SettingsPanel
from app.ui.video_display_label import VideoDisplayLabel
from app.config.settings_store import load_saved_settings, save_settings_to_file
from app.recognition.processor import FaceProcessor
from app.video.thread import VideoThread
from app.storage.db import (
    clear_detections,
    init_db,
)
from app.storage.async_saver import AsyncSaver
from app.storage.files import imread_utf8, DETECTIONS_DIR
from app.video.win_capture import list_capture_windows

app_version = "1.01"

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        init_db()
        
        self.async_saver = AsyncSaver()
        self.async_saver.saved_signal.connect(self.on_new_detection_saved)
        
        self.setWindowIcon(QIcon('icon.png'))
        self.setWindowTitle("Face Monitor - Watchlist Detection")
        self.setMinimumSize(640, 480)
        
        # Load watchlist data
        self.watchlist_data = {}
        self.watchlist_images = {}
        self.watchlist_info = {}
        self.detected_faces = OrderedDict()
        self.detected_widgets = {}
        
        # Add sound cooldown tracking
        self.last_alert_time = 0
        self.alert_cooldown = 1.0  # seconds between alerts
        self._ignore_detections_until = 0.0
        self._screen_capture_hwnd = None
        self._screen_window_hwnds = [None]
        self._last_video_path = None

        # Load watchlist
        self.load_watchlist()
        
        # Setup UI
        self.setup_ui()
        self.create_menu()
        self.create_status_bar()
        
        # Redirect stdout to Logs tab (keep original for console)
        self._log_stream = LogStream(getattr(sys, "__stdout__", sys.stdout))
        self._log_stream.new_text.connect(self._append_log)
        sys.stdout = self._log_stream
        
        # Setup video thread and processor
        self.processor = FaceProcessor(
            self.watchlist_data,
            self.watchlist_images,
            self.watchlist_info
        )
        self.thread = VideoThread()
        self.thread.set_processor(self.processor)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.detection_signal.connect(self.handle_detections)
        self.thread.fps_signal.connect(self.update_fps)
        
        # Start video (saved settings already applied via load_saved_settings in main())
        self.thread.start()
        mode = "async" if config.ASYNC_PROCESSING else "sync"
        print(f"Video started ({mode}). Detector: {config.DETECTOR}, Model: {config.MODEL_NAME}, Threshold: {config.SIMILARITY_THRESHOLD}")
        if not self.watchlist_data:
            print("WARNING: Watchlist is empty. Add photos to the watchlist folder.")
        
        self.source_combo.setCurrentText("Camera")
        
        QTimer.singleShot(100, self.ensure_video_visible)

    def on_new_detection_saved(self, name, timestamp, path, detection_id, source):
        self.history_panel.add_row(name, timestamp, path, source, detection_id)

    def _detection_source_label(self):
        """Label stored in DB/history: Screen → window title; Video File → file basename."""
        mode = self.source_combo.currentText()
        if mode == "Screen" and hasattr(self, "screen_window_combo"):
            try:
                label = self.screen_window_combo.currentText().strip()
                return label if label else mode
            except Exception:
                return mode
        if mode == "Video File":
            vp = getattr(self, "_last_video_path", None)
            if vp:
                return os.path.basename(vp)
            return mode
        return mode
    
    def play_alert_sound(self):
        """Plays alert sound using settings from config"""
        # Check cooldown to avoid too many sounds
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.last_alert_time = current_time
        
        def sound_thread():
            try:
                for freq, duration in config.ALERT_SOUNDS:
                    winsound.Beep(freq, duration)
                    time.sleep(config.ALERT_SOUND_DELAY)
            except Exception as e:
                # Fallback to terminal bell
                print("\a" * len(config.ALERT_SOUNDS))
                if config.DEBUG_MODE:
                    print(f"Sound error: {e}")
                
        threading.Thread(target=sound_thread, daemon=True).start()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top bar with title and stats
        top_bar = self.create_top_bar()
        main_layout.addWidget(top_bar)
        
        # Content area with splitter — stretch=1 to take up all vertical space under the top_bar
        content_splitter = QSplitter(Qt.Horizontal)
        self.content_splitter = content_splitter
        content_splitter.setHandleWidth(5)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #3d3d3d;
            }
            QSplitter::handle:hover {
                background-color: #555;
            }
        """)
        
        # Left side - Video display
        video_widget = QWidget()
        video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_widget.setStyleSheet("background-color: #1e1e1e;")
        
        # Using QVBoxLayout for video
        video_layout = QVBoxLayout(video_widget)
        video_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create a container for video with a frame
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.Box)
        video_frame.setLineWidth(2)
        video_frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
            }
        """)
        video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Layout for frame — stretch=1 so the video takes up all space in the left frame
        frame_layout = QVBoxLayout(video_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        
        # Video: VideoDisplayLabel + Ignored — layout doesn't "cling" to the pixmap size
        self.video_label = VideoDisplayLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(0, 0)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: none;
            }
        """)
        
        frame_layout.addWidget(self.video_label, 1)
        video_layout.addWidget(video_frame, 1)
        
        content_splitter.addWidget(video_widget)
    
        
        # Right side - Tab widget with panel and settings
        right_tabs = QTabWidget()
        right_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        right_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #252525;
                border-bottom-left-radius: 5px;
                border-bottom-right-radius: 5px;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #ccc;
                padding: 8px 15px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3d3d3d;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #353535;
            }
        """)
        
        # Detected faces panel
        self.detected_panel = self.create_detected_panel()
        right_tabs.addTab(self.detected_panel, "📋 Detected Faces")
        
        # History panel
        self.history_panel = HistoryPanel()
        right_tabs.addTab(self.history_panel, "📜 History")
        
        # Watchlist panel - second tab
        self.watchlist_panel = StopListPanel(self)
        right_tabs.addTab(self.watchlist_panel, "📁 Watchlist")
        
        # Settings panel
        self.settings_panel = SettingsPanel()
        self.settings_panel.settings_applied.connect(self.apply_settings)
        right_tabs.addTab(self.settings_panel, "⚙️ Settings")
        
        # Logs tab
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                font-size: 9pt;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
            }
        """)
        self.log_text_edit.setPlaceholderText("Application log will appear here...")
        right_tabs.addTab(self.log_text_edit, "📋 Logs")
        
        # Info panel
        info_panel = self.create_info_panel()
        right_tabs.addTab(info_panel, "ℹ️ Info")
        
        content_splitter.addWidget(right_tabs)
        
        # Set initial splitter sizes (70% video, 30% panel)
        content_splitter.setSizes([850, 400])
        
        # Set stretch factors (1:1 - both widgets stretch proportionally)
        content_splitter.setStretchFactor(0, 1)  # video_widget - stretches
        content_splitter.setStretchFactor(1, 1)  # right_tabs - stretches
        
        main_layout.addWidget(content_splitter, 1)
        
        central_widget.setLayout(main_layout)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Initialize watchlist display
        self.update_watchlist_display()
    
    def update_watchlist_display(self):
        """Update the watchlist panel with current data"""
        if hasattr(self, 'watchlist_panel'):
            self.watchlist_panel.update_display(self.watchlist_data, self.watchlist_images, self.watchlist_info)

    def load_person_info(self, person_name, person_path):
        """Load txt description for person"""
        # if directory
        if os.path.isdir(person_path):
            txt_path = os.path.join(person_path, f"{person_name}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        
        # if file (for single photo without folder)
        txt_path = person_path + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        
        return ""
        
    def create_top_bar(self):
        """Create top bar with title and stats"""
        bar = QFrame()
        bar.setFixedHeight(60)
        bar.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border-bottom: 2px solid #3d3d3d;
            }
            QLabel {
                color: #ccc;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 0, 15, 0)
        
        # Logo/Title
        title_layout = QHBoxLayout()
        
        title = QLabel("🔍 Face monitor")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title.setFont(title_font)
        title.setStyleSheet("color: #ffaa00;")
        title_layout.addWidget(title)
        
        # Version
        version = QLabel(f"v{app_version} (GUI)")
        version.setStyleSheet("color: #666; font-size: 10px; padding-left: 5px;")
        title_layout.addWidget(version)
        
        layout.addLayout(title_layout)
        
        # Stats
        self.stats_label = QLabel(f"👥 Watchlist: {len(self.watchlist_data)} | 🎯 Threshold: {config.SIMILARITY_THRESHOLD}")
        self.stats_label.setStyleSheet("color: #888; padding-left: 20px;")
        layout.addWidget(self.stats_label)
        
        layout.addStretch()
        
        # FPS display
        self.fps_label = QLabel("📊 FPS: --")
        self.fps_label.setStyleSheet("color: #88ff88; font-weight: bold; padding-right: 15px;")
        layout.addWidget(self.fps_label)
        
        ##Preset selector
        #preset_label = QLabel("Preset:")
        #preset_label.setStyleSheet("color: #888;")
        #layout.addWidget(preset_label)
        
        #self.preset_combo = QComboBox()
        #presets = ["fast", "streaming", "balanced"]
        #self.preset_combo.addItems(presets)
        #self.preset_combo.blockSignals(True)
        #self.preset_combo.setCurrentText("balanced")
        #self.preset_combo.blockSignals(False)
        #self.preset_combo.currentTextChanged.connect(self.apply_preset)
        #layout.addWidget(self.preset_combo)

        #Source selector
        preset_label = QLabel("Source:")
        preset_label.setStyleSheet("color: #888;")
        layout.addWidget(preset_label)
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Camera", "Screen", "Video File"])
        self.source_combo.currentTextChanged.connect(self.change_source)
        layout.addWidget(self.source_combo)

        self.screen_window_label = QLabel("Window:")
        self.screen_window_label.setStyleSheet("color: #888;")
        self.screen_window_label.setVisible(False)
        layout.addWidget(self.screen_window_label)

        self.screen_window_combo = QComboBox()
        self.screen_window_combo.setMinimumWidth(240)
        self.screen_window_combo.setVisible(False)
        self.screen_window_combo.currentIndexChanged.connect(self.on_screen_window_changed)
        layout.addWidget(self.screen_window_combo)

        self.screen_window_refresh = QPushButton("Refresh")
        self.screen_window_refresh.setVisible(False)
        self.screen_window_refresh.setToolTip("Refresh list of windows")
        self.screen_window_refresh.clicked.connect(self.populate_screen_window_combo)
        layout.addWidget(self.screen_window_refresh)

        self.video_file_label = QLabel("File:")
        self.video_file_label.setStyleSheet("color: #888;")
        self.video_file_label.setVisible(False)
        layout.addWidget(self.video_file_label)

        self.video_file_name_label = QLabel("")
        self.video_file_name_label.setStyleSheet("color: #ccc;")
        self.video_file_name_label.setMinimumWidth(200)
        self.video_file_name_label.setMaximumWidth(420)
        self.video_file_name_label.setVisible(False)
        layout.addWidget(self.video_file_name_label)

        self.video_file_browse_btn = QPushButton("…")
        self.video_file_browse_btn.setFixedWidth(30)
        self.video_file_browse_btn.setVisible(False)
        self.video_file_browse_btn.setToolTip("Choose video file")
        self.video_file_browse_btn.clicked.connect(self.open_video_file)
        layout.addWidget(self.video_file_browse_btn)

        bar.setLayout(layout)
        return bar
    
    def create_detected_panel(self):
        """Create panel for displaying detected faces"""
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        panel.setStyleSheet("background-color: #252525;")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Header with count
        header_layout = QHBoxLayout()
        
        header = QLabel("DETECTED")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(11)
        header.setFont(header_font)
        header.setStyleSheet("color: #ffaa00; padding: 5px;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        self.count_badge = QLabel("0")
        self.count_badge.setAlignment(Qt.AlignCenter)
        self.count_badge.setFixedSize(24, 24)
        self.count_badge.setStyleSheet("""
            QLabel {
                background-color: #ffaa00;
                color: #1a1a1a;
                border-radius: 12px;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        header_layout.addWidget(self.count_badge)
        
        layout.addLayout(header_layout)
        
        # Scroll area for faces
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        # Container for face widgets
        self.faces_container = QWidget()
        self.faces_container.setObjectName("faces_container")
        self.faces_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.faces_container.setStyleSheet("""
            QWidget#faces_container {
                background-color: #1e1e1e;
            }
        """)
        
        # Use QVBoxLayout for the container
        self.faces_layout = QVBoxLayout(self.faces_container)
        self.faces_layout.setContentsMargins(5, 5, 5, 5)
        self.faces_layout.setSpacing(5)
        self.faces_layout.setAlignment(Qt.AlignTop)
        
        self.faces_container.setLayout(self.faces_layout)
        
        self.scroll_area.setWidget(self.faces_container)
        layout.addWidget(self.scroll_area)
        
        panel.setLayout(layout)
        return panel
    
    def create_info_panel(self):
        """Create info panel with system information"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # System info
        info_group = QGroupBox("System Information")
        info_group.setStyleSheet("""
            QGroupBox {
                color: #ffaa00;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)
        
        # Python version
        py_label = QLabel(f"🐍 Python: {sys.version.split()[0]}")
        py_label.setStyleSheet("color: #ccc; padding: 2px; font-size: 10px;")
        info_layout.addWidget(py_label)
        
        # OpenCV version
        cv_label = QLabel(f"📷 OpenCV: {cv2.__version__}")
        cv_label.setStyleSheet("color: #ccc; padding: 2px; font-size: 10px;")
        info_layout.addWidget(cv_label)
        
        # DeepFace info
        info_layout.addWidget(QLabel("🤖 DeepFace: loaded"))
        
        # Watchlist info
        info_layout.addWidget(QLabel(f"📁 Watchlist: {config.WATCHLIST_PATH}"))
        
        # Model info
        info_layout.addWidget(QLabel(f"🎯 Model: {config.MODEL_NAME}"))
        info_layout.addWidget(QLabel(f"🔍 Detector: {config.DETECTOR}"))
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Help
        help_group = QGroupBox("Keyboard Shortcuts")
        help_group.setStyleSheet("""
            QGroupBox {
                color: #ffaa00;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        help_layout = QVBoxLayout()
        help_layout.setSpacing(3)
        help_layout.addWidget(QLabel("• Ctrl+D - Clear all detections"))
        help_layout.addWidget(QLabel("• Ctrl+Shift+D - Clear All History"))
        help_layout.addWidget(QLabel("• Ctrl+L - Clear Logs"))
        help_layout.addWidget(QLabel("• Ctrl+X - Exit"))
        
        help_group.setLayout(help_layout)
        layout.addWidget(help_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_menu(self):
        """Create application menu"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #1a1a1a;
                color: #ccc;
                border-bottom: 1px solid #333;
            }
            QMenuBar::item:selected {
                background-color: #333;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #ccc;
                border: 1px solid #444;
            }
            QMenu::item:selected {
                background-color: #3d3d3d;
            }
        """)
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        #open_action = QAction("Open Video File...", self)
        #open_action.triggered.connect(self.open_video_file)
        #file_menu.addAction(open_action)
        
        #file_menu.addSeparator()
        
        quit_action = QAction("Exit", self)
        quit_action.setShortcut("Ctrl+X")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Detection menu
        detection_menu = menubar.addMenu("Detection")
        
        clear_action = QAction("Clear All Detections", self)
        clear_action.setShortcut("Ctrl+D")
        clear_action.triggered.connect(self.clear_all_detections)
        detection_menu.addAction(clear_action)

        clear_history_action = QAction("Clear All History", self)
        clear_history_action.setShortcut("Ctrl+Shift+D")
        clear_history_action.triggered.connect(self.clear_all_history)
        detection_menu.addAction(clear_history_action)
    
        clear_logs_action = QAction("Clear Logs", self)
        clear_logs_action.setShortcut("Ctrl+L")
        clear_logs_action.triggered.connect(self.clear_logs)
        detection_menu.addAction(clear_logs_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def open_watchlist_folder(self):
        """Open the watchlist folder in file explorer"""
        watchlist_path = config.WATCHLIST_PATH
        if os.path.exists(watchlist_path):
            os.startfile(watchlist_path)
        else:
            QMessageBox.warning(self, "Folder Not Found", f"Watchlist folder not found: {watchlist_path}")
    
    def create_status_bar(self):
        """Create status bar"""
        status_bar = QStatusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #1a1a1a;
                color: #888;
                border-top: 1px solid #333;
            }
        """)
        self.setStatusBar(status_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)
        
        # Progress bar for loading
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumSize(100, 15)
        self.progress_bar.setVisible(False)
        status_bar.addPermanentWidget(self.progress_bar)
    
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ccc;
            }
            QSplitter::handle {
                background-color: #3d3d3d;
            }
            QSplitter::handle:hover {
                background-color: #555;
            }
            QLabel {
                color: #ccc;
            }
        """)
    
    def load_watchlist(self):
        """Load watchlist data with UTF-8 path support"""
        print("Loading watchlist...")
        self.watchlist_data.clear()
        self.watchlist_images.clear()
        watchlist_path = config.WATCHLIST_PATH

        if not os.path.exists(watchlist_path):
            os.makedirs(watchlist_path)
            print(f"Created folder {watchlist_path}")
            return

        for root, dirs, files in os.walk(watchlist_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    
                    # Get person name
                    rel_path = os.path.relpath(root, watchlist_path)
                    if rel_path == '.':
                        person_name = os.path.splitext(file)[0]
                    else:
                        person_name = rel_path
                    
                    if person_name not in self.watchlist_info:
                        info = self.load_person_info(person_name, root)
                        self.watchlist_info[person_name] = info

                    try:
                        # Read image with UTF-8 support
                        img_array = imread_utf8(img_path)
                        if img_array is None:
                            print(f"  Failed to read: {img_path}")
                            continue

                        # Calculate embedding using image array instead of file path
                        embedding = DeepFace.represent(
                            img_path=img_array,  # Pass image array instead of path
                            model_name=config.MODEL_NAME,
                            detector_backend=config.DETECTOR,
                            enforce_detection=config.ENFORCE_DETECTION
                        )[0]["embedding"]
                        
                        if person_name not in self.watchlist_data:
                            self.watchlist_data[person_name] = []
                            self.watchlist_images[person_name] = []
                        
                        self.watchlist_data[person_name].append({
                            'file': file,
                            'embedding': embedding,
                            'path': img_path
                        })
                        self.watchlist_images[person_name].append(img_path)
                        
                        print(f"  Loaded: {person_name} - {file}")
                        
                    except Exception as e:
                        err_msg = str(e)
                        if "dlib" in err_msg.lower() or "no module named 'dlib'" in err_msg.lower():
                            QMessageBox.warning(
                                self,
                                "Dlib not installed",
                                "The Dlib model requires the dlib package.\n\n"
                                "Install with: pip install dlib\n\n"
                                "Note: On Windows, dlib installation may require CMake and a C++ compiler."
                            )
                            print(f"Error loading {img_path}: {e}")
                            print("Loaded 0 person(s) — switch to another model in Settings.")
                            return
                        print(f"Error loading {img_path}: {e}")
        
        proc = getattr(self, "processor", None)
        if proc is not None:
            proc.invalidate_watchlist_index()

        print(f"Loaded {len(self.watchlist_data)} person(s)")

    def load_person_info(self, person_name, person_path):
        """Load txt description for person"""

        # if directory
        if os.path.isdir(person_path):
            txt_path = os.path.join(person_path, f"{person_name}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    return f.read().strip()

        # if file
        txt_path = person_path + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()

        return ""
    
    @Slot(QImage)
    def update_image(self, image):
        """Update video display with proper scaling and centering"""
        # Get video_label size
        label_size = self.video_label.size()
        
        if label_size.width() <= 0 or label_size.height() <= 0:
            # If size is not yet determined, use parent widget size
            parent_size = self.video_label.parent().size()
            if parent_size.width() > 0 and parent_size.height() > 0:
                label_size = parent_size
            else:
                return
        
        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(image)
        
        # Save original for resizeEvent
        self._original_pixmap = pixmap
        
        # Calculate scale to preserve aspect ratio
        scaled_pixmap = pixmap.scaled(
            label_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # Set scaled image
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)
        # QLabel with pixmap sets a large minimumSizeHint — reset so layout stretches the widget
        self.video_label.setMinimumSize(0, 0)
        
        # Force update
        self.video_label.update()
    
    @Slot(float)
    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_label.setText(f"📊 FPS: {fps:.1f}")
    
    @Slot(list)
    def handle_detections(self, detections):
        """Handle detection results"""
        if time.time() < self._ignore_detections_until:
            return

        current_time = datetime.now()
        new_detections = False
        
        for detection in detections:
            name = detection.get('name')
            
            if not name:
                continue
            
            if name not in self.detected_faces:
                # New detection
                print(f"    NEW DETECTION: {name}")
                
                self.detected_faces[name] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'count': 1,
                    'face_img': detection.get('face_img'),
                    'thumb': detection.get('thumb')
                }
                
                info = detection.get('info', '')

                widget = DetectedFaceWidget(
                    name,
                    detection.get('face_img'),
                    detection.get('thumb'),
                    current_time,
                    info,
                    self.faces_container
                )
                
                # Connect the remove signal
                widget.remove_clicked.connect(self.remove_detected_face)
                
                # Add widget to layout
                self.faces_layout.addWidget(widget)
                self.detected_widgets[name] = widget
                
                # Make widget visible
                widget.show()
                widget.update()
                
                new_detections = True
                
                # Play alert sound for new detection
                self.play_alert_sound()
                
                # Update status
                self.status_label.setText(f"New violator detected: {name}")
                
                # Save the frame WITH bounding box
                source = self._detection_source_label()
                full_frame = detection.get('full_frame')  # This frame already contains the drawn bounding box
                
                if full_frame is not None:
                    self.async_saver.save(full_frame, name, source)
                    print(f"    Saved full frame with bounding box for {name}")
                else:
                    print(f"    Warning: No frame with bounding box available for {name}")

            else:
                # Update existing
                print(f"    Updating existing: {name}")
                self.detected_faces[name]['last_seen'] = current_time
                self.detected_faces[name]['count'] += 1
        
        # Update count badge
        count = len(self.detected_faces)
        self.count_badge.setText(str(count))
        
        # Update stats
        self.stats_label.setText(
            f"👥 Watchlist: {len(self.watchlist_data)} | "
            f"🎯 Threshold: {config.SIMILARITY_THRESHOLD} | "
            f"🚨 Detected: {count}"
        )
        
        # Force update
        self.faces_container.update()
        self.scroll_area.update()
        QApplication.processEvents()
        
        # Auto-scroll to new detection
        if new_detections and count > 0:
            QTimer.singleShot(100, self.scroll_to_bottom)

    def _suspend_detection_stream(self, guard_ms=900):
        """Pause detection and flush async queues to avoid stale detections reaching UI."""
        if hasattr(self.processor, "pause_detection"):
            self.processor.pause_detection()
        if hasattr(self.thread, "clear_detection_queues"):
            self.thread.clear_detection_queues()
        self._ignore_detections_until = max(self._ignore_detections_until, time.time() + (guard_ms / 1000.0))

    def _resume_detection_stream(self, delay_ms=900):
        """Resume detection after short delay, giving pipeline time to settle."""
        if hasattr(self.processor, "resume_detection"):
            QTimer.singleShot(delay_ms, self.processor.resume_detection)

    @Slot(str)
    def remove_detected_face(self, name):
        """Remove a specific detected face from the panel"""
        if name in self.detected_faces:
            # Ask for confirmation
            reply = QMessageBox.question(
                self, 'Confirm Removal',
                f'Are you sure you want to remove "{name}" from the detected faces list?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Mirror clear-all behavior: pause detection and drop queued stale results.
                self._suspend_detection_stream()

                # Remove from dictionary
                del self.detected_faces[name]
                
                # Remove widget
                if name in self.detected_widgets:
                    widget = self.detected_widgets[name]
                    self.faces_layout.removeWidget(widget)
                    widget.deleteLater()
                    del self.detected_widgets[name]
                
                # Update counter
                count = len(self.detected_faces)
                self.count_badge.setText(str(count))
                
                # Update stats
                self.stats_label.setText(
                    f"👥 Watchlist: {len(self.watchlist_data)} | "
                    f"🎯 Threshold: {config.SIMILARITY_THRESHOLD} | "
                    f"🚨 Detected: {count}"
                )
                
                self.status_label.setText(f"Removed {name} from detected faces")
                print(f"🗑️ Removed {name} from detected faces")

                # Resume detection after guard delay.
                self._resume_detection_stream()
                
                # Force update
                self.faces_container.update()
    
    def update_durations(self):
        """No-op: duration no longer displayed"""
        pass
    
    def scroll_up(self):
        """Scroll detected faces panel up"""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.value() - 50)
    
    def scroll_down(self):
        """Scroll detected faces panel down"""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.value() + 50)
    
    def reset_scroll(self):
        """Reset scroll position to top"""
        self.scroll_area.verticalScrollBar().setValue(0)
        self.status_label.setText("Scroll reset")
    
    def scroll_to_bottom(self):
        """Scroll to bottom of panel"""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    @Slot(str)
    def _append_log(self, text):
        """Append a line to the Logs tab."""
        if hasattr(self, "log_text_edit") and self.log_text_edit:
            self.log_text_edit.append(text)
            scrollbar = self.log_text_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def clear_logs(self):
        """Clear the Logs tab content."""
        reply = QMessageBox.question(
            self, 'Clear Logs',
            'Are you sure you want to clear all logs?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if hasattr(self, "log_text_edit") and self.log_text_edit:
                self.log_text_edit.clear()
            self.status_label.setText("Logs cleared")
            print("🧹 Logs cleared")
    
    def clear_all_detections(self):
        """Clear all detected faces from panel"""
        reply = QMessageBox.question(
            self, 'Clear All',
            'Are you sure you want to clear all detected faces?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Stop accepting/producing detections while clearing UI state.
            self._suspend_detection_stream()

            # Clear data
            self.detected_faces.clear()
            
            # Remove all widgets from layout
            while self.faces_layout.count() > 0:
                item = self.faces_layout.takeAt(0)
                if item and item.widget():
                    item.widget().deleteLater()
            
            self.detected_widgets.clear()
            
            # Update count
            self.count_badge.setText("0")
            self.stats_label.setText(
                f"👥 Watchlist: {len(self.watchlist_data)} | "
                f"🎯 Threshold: {config.SIMILARITY_THRESHOLD} | "
                f"🚨 Detected: 0"
            )
            self.status_label.setText("All detections cleared")
            print("🧹 All detections cleared")
            
            # Force update
            self.faces_container.update()
            
            # Resume detection after a small delay to skip stale in-flight async results.
            self._resume_detection_stream()
            print("▶️ Detection will resume shortly")

    def clear_all_history(self):
        """Clear all history: delete database records and all photos from detections folder"""
        reply = QMessageBox.question(
            self, 'Clear All History',
            '⚠️ WARNING: This will delete ALL detection records from the database\n'
            'and ALL photos from the detections folder.\n\n'
            'This action cannot be undone!\n\n'
            'Are you sure you want to continue?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        try:
            deleted_files = 0
            failed_files = 0
            detections_folder = DETECTIONS_DIR

            if os.path.exists(detections_folder):
                for filename in os.listdir(detections_folder):
                    file_path = os.path.join(detections_folder, filename)
                    if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            os.remove(file_path)
                            deleted_files += 1
                        except Exception as e:
                            failed_files += 1
                            print(f"Failed to delete {file_path}: {e}")

            # Clear database
            deleted_records = clear_detections()

            # Refresh history panel if it exists
            if hasattr(self, 'history_panel'):
                self.history_panel.load_data()
                self.history_panel.image_label.setText("Select row to view image")
                self.history_panel.image_label.setPixmap(QPixmap())

            # Update status
            self.status_label.setText(
                f"History cleared: {deleted_records} record(s) deleted, "
                f"{deleted_files} photo(s) deleted"
            )

            if failed_files > 0:
                self.status_label.setText(
                    f"History cleared: {deleted_records} record(s) deleted, "
                    f"{deleted_files} photo(s) deleted, {failed_files} failed"
                )

            print(f"🗑️ Cleared all history: {deleted_records} records, {deleted_files} photos")

            QMessageBox.information(
                self, 'History Cleared',
                f'Successfully cleared all history!\n\n'
                f'📊 Records deleted: {deleted_records}\n'
                f'🖼️ Photos deleted: {deleted_files}'
            )

        except Exception as e:
            QMessageBox.critical(
                self, 'Error',
                f'Failed to clear history: {str(e)}'
            )
            print(f"Error clearing history: {e}")
            
    def open_video_file(self):
        """Open video file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        
        if file_path:
            self._last_video_path = file_path
            self.screen_window_label.setVisible(False)
            self.screen_window_combo.setVisible(False)
            self.screen_window_refresh.setVisible(False)
            self.video_file_label.setVisible(True)
            self.video_file_name_label.setVisible(True)
            self.video_file_browse_btn.setVisible(True)
            bn = os.path.basename(file_path)
            self.video_file_name_label.setText(bn)
            self.video_file_name_label.setToolTip(file_path)

            self.thread.stop()
            self.thread.wait()

            self.thread = VideoThread()
            self.thread.set_processor(self.processor)
            self.thread.set_video_source(file_path, "video")
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.detection_signal.connect(self.handle_detections)
            self.thread.fps_signal.connect(self.update_fps)
            self.thread.start()

            self.source_combo.blockSignals(True)
            self.source_combo.setCurrentText("Video File")
            self.source_combo.blockSignals(False)

            self.status_label.setText(f"Playing: {bn}")
    
    def apply_preset(self, preset_name):
        """Apply configuration preset"""
        from app.config.config import apply_preset as apply_preset_config
        preset = apply_preset_config(preset_name)
        
        # Reload watchlist so embeddings match the new model
        self.load_watchlist()
        self.update_watchlist_display()
        
        # Update processor settings
        self.processor.model_name = config.MODEL_NAME
        self.processor.detector = config.DETECTOR
        self.processor.threshold = config.SIMILARITY_THRESHOLD
        self.processor.metric = config.DISTANCE_METRIC
        
        # Update display
        self.stats_label.setText(
            f"👥 Watchlist: {len(self.watchlist_data)} | "
            f"🎯 Threshold: {config.SIMILARITY_THRESHOLD} | "
            f"🚨 Detected: {len(self.detected_faces)}"
        )
        
        self.status_label.setText(f"Applied preset: {preset_name}")
        print(f"Applied preset: {preset_name}")
        save_settings_to_file()
    
    def apply_settings(self, settings):
        """Apply new settings from settings panel"""
        old_model = config.MODEL_NAME
        old_detector = config.DETECTOR
        old_metric = config.DISTANCE_METRIC
        old_thumb_size = getattr(config, 'DETECTED_FACE_THUMB_SIZE', 100)

        # Update config
        config.MODEL_NAME = settings['model_name']
        config.DETECTOR = settings['detector']
        config.SIMILARITY_THRESHOLD = settings['threshold']
        config.DISTANCE_METRIC = settings['metric']
        config.PROCESS_INTERVAL = settings['process_interval']
        config.DETECTION_SCALE = settings['detection_scale']
        config.MAX_FACES_TO_CHECK = settings['max_faces']
        config.ASYNC_PROCESSING = settings['async_mode']
        config.EXPAND_FACE_BOX = settings['expand_enabled']
        config.FACE_BOX_EXPAND_FACTOR = settings['expand_factor']
        config.FACE_BOX_HEADROOM = settings['headroom']
        config.DETECTED_FACE_THUMB_SIZE = settings['detected_face_thumb_size']

        # If thumbnail size changed, recreate all widgets
        if old_thumb_size != config.DETECTED_FACE_THUMB_SIZE:
            print(f"Thumbnail size changed from {old_thumb_size} to {config.DETECTED_FACE_THUMB_SIZE}")
            self.recreate_all_face_widgets()

        # Reload watchlist when model or detector changes (embeddings are model-specific)
        if (config.MODEL_NAME != old_model or config.DETECTOR != old_detector):
            try:
                self.load_watchlist()
                self.update_watchlist_display()
            except Exception as e:
                err_msg = str(e)
                if "dlib" in err_msg.lower() or "no module named 'dlib'" in err_msg.lower():
                    QMessageBox.warning(
                        self,
                        "Dlib not installed",
                        "The Dlib model requires the dlib package.\n\n"
                        "Install with: pip install dlib\n\n"
                        "Reverting to previous model."
                    )
                    config.MODEL_NAME = old_model
                    config.DETECTOR = old_detector
                    self.load_watchlist()
                    self.update_watchlist_display()
                    self.settings_panel.model_combo.setCurrentText(old_model)
                    self.settings_panel.detector_combo.setCurrentText(old_detector)

        # Update processor
        self.processor.model_name = config.MODEL_NAME
        self.processor.detector = config.DETECTOR
        self.processor.threshold = settings['threshold']
        self.processor.metric = settings['metric']
        self.processor.expand_enabled = settings['expand_enabled']
        self.processor.expand_factor = settings['expand_factor']
        self.processor.headroom = settings['headroom']
        if settings['metric'] != old_metric:
            self.processor.invalidate_watchlist_index()

        # Update thread (need to restart if async mode changed)
        if getattr(self.thread, 'async_mode', None) != settings['async_mode']:
            self.thread.stop()
            self.thread.wait()
            self.thread = VideoThread()
            self.thread.set_processor(self.processor)
            self._restore_thread_video_source()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.detection_signal.connect(self.handle_detections)
            self.thread.fps_signal.connect(self.update_fps)
            self.thread.start()
        
        # Update display
        self.stats_label.setText(
            f"👥 Watchlist: {len(self.watchlist_data)} | "
            f"🎯 Threshold: {settings['threshold']} | "
            f"🚨 Detected: {len(self.detected_faces)}"
        )
        
        self.status_label.setText("Settings applied")
        print("Settings applied")
        save_settings_to_file()
    
    def recreate_all_face_widgets(self):
        """Recreate all face widgets when thumbnail size changes"""
        if not self.detected_faces:
            return
        
        print("Recreating all face widgets with new thumbnail size...")
        
        # Store current faces data
        faces_data = list(self.detected_faces.items())
        
        # Clear current widgets
        while self.faces_layout.count() > 0:
            item = self.faces_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        
        self.detected_widgets.clear()
        
        # Recreate widgets with new size
        for name, face_data in faces_data:
            info = face_data.get('info', '')

            widget = DetectedFaceWidget(
                name,
                face_data.get('face_img'),
                face_data.get('thumb'),
                face_data.get('first_seen'),
                info,
                self.faces_container
            )
                     
            widget.remove_clicked.connect(self.remove_detected_face)
            self.faces_layout.addWidget(widget)
            self.detected_widgets[name] = widget
            widget.show()
        
        # Force update
        self.faces_container.update()
        QApplication.processEvents()
        
        print(f"Recreated {len(self.detected_widgets)} widgets")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Face Monitor",
            f"<h2>Face Monitor {app_version}</h2>"
            "<p>A face recognition system for monitoring visitors "
            "and detecting persons from a watchlist.</p>"
            "<p><b>OpenCV</b> for video processing<br>"
            "<b>DeepFace</b> for face recognition<br>"
            "<b>PySide6</b> for GUI</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close event"""
        if hasattr(self, 'async_saver'):
            self.async_saver.stop()
        super().closeEvent(event)
        reply = QMessageBox.question(
            self, 'Confirm Exit',
            'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            save_settings_to_file()
            self.thread.stop()
            event.accept()
        else:
            event.ignore()

    def ensure_video_visible(self):
        """Ensure video is visible and can grow with the left panel (never use setFixedSize here)."""
        self.video_label.show()
        self.video_label.setMinimumSize(0, 0)
        self.video_label.setMaximumSize(16777215, 16777215)
        self.video_label.updateGeometry()

    def populate_screen_window_combo(self):
        """Fill window list for Screen capture (Windows)."""
        self.screen_window_combo.blockSignals(True)
        self.screen_window_combo.clear()
        self._screen_window_hwnds = [None]
        self.screen_window_combo.addItem("Entire screen")

        if sys.platform != "win32":
            self.screen_window_combo.setEnabled(False)
            self.screen_window_refresh.setEnabled(False)
            self.screen_window_combo.blockSignals(False)
            self._screen_capture_hwnd = None
            if self.thread is not None:
                self.thread.set_screen_capture_hwnd(None)
            return

        self.screen_window_combo.setEnabled(True)
        self.screen_window_refresh.setEnabled(True)
        try:
            exclude = int(self.winId())
        except Exception:
            exclude = None
        try:
            for w in list_capture_windows(exclude_hwnd=exclude):
                title = w["title"]
                if len(title) > 90:
                    title = title[:87] + "..."
                self.screen_window_combo.addItem(title)
                self._screen_window_hwnds.append(int(w["hwnd"]))
        except Exception as e:
            print(f"Window list error: {e}")

        self.screen_window_combo.blockSignals(False)
        self.on_screen_window_changed(self.screen_window_combo.currentIndex())

    @Slot(int)
    def on_screen_window_changed(self, _index=-1):
        """Use combo currentIndex() — slot index can be stale in some Qt versions."""
        idx = self.screen_window_combo.currentIndex()
        if idx < 0:
            return
        hwnds = getattr(self, "_screen_window_hwnds", [None])
        if idx < len(hwnds):
            hwnd = hwnds[idx]
        else:
            hwnd = None
        self._screen_capture_hwnd = hwnd
        if self.thread is not None:
            self.thread.set_screen_capture_hwnd(hwnd)

    def _restore_thread_video_source(self):
        """Apply Camera / Screen / Video File to a newly created VideoThread from UI state."""
        text = self.source_combo.currentText()
        if text == "Camera":
            self.thread.set_video_source(0, "camera")
        elif text == "Screen":
            self.thread.set_video_source(None, "screen")
            try:
                self.thread.set_exclude_capture_hwnd(int(self.winId()))
            except Exception:
                self.thread.set_exclude_capture_hwnd(None)
            self.thread.set_screen_capture_hwnd(getattr(self, "_screen_capture_hwnd", None))
        elif text == "Video File":
            vp = getattr(self, "_last_video_path", None)
            if vp and os.path.isfile(vp):
                self.thread.set_video_source(vp, "video")
            else:
                self.thread.set_video_source(0, "camera")
    
    def show_screen_placeholder(self):
        """Show placeholder widget for screen mode"""
        # Find video_frame (parent of video_label)
        video_frame = self.video_label.parent()
        if not video_frame:
            return
        
        if not hasattr(self, 'placeholder_label'):
            self.placeholder_label = QLabel(video_frame.parent())
            self.placeholder_label.setAlignment(Qt.AlignCenter)
            self.placeholder_label.setStyleSheet("""
                QLabel {
                    background-color: #1e1e1e;
                    color: #888;
                    font-size: 18px;
                    font-weight: bold;
                    border: 2px solid #3d3d3d;
                    border-radius: 5px;
                }
            """)
        
        # Set text
        self.placeholder_label.setText(
            "📺 SCREEN CAPTURE MODE\n\n"
            "Face detection running in background\n"
            "Detected faces will appear in the right panel"
        )
        
        # Set geometry equal to video_frame geometry
        self.placeholder_label.setGeometry(video_frame.geometry())
        self.placeholder_label.show()
        self.placeholder_label.raise_()
    
    def force_video_resize(self):
        """Force video label to resize properly after source change"""
        if self.video_label.isVisible() and hasattr(self, '_original_pixmap'):
            new_size = self.video_label.size()
            if new_size.width() > 0 and new_size.height() > 0:
                scaled = self._original_pixmap.scaled(
                    new_size, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled)
                self.video_label.setMinimumSize(0, 0)
                self.video_label.update()

    def change_source(self, text):
        # Stop current thread
        self.thread.stop()
        self.thread.wait()

        # Create new thread
        self.thread = VideoThread()
        self.thread.set_processor(self.processor)

        # Hide all special widgets
        if hasattr(self, 'placeholder_label'):
            self.placeholder_label.hide()
        if hasattr(self, 'stats_widget'):
            self.stats_widget.hide()

        self.screen_window_label.setVisible(False)
        self.screen_window_combo.setVisible(False)
        self.screen_window_refresh.setVisible(False)
        self.video_file_label.setVisible(False)
        self.video_file_name_label.setVisible(False)
        self.video_file_browse_btn.setVisible(False)

        # Show video_label and reset size constraints (stretch with left frame)
        self.video_label.show()
        self.video_label.setPixmap(QPixmap())  # Clear current image
        self.video_label.setMinimumSize(0, 0)
        self.video_label.setMaximumSize(16777215, 16777215)
        
        # Force geometry update
        self.video_label.updateGeometry()
        self.video_label.parent().updateGeometry()

        if text == "Camera":
            self.thread.set_video_source(0, "camera")
            self.status_label.setText("Camera mode - showing video")
            print("Switched to Camera")

        elif text == "Screen":
            self.thread.set_video_source(None, "screen")
            try:
                self.thread.set_exclude_capture_hwnd(int(self.winId()))
            except Exception:
                self.thread.set_exclude_capture_hwnd(None)
            self.screen_window_label.setVisible(True)
            self.screen_window_combo.setVisible(True)
            self.screen_window_refresh.setVisible(True)
            self.populate_screen_window_combo()
            self.status_label.setText("Screen capture — preview on the left, pick a window or entire screen")
            print("Switched to Screen")

        elif text == "Video File":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Video File", "",
                "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
            if not file_path:
                # If user canceled selection, revert combo box back
                self.source_combo.setCurrentText("Camera")
                return
            self.thread.set_video_source(file_path, "video")
            self._last_video_path = file_path
            self.video_file_label.setVisible(True)
            self.video_file_name_label.setVisible(True)
            self.video_file_browse_btn.setVisible(True)
            bn = os.path.basename(file_path)
            self.video_file_name_label.setText(bn)
            self.video_file_name_label.setToolTip(file_path)
            self.status_label.setText(f"Playing: {bn}")
            print(f"Switched to Video File: {file_path}")

        # Connect signals
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.detection_signal.connect(self.handle_detections)
        self.thread.fps_signal.connect(self.update_fps)

        # Start new thread
        self.thread.start()
        
        # Force video resize
        QTimer.singleShot(500, self.force_video_resize)
        
        # Force UI update
        QApplication.processEvents()
    
    def resizeEvent(self, event):
        """Handle window resize to update placeholder and video"""
        super().resizeEvent(event)
        
        # Update placeholder size if it's visible
        if hasattr(self, 'placeholder_label') and self.placeholder_label.isVisible():
            # Find video_frame (parent of video_label)
            video_frame = self.video_label.parent()
            if video_frame:
                self.placeholder_label.setGeometry(video_frame.geometry())
                self.placeholder_label.show()
        
        # If video is visible and there's an original image, update its size
        if (self.video_label.isVisible() and 
            hasattr(self, '_original_pixmap') and 
            self._original_pixmap and 
            not self._original_pixmap.isNull()):
            
            # Get new size
            new_size = self.video_label.size()
            
            if new_size.width() > 0 and new_size.height() > 0:
                # Scale original to new size
                scaled = self._original_pixmap.scaled(
                    new_size, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled)
                self.video_label.setMinimumSize(0, 0)
                self.video_label.update()
        
        QApplication.processEvents()

def main():
    """Main entry point"""
    load_saved_settings()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application-wide font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Set application icon (optional)
    app.setWindowIcon(QIcon())
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

