"""Main application window with cozy warm theme.

Layout:
- Title bar
- Control bar (Start/Stop, Mode toggle, FPS, Exit)
- Main split area:
    - Left: Camera feed (always visible)
    - Right: Stacked panel — Running mode (result + landmarks stacked vertically)
                            OR Data collection (gesture grid)
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from camera_thread import CameraThread
from config.gesture_config import GestureConfig
from data_collector import DataCollector, DataCollectorWidget
from inference_thread import InferenceThread
from landmark_panel import LandmarkPanel
from model_adapter import ModelAdapter
from result_widget import ResultWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Recognition")
        self.resize(1250, 800)
        self.setMinimumSize(1000, 700)

        # ---- workers ---------------------------------------------------
        self.camera_thread = CameraThread(camera_index=0, width=800, height=600)
        self.gesture_config = GestureConfig()
        self.adapter = ModelAdapter(use_mediapipe=True, stable_frames=3)
        self.data_collector = DataCollector(config=self.gesture_config)
        self.inference_thread = InferenceThread(
            camera_thread=self.camera_thread, adapter=self.adapter, fps=30.0
        )

        # ---- UI --------------------------------------------------------
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # Title
        title = QLabel("Hand Gesture Recognition")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # Control bar
        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("Start Camera")
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.setEnabled(False)
        self.btn_mode = QPushButton("📊 Data Collection Mode")
        self.btn_mode.setObjectName("btnMode")
        self.btn_mode.setCheckable(True)
        self.btn_mode.setChecked(False)
        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setObjectName("btnExit")

        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch()
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("fpsLabel")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.fps_label.setStyleSheet(
            "color: #5a4d43; font-size: 12px; font-weight: 500; background: transparent;"
        )
        ctrl.addWidget(self.fps_label)
        ctrl.addWidget(self.btn_mode)
        ctrl.addWidget(self.btn_exit)
        root.addLayout(ctrl)

        # ---- Main split area: Camera left, panel right ----------------
        split = QHBoxLayout()
        split.setSpacing(10)
        split.setContentsMargins(0, 0, 0, 0)

        # Left: Camera feed
        camera_container = QWidget()
        camera_container.setObjectName("cameraContainer")
        camera_layout = QVBoxLayout(camera_container)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(8)

        self.camera_label = QLabel("Camera off")
        self.camera_label.setObjectName("cameraLabel")
        self.camera_label.setFixedSize(800, 600)
        self.camera_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        camera_layout.addWidget(self.camera_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        camera_layout.addStretch()

        split.addWidget(camera_container, stretch=0)

        # Right: Stacked panel (Running mode vs Data collection)
        self.right_stack = QStackedWidget()
        self.right_stack.setObjectName("rightStack")

        # Running mode panel — result on top, landmark panel below (vertical)
        self.running_panel = QWidget()
        running_layout = QVBoxLayout(self.running_panel)
        running_layout.setContentsMargins(0, 0, 0, 0)
        running_layout.setSpacing(10)

        self.result_widget = ResultWidget(config=self.gesture_config)
        self.result_widget.setFixedHeight(120)
        running_layout.addWidget(self.result_widget)

        self.landmark_panel = LandmarkPanel()
        self.landmark_panel.setMinimumHeight(140)
        running_layout.addWidget(self.landmark_panel)

        running_layout.addStretch()

        self.right_stack.addWidget(self.running_panel)

        # Data collection panel
        self.data_collector_widget = DataCollectorWidget(
            self.data_collector
        )
        self.right_stack.addWidget(self.data_collector_widget)

        split.addWidget(self.right_stack, stretch=1)

        root.addLayout(split, stretch=1)

        # ---- signal wiring --------------------------------------------
        self.btn_start.clicked.connect(self._start_camera)
        self.btn_stop.clicked.connect(self._stop_camera)
        self.btn_exit.clicked.connect(self.close)

        # Annotated frame (with landmarks) goes to camera display
        self.inference_thread.annotated_frame_ready.connect(self._update_frame)
        self.inference_thread.result_ready.connect(self.result_widget.set_result)
        self.inference_thread.landmarks_ready.connect(self.landmark_panel.set_landmarks)
        self.inference_thread.landmarks_ready.connect(self.data_collector_widget.update_landmarks)
        self.data_collector.saved.connect(self._on_data_saved)
        self.camera_thread.error_occurred.connect(self._show_error)
        self.inference_thread.error_occurred.connect(self._show_error)

        # Mode toggle
        self.btn_mode.clicked.connect(self._toggle_mode)
        self._toggle_mode()  # set initial state

        # Status bar
        self.statusBar().showMessage("Ready. Click Start Camera.")

        # FPS counter
        self._frame_count = 0
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps_display)

    # ------------------------------------------------------------------ #
    # Camera / inference control
    # ------------------------------------------------------------------ #
    def _start_camera(self) -> None:
        self.camera_thread.start()
        self.inference_thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._frame_count = 0
        self._fps_timer.start(1000)  # update every second
        self.fps_label.setText("FPS: 0")
        self.statusBar().showMessage("Camera running...")

    def _stop_camera(self) -> None:
        self.camera_thread.stop()
        self.inference_thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._fps_timer.stop()
        self.camera_label.clear()
        self.camera_label.setText("Camera off")
        self.result_widget.set_result("--", 0.0)
        self.landmark_panel.clear()
        self.fps_label.setText("FPS: --")
        self.statusBar().showMessage("Camera stopped.")

    def _on_data_saved(self, gesture: str, filepath: str) -> None:
        self.data_collector_widget._refresh_counts()
        self.statusBar().showMessage(f"Data saved: {gesture}", 3000)

    def _toggle_mode(self) -> None:
        """Switch between running (recognition) and data collection modes."""
        is_data_mode = self.btn_mode.isChecked()

        if is_data_mode:
            self.btn_mode.setText("👁️ Running Mode")
            self.right_stack.setCurrentWidget(self.data_collector_widget)
            self.statusBar().showMessage("Data collection mode — click gestures to save frames")
        else:
            self.btn_mode.setText("📊 Data Collection Mode")
            self.right_stack.setCurrentWidget(self.running_panel)
            self.statusBar().showMessage("Running mode — camera recognizes gestures")

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _update_frame(self, q_image) -> None:
        self._frame_count += 1
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.camera_label.setPixmap(pixmap)

    def _update_fps_display(self) -> None:
        fps = self._frame_count
        self._frame_count = 0
        self.fps_label.setText(f"FPS: {fps}")
        self.statusBar().showMessage(f"Camera running — {fps} FPS")

    def _show_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Error: {message}", 5000)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:  # noqa: N802
        self.adapter.close()
        self.camera_thread.stop()
        self.inference_thread.stop()
        super().closeEvent(event)
