"""Custom QGraphicsView for rendering audio waveforms with dysfluency overlays."""

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from app.ui.theme import COLORS


class WaveformView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._audio = None
        self._sample_rate = 16000
        self._overlays = []
        self._resize_timer = None
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(120)

    def set_audio(self, audio: np.ndarray, sample_rate: int = 16000):
        """Set the audio data to display."""
        self._audio = audio
        self._sample_rate = sample_rate
        self._overlays = []
        self._draw()

    def set_overlays(self, overlays: list[tuple[float, float, str]]):
        """Set colored overlay regions. Each tuple: (start_sec, end_sec, color_hex)."""
        self._overlays = overlays
        self._draw()

    def clear_overlays(self):
        """Remove all overlays."""
        self._overlays = []
        self._draw()

    def _draw(self):
        """Redraw the waveform and overlays."""
        self._scene.clear()
        if self._audio is None or len(self._audio) == 0:
            return

        view_width = self.viewport().width()
        view_height = self.viewport().height()
        self._scene.setSceneRect(0, 0, view_width, view_height)

        duration = len(self._audio) / self._sample_rate
        mid_y = view_height / 2

        # Draw overlays first (behind waveform)
        for start_sec, end_sec, color_hex in self._overlays:
            x1 = (start_sec / duration) * view_width
            x2 = (end_sec / duration) * view_width
            color = QColor(color_hex)
            color.setAlpha(80)
            rect = QRectF(x1, 0, x2 - x1, view_height)
            self._scene.addRect(rect, QPen(Qt.PenStyle.NoPen), QBrush(color))

        # Downsample audio for drawing
        num_samples = len(self._audio)
        num_points = min(num_samples, view_width * 2)
        indices = np.linspace(0, num_samples - 1, num_points, dtype=int)
        samples = self._audio[indices]

        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val

        # Draw waveform using QPainterPath (single item, not N individual lines)
        pen = QPen(QColor(COLORS["primary"]))
        pen.setWidthF(1.5)
        amplitude = mid_y * 0.8

        xs = np.linspace(0, view_width, num_points)
        ys = mid_y - samples * amplitude

        path = QPainterPath()
        path.moveTo(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(x, y)
        self._scene.addPath(path, pen)

        # Draw center line
        center_pen = QPen(QColor(COLORS["outline"]))
        center_pen.setWidthF(0.5)
        self._scene.addLine(0, mid_y, view_width, mid_y, center_pen)

    def resizeEvent(self, event):
        """Debounce redraws during resize drags."""
        super().resizeEvent(event)
        if self._resize_timer is None:
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._draw)
        self._resize_timer.start(50)
