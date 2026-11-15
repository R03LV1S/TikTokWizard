"""Визуализатор спектра: радиальные лучи, бары, кольцо, волна, точки."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ..audio import Analysis, slice_bands
from ..config import VisualizerCfg
from ..draw import Layer, RGBA, gradient_colors, parse_color, ui_scale, with_alpha
from ..i18n import tr

STYLES = (
    "circle", "bars", "mirror", "wave", "ring", "dots",
    "segmented_bars", "peak_dots", "dual_wave", "radial_rays",
    "polygon_mesh", "crosshair_bars", "center_flower", "starburst",
    "sound_pillows", "curved_ribbon"
)


class Visualizer:
    """Считает геометрию один раз, а по кадрам только рисует."""

    def __init__(self, cfg: VisualizerCfg, size: tuple[int, int],
                 analysis: Analysis) -> None:
        if cfg.style not in STYLES:
            raise ValueError(
                tr("err.unknown_style", name=cfg.style,
                  available=", ".join(STYLES))
            )
        self.cfg = cfg
        self.size = size
        self.enabled = cfg.enabled

        self.lo, self.hi = slice_bands(analysis, cfg.range)
        self.n = max(2, self.hi - self.lo)

        #: размеры в конфиге заданы для кадра с короткой стороной 1080
        self.k = ui_scale(size)
        self.amp = cfg.amplitude * self.k
        self.radius = cfg.radius * self.k
        self.thickness = max(1.0, cfg.thickness * self.k)
        self.gap = cfg.gap * self.k
        self.center = (cfg.center[0] * size[0], cfg.center[1] * size[1])
        self.baseline = cfg.baseline * size[1]

        base = parse_color(cfg.color)
        second = parse_color(cfg.color2) if cfg.color2 else None
        ramp = gradient_colors(base, second, self.n)
        if self._is_mirrored():
            # зеркалим и палитру, иначе половинки кольца окрашены по-разному
            ramp = ramp + ramp[::-1]
        self.colors: list[RGBA] = [with_alpha(c, cfg.opacity) for c in ramp]
        self._bbox = self._compute_bbox()

        # Compute per-band adaptive floor and ceiling across the full audio track
        # to maximize dynamic range and reactivity (bars bounce cleanly from 0 to 1)
        raw_track = analysis.bands[:, self.lo:self.hi]
        self._band_p10 = np.percentile(raw_track, 12.0, axis=0)
        self._band_p98 = np.percentile(raw_track, 98.0, axis=0)
        self._band_span = np.maximum(self._band_p98 - self._band_p10, 0.12)

    # ------------------------------------------------------------------ #
    def _is_mirrored(self) -> bool:
        return self.cfg.style in ("circle", "ring", "dots", "radial_rays", "center_flower", "starburst", "polygon_mesh") and self.cfg.mirror

    def _compute_bbox(self) -> tuple[int, int, int, int]:
        """Минимальная область кадра, которую затрагивает визуализатор."""
        w, h = self.size
        pad = int(self.thickness * 2 + 8)
        radial_styles = ("circle", "ring", "dots", "radial_rays", "center_flower", "starburst", "polygon_mesh")
        if self.cfg.style in radial_styles:
            reach = self.radius + (0 if self.cfg.inward else self.amp) + pad
            cx, cy = self.center
            box = (cx - reach, cy - reach, cx + reach, cy + reach)
        elif self.cfg.style in ("mirror", "wave", "dual_wave", "sound_pillows", "curved_ribbon", "crosshair_bars"):
            box = (0, self.baseline - self.amp - pad, w, self.baseline + self.amp + pad)
        else:  # bars, segmented_bars, peak_dots
            box = (0, self.baseline - self.amp - pad, w, self.baseline + pad)

        x0 = max(0, int(math.floor(box[0])))
        y0 = max(0, int(math.floor(box[1])))
        x1 = min(w, int(math.ceil(box[2])))
        y1 = min(h, int(math.ceil(box[3])))
        if x1 <= x0 or y1 <= y0:
            return (0, 0, w, h)
        return (x0, y0, x1, y1)

    def _levels(self, analysis: Analysis, index: int) -> np.ndarray:
        raw_lvl = analysis.bands[analysis.clamp(index), self.lo:self.hi]
        # Dynamic per-band normalization:
        # Subtract adaptive floor and expand to [0, 1] using power curve for punchy reactivity
        norm = np.clip((raw_lvl - self._band_p10) / self._band_span, 0.0, 1.0)
        # Power curve: 1.4 creates sharp kicks and deep valleys
        expanded = np.power(norm, 1.4)
        # Beat transient modulation: pulse peaks with transient beat envelope
        beat_val = float(analysis.beat[analysis.clamp(index)])
        reactive = np.clip(expanded * (0.85 + 0.25 * beat_val), 0.0, 1.0)
        return reactive

    # ------------------------------------------------------------------ #
    def render(self, analysis: Analysis, index: int) -> tuple[Image.Image, tuple[int, int]] | None:
        """Возвращает RGBA-слой и координаты его вставки, либо None."""
        if not self.enabled:
            return None
        levels = self._levels(analysis, index)
        x0, y0, x1, y1 = self._bbox
        layer = Layer((x1 - x0, y1 - y0), scale=2, origin=(x0, y0))

        drawer = getattr(self, f"_draw_{self.cfg.style}")
        drawer(layer, levels)

        img = layer.resolve(self.cfg.glow * self.k, self.cfg.glow_strength)
        return img, (x0, y0)

    # ------------------------------------------------------------------ #
    #  Радиальные стили
    # ------------------------------------------------------------------ #
    def _radial_values(self, levels: np.ndarray) -> np.ndarray:
        """Спектр по кругу: при mirror — зеркальная копия для симметрии."""
        if self.cfg.mirror:
            return np.concatenate([levels, levels[::-1]])
        return levels

    def _draw_circle(self, layer: Layer, levels: np.ndarray) -> None:
        values = self._radial_values(levels)
        cx, cy = self.center
        step = 360.0 / len(values)
        sign = -1.0 if self.cfg.inward else 1.0
        round_cap = self.cfg.cap == "round"

        for i, v in enumerate(values):
            angle = math.radians(self.cfg.start_angle + i * step)
            ca, sa = math.cos(angle), math.sin(angle)
            r0 = self.radius
            r1 = self.radius + sign * (self.amp * float(v) + self.thickness * 0.5)
            layer.line(
                (cx + ca * r0, cy + sa * r0),
                (cx + ca * r1, cy + sa * r1),
                self.colors[i], self.thickness, round_cap,
            )

    def _draw_dots(self, layer: Layer, levels: np.ndarray) -> None:
        values = self._radial_values(levels)
        cx, cy = self.center
        step = 360.0 / len(values)
        sign = -1.0 if self.cfg.inward else 1.0

        for i, v in enumerate(values):
            angle = math.radians(self.cfg.start_angle + i * step)
            r = self.radius + sign * self.amp * float(v)
            layer.circle(
                (cx + math.cos(angle) * r, cy + math.sin(angle) * r),
                self.thickness * (0.5 + 0.7 * float(v)),
                self.colors[i],
            )

    def _draw_ring(self, layer: Layer, levels: np.ndarray) -> None:
        """Замкнутая гладкая линия — радиус модулируется спектром."""
        values = self._radial_values(levels)
        cx, cy = self.center
        sign = -1.0 if self.cfg.inward else 1.0
        # сглаживаем стык, чтобы кольцо было непрерывным
        smooth = _wrap_smooth(values, 1.2)
        step = 2.0 * math.pi / len(smooth)

        points = []
        for i, v in enumerate(smooth):
            angle = math.radians(self.cfg.start_angle) + i * step
            r = self.radius + sign * self.amp * float(v)
            points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))

        points.append(points[0])
        for i in range(len(points) - 1):
            layer.line(points[i], points[i + 1],
                       self.colors[min(i, len(self.colors) - 1)], self.thickness)

    # ------------------------------------------------------------------ #
    #  Линейные стили
    # ------------------------------------------------------------------ #
    def _bar_geometry(self) -> tuple[float, float]:
        """Ширина бара и шаг между барами по всей ширине кадра."""
        total = self.size[0]
        pitch = total / self.n
        width = max(1.0, min(self.thickness, pitch - self.gap))
        return width, pitch

    def _draw_bars(self, layer: Layer, levels: np.ndarray) -> None:
        width, pitch = self._bar_geometry()
        radius = width / 2.0 if self.cfg.cap == "round" else 0.0
        for i, v in enumerate(levels):
            x = pitch * (i + 0.5)
            height = max(width, self.amp * float(v))
            layer.rect(
                (x - width / 2.0, self.baseline - height, x + width / 2.0, self.baseline),
                self.colors[i], radius,
            )

    def _draw_mirror(self, layer: Layer, levels: np.ndarray) -> None:
        width, pitch = self._bar_geometry()
        radius = width / 2.0 if self.cfg.cap == "round" else 0.0
        for i, v in enumerate(levels):
            x = pitch * (i + 0.5)
            half = max(width / 2.0, self.amp * float(v) / 2.0)
            layer.rect(
                (x - width / 2.0, self.baseline - half,
                 x + width / 2.0, self.baseline + half),
                self.colors[i], radius,
            )

    def _draw_wave(self, layer: Layer, levels: np.ndarray) -> None:
        """Плавная симметричная линия по ширине кадра."""
        values = _wrap_smooth(np.concatenate([levels[::-1], levels]), 1.6)
        count = len(values)
        step = self.size[0] / (count - 1)

        points = [
            (i * step, self.baseline - (float(v) - 0.5) * self.amp)
            for i, v in enumerate(values)
        ]
        for i in range(len(points) - 1):
            layer.line(points[i], points[i + 1],
                       self.colors[min(i % self.n, len(self.colors) - 1)],
                       self.thickness)

    def _draw_segmented_bars(self, layer: Layer, levels: np.ndarray) -> None:
        """Дискретные сегментированные блоки (LED EQ style)."""
        width, pitch = self._bar_geometry()
        radius = width / 2.0 if self.cfg.cap == "round" else 0.0
        n_segments = 12
        seg_h = max(2.0, (self.amp / n_segments) - 3.0)
        seg_gap = 3.0

        for i, v in enumerate(levels):
            x = pitch * (i + 0.5)
            active_segs = int(round(float(v) * n_segments))
            for s_idx in range(active_segs):
                y_bot = self.baseline - s_idx * (seg_h + seg_gap)
                y_top = y_bot - seg_h
                layer.rect((x - width / 2.0, y_top, x + width / 2.0, y_bot), self.colors[i], radius)

    def _draw_peak_dots(self, layer: Layer, levels: np.ndarray) -> None:
        """Частотные бары с парящими точками-пиками."""
        width, pitch = self._bar_geometry()
        radius = width / 2.0 if self.cfg.cap == "round" else 0.0
        for i, v in enumerate(levels):
            x = pitch * (i + 0.5)
            height = max(width, self.amp * float(v))
            # Main bar
            layer.rect((x - width / 2.0, self.baseline - height, x + width / 2.0, self.baseline), self.colors[i], radius)
            # Floating peak dot
            peak_y = self.baseline - height - width - 6.0
            layer.circle((x, peak_y), width * 0.75, self.colors[i])

    def _draw_dual_wave(self, layer: Layer, levels: np.ndarray) -> None:
        """Двойная синусоидальная фазовая волна."""
        values = _wrap_smooth(np.concatenate([levels[::-1], levels]), 1.4)
        count = len(values)
        step = self.size[0] / (count - 1)
        for wave_idx, mult in enumerate([1.0, -0.75]):
            points = [
                (i * step, self.baseline - (float(v) - 0.5) * self.amp * mult)
                for i, v in enumerate(values)
            ]
            col_idx = 0 if wave_idx == 0 else len(self.colors) - 1
            for i in range(len(points) - 1):
                layer.line(points[i], points[i + 1], self.colors[col_idx], self.thickness * (1.0 if wave_idx == 0 else 0.7))

    def _draw_radial_rays(self, layer: Layer, levels: np.ndarray) -> None:
        """Тонкие высокоплотные радиальные лучи."""
        values = self._radial_values(levels)
        cx, cy = self.center
        step = 360.0 / len(values)
        for i, v in enumerate(values):
            angle = math.radians(self.cfg.start_angle + i * step)
            ca, sa = math.cos(angle), math.sin(angle)
            r0 = self.radius * 0.85
            r1 = self.radius + self.amp * float(v) * 1.3
            layer.line((cx + ca * r0, cy + sa * r0), (cx + ca * r1, cy + sa * r1), self.colors[i], max(1.5, self.thickness * 0.6))

    def _draw_polygon_mesh(self, layer: Layer, levels: np.ndarray) -> None:
        """Геометрический многоугольник, пульсирующий гранями."""
        values = _wrap_smooth(self._radial_values(levels), 1.2)
        cx, cy = self.center
        step = 2.0 * math.pi / len(values)
        points = []
        for i, v in enumerate(values):
            angle = math.radians(self.cfg.start_angle) + i * step
            r = self.radius + self.amp * float(v)
            points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
        points.append(points[0])
        for i in range(len(points) - 1):
            layer.line(points[i], points[i + 1], self.colors[min(i, len(self.colors) - 1)], self.thickness * 1.2)
            # Cross lattice to center
            if i % 3 == 0:
                layer.line((cx, cy), points[i], with_alpha(self.colors[min(i, len(self.colors) - 1)], 0.35), 1.0)

    def _draw_crosshair_bars(self, layer: Layer, levels: np.ndarray) -> None:
        """Симметричные горизонтальные кроссхейр бары от центра к краям."""
        w, _ = self.size
        cx = w / 2.0
        n_half = len(levels) // 2
        step_x = (cx - 40.0) / max(1, n_half)
        bar_w = max(2.0, step_x - self.gap)
        for i in range(n_half):
            v = levels[i]
            h = max(2.0, self.amp * float(v))
            # Left side
            x_l = cx - (i + 1) * step_x
            layer.rect((x_l, self.baseline - h / 2.0, x_l + bar_w, self.baseline + h / 2.0), self.colors[i])
            # Right side
            x_r = cx + i * step_x
            layer.rect((x_r, self.baseline - h / 2.0, x_r + bar_w, self.baseline + h / 2.0), self.colors[len(levels) - 1 - i])

    def _draw_center_flower(self, layer: Layer, levels: np.ndarray) -> None:
        """Многолепестковый цветок с градиентным раскрытием."""
        values = self._radial_values(levels)
        cx, cy = self.center
        step = 360.0 / len(values)
        for i, v in enumerate(values):
            angle = math.radians(self.cfg.start_angle + i * step)
            ca, sa = math.cos(angle), math.sin(angle)
            r_tip = self.radius + self.amp * float(v)
            # Petal loop
            layer.circle((cx + ca * r_tip, cy + sa * r_tip), max(2.0, self.thickness * (0.8 + 1.2 * float(v))), self.colors[i])
            layer.line((cx + ca * self.radius, cy + sa * self.radius), (cx + ca * r_tip, cy + sa * r_tip), self.colors[i], self.thickness)

    def _draw_starburst(self, layer: Layer, levels: np.ndarray) -> None:
        """Сверкающая звезда с динамическими пиками."""
        values = self._radial_values(levels)
        cx, cy = self.center
        step = 2.0 * math.pi / len(values)
        points = []
        for i, v in enumerate(values):
            a = i * step
            # Alternate inner and outer vertices for starburst
            r = (self.radius * 0.6) if (i % 2 == 0) else (self.radius + self.amp * float(v) * 1.4)
            points.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
        points.append(points[0])
        for i in range(len(points) - 1):
            layer.line(points[i], points[i + 1], self.colors[min(i, len(self.colors) - 1)], self.thickness)

    def _draw_sound_pillows(self, layer: Layer, levels: np.ndarray) -> None:
        """Скругленные парящие звуковые капсулы."""
        width, pitch = self._bar_geometry()
        for i, v in enumerate(levels):
            x = pitch * (i + 0.5)
            h = max(width * 2.0, self.amp * float(v))
            top_y = self.baseline - h
            layer.rect((x - width / 2.0, top_y, x + width / 2.0, top_y + width * 2.0), self.colors[i], width / 2.0)

    def _draw_curved_ribbon(self, layer: Layer, levels: np.ndarray) -> None:
        """Широкая плавная изогнутая неоновая лента."""
        values = _wrap_smooth(np.concatenate([levels[::-1], levels]), 2.0)
        count = len(values)
        step = self.size[0] / (count - 1)
        for offset_y, col_mult in [(-8, 0.7), (0, 1.0), (8, 0.7)]:
            points = [
                (i * step, self.baseline - (float(v) - 0.5) * self.amp + offset_y)
                for i, v in enumerate(values)
            ]
            for i in range(len(points) - 1):
                c = with_alpha(self.colors[min(i % self.n, len(self.colors) - 1)], col_mult)
                layer.line(points[i], points[i + 1], c, self.thickness * 1.5)


def _wrap_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Циклическое гауссово сглаживание — убирает угловатость кривых."""
    if sigma <= 0 or len(values) < 3:
        return values
    radius = max(1, int(round(sigma * 2.5)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.concatenate([values[-radius:], values, values[:radius]])
    return np.convolve(padded, k, mode="valid")
