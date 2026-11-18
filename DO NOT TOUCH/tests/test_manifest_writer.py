"""Unit test suite for manifest_writer module."""

import sys
import os
from pathlib import Path

# Ensure DO NOT TOUCH root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.manifest_writer import write_video_render_manifest, format_duration, sha256_file


def test_format_duration():
    assert format_duration(65.5) == "01:05.50"
    assert format_duration(3665.0) == "01:01:05.00"
    assert format_duration(0.0) == "00:00.00"


def test_manifest_generation(tmp_path: Path):
    video_path = tmp_path / "test_song v1.mp4"
    video_path.write_bytes(b"dummy video content for test")
    
    audio_path = tmp_path / "test_song.mp3"
    audio_path.write_bytes(b"dummy audio content")
    
    lyrics_path = tmp_path / "lyrics.txt"
    lyrics_path.write_text("hello world", encoding="utf-8")
    
    config = {
        "video_width": 1080,
        "video_height": 1920,
        "fps": 30,
        "codec": "h264_qsv",
        "crf": 22,
        "audio_codec": "aac",
        "whisper_model": "small",
        "whisper_compute_type": "int8",
        "tau_song": 0.23,
        "word_target": 0.95
    }
    
    effect_cfg = {
        "name": "Cinematic Test",
        "schema": "shotcut_tiktok.effects.v1",
        "description": "Test preset",
        "visualizer": {
            "type": "minimal_line",
            "color": "#d4af37",
            "height_pct": 8.0,
            "position": "above_pad"
        },
        "lyrics": {
            "font": "Fraunces",
            "font_size": 68,
            "highlight_color": "&H0037AFD4&",
            "base_color": "&H00D4AF37&",
            "outline_color": "&H00000000&",
            "outline_width": 4,
            "shadow_depth": 4,
            "position_y_percent": 70
        },
        "particles": {
            "type": "cinematic_snow",
            "count": 55,
            "speed": 0.9,
            "color": "#F0E6D2",
            "opacity": 0.55
        },
        "tiktok_pad": {
            "enabled": True,
            "opacity": 0.90,
            "height_pct": 12.5
        },
        "creator_handle": {
            "text": "@Tester",
            "position": "top_center",
            "opacity": 0.85
        }
    }
    
    txt_path = write_video_render_manifest(
        output_video_path=video_path,
        config=config,
        effect_cfg=effect_cfg,
        audio_path=audio_path,
        lyrics_path=lyrics_path,
        duration=30.0,
        render_seconds=5.2,
        iteration=1,
        profile_key="cinematic_test",
        project_folder=tmp_path,
        wizard_version="v4.9"
    )
    
    assert txt_path.exists()
    assert txt_path.name == "test_song v1.txt"
    content = txt_path.read_text(encoding="utf-8")
    
    assert "TIKTOK WIZARD VIDEO RENDER SPECIFICATION & SETTINGS MANIFEST" in content
    assert "TikTok Wizard v4.9" in content
    assert "Resolution         : 1080 x 1920" in content
    assert "minimal_line" in content
    assert "Fraunces" in content
    assert "cinematic_snow" in content
    assert "Intel Arc Pro B70" in content
    assert "AMD Ryzen 9 7950X3D" in content
    assert "@Tester" in content
    assert "Strictly 1 line displayed" in content
    assert "Two-Tier Cache" in content
    assert "Audio Track Mode" in content
