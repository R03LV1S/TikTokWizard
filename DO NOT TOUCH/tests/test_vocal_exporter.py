"""Unit test suite for Vocal Stem & Acapella Isolation Exporter."""

import sys
import os
from pathlib import Path

# Ensure DO NOT TOUCH root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.vocal_exporter import export_vocal_stems, find_stems_dir


def test_vocal_stem_export_with_dummy_data(tmp_path: Path):
    project_dir = tmp_path / "Test_Song_Project"
    project_dir.mkdir()
    
    # Create mock .data/stems with wav files
    stems_dir = project_dir / ".data" / "stems"
    stems_dir.mkdir(parents=True)
    
    dummy_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    (stems_dir / "acapella_full.wav").write_bytes(dummy_wav)
    (stems_dir / "acapella_lead_dry.wav").write_bytes(dummy_wav)
    (stems_dir / "acapella_backing_echo.wav").write_bytes(dummy_wav)
    (stems_dir / "original_mix_vocal_band.wav").write_bytes(dummy_wav)
    
    res = export_vocal_stems(project_dir, export_mp3=False, log_fn=lambda _: None)
    
    assert "full_wav" in res
    assert "lead_wav" in res
    assert "backing_wav" in res
    assert "vocal_band_wav" in res
    
    dest_dir = project_dir / "Vocal Stem"
    assert dest_dir.exists()
    assert (dest_dir / "Acapella (Full Vocals).wav").exists()
    assert (dest_dir / "Acapella (Lead Vocals - Dry).wav").exists()
    assert (dest_dir / "Acapella (Backing & Harmonies).wav").exists()
    assert (dest_dir / "README.txt").exists()
