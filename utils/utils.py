#!/usr/bin/env python3
"""
Utility functions for the Voice Transcriber application.

This module provides helper functions for loading the configuration, playing audio feedback,
and controlling the system volume.
"""

import yaml
from pathlib import Path
import numpy as np
import pyaudio
import platform
import subprocess

def load_config(config_path):
    """Load configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    try:
        config_file = Path(__file__).parent.parent / config_path
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"⚠️  Config file not found at {config_path}, using defaults")
        return get_default_config()
    except Exception as e:
        print(f"⚠️  Error loading config: {e}, using defaults")
        return get_default_config()

def get_default_config():
    """Return the default configuration settings.

    Returns:
        dict: A dictionary containing the default configuration.
    """
    return {
        'whisper_model': 'base.en',
        'auto_paste': True,
        'audio_feedback': True,
        'hotkey_code': 176,
        'attenuate_volume': True,
        'attenuation_percent': 10,
        'audio': {
            'sample_rate': 16000,
            'channels': 1,
            'chunk_size': 1024
        }
    }

def play_beep(audio_instance, audio_feedback=True, frequency=800, duration=0.1):
    """Play a beep sound for audio feedback.

    Args:
        audio_instance (pyaudio.PyAudio): The PyAudio instance.
        audio_feedback (bool, optional): Whether to play the beep. Defaults to True.
        frequency (int, optional): The frequency of the beep. Defaults to 800.
        duration (float, optional): The duration of the beep. Defaults to 0.1.
    """
    if not audio_feedback:
        return

    try:
        sample_rate = 44100
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        wave_data = np.sin(frequency * 2 * np.pi * t)
        audio_data = (wave_data * 3000).astype(np.int16)

        stream = audio_instance.open(format=pyaudio.paInt16,
                                   channels=1,
                                   rate=sample_rate,
                                   output=True)
        stream.write(audio_data.tobytes())
        stream.stop_stream()
        stream.close()
    except Exception:
        pass  # Silently fail

def get_system_volume():
    """Get the current system volume (macOS only).

    Returns:
        int: The current system volume (0-100), or None if not on macOS.
    """
    if platform.system() != 'Darwin':
        return None
    try:
        result = subprocess.run(
            ['osascript', '-e', 'output volume of (get volume settings)'],
            capture_output=True, text=True, timeout=1, check=True
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None

def set_system_volume(volume):
    """Set the system volume (macOS only, 0-100).

    Args:
        volume (int): The desired volume level (0-100).

    Returns:
        bool: True if the volume was set successfully, False otherwise.
    """
    if platform.system() != 'Darwin':
        return False
    try:
        volume = max(0, min(100, int(volume)))
        subprocess.run(
            ['osascript', '-e', f'set volume output volume {volume}'],
            capture_output=True, timeout=1, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
