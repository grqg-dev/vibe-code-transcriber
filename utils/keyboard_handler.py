#!/usr/bin/env python3
"""
Keyboard handling for the Voice Transcriber application.

This module provides functions for handling keyboard press and release events.
"""

from pynput import keyboard

def on_press(key, record_key, start_recording_func):
    """Callback function for key press events.

    Args:
        key (pynput.keyboard.Key): The key that was pressed.
        record_key (pynput.keyboard.KeyCode): The key code for the record button.
        start_recording_func (function): The function to call when the record key is pressed.
    """
    if key == record_key:
        start_recording_func()

def on_release(key, record_key, stop_recording_func):
    """Callback function for key release events.

    Args:
        key (pynput.keyboard.Key): The key that was released.
        record_key (pynput.keyboard.KeyCode): The key code for the record button.
        stop_recording_func (function): The function to call when the record key is released.
    """
    if key == record_key:
        stop_recording_func()
