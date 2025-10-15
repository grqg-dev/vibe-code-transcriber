#!/usr/bin/env python3
"""
Audio handling for the Voice Transcriber application.

This module provides functions for checking microphone access, recording audio,
and saving audio to a file.
"""

import pyaudio
import wave
import numpy as np

def check_microphone_access(audio_instance, config):
    """Check if the microphone is accessible.

    Args:
        audio_instance (pyaudio.PyAudio): The PyAudio instance.
        config (dict): The application configuration.

    Returns:
        bool: True if the microphone is accessible, False otherwise.
    """
    try:
        stream = audio_instance.open(
            format=pyaudio.paInt16,
            channels=config['audio']['channels'],
            rate=config['audio']['sample_rate'],
            input=True,
            frames_per_buffer=config['audio']['chunk_size']
        )
        stream.close()
        return True
    except Exception:
        return False

def record_audio(stream, data_queue, is_recording, chunk_size, real_time_mode, audio_data):
    """Record audio from the microphone in a separate thread.

    Args:
        stream (pyaudio.Stream): The audio stream.
        data_queue (Queue): A queue to store audio data for real-time mode.
        is_recording (bool): A flag to indicate if recording is in progress.
        chunk_size (int): The size of each audio chunk.
        real_time_mode (bool): Whether real-time mode is enabled.
        audio_data (list): A list to store audio data for batch mode.
    """
    while is_recording:
        try:
            data = stream.read(chunk_size, exception_on_overflow=False)
            if real_time_mode:
                data_queue.put(data)
            else:
                audio_data.append(data)
        except Exception as e:
            print(f"❌ Error during recording: {e}")
            break

def save_audio_to_file(audio_instance, file_path, audio_data, config):
    """Save recorded audio data to a WAV file.

    Args:
        audio_instance (pyaudio.PyAudio): The PyAudio instance.
        file_path (str): The path to save the audio file.
        audio_data (list): A list of audio data chunks.
        config (dict): The application configuration.
    """
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(config['audio']['channels'])
    wf.setsampwidth(audio_instance.get_sample_size(pyaudio.paInt16))
    wf.setframerate(config['audio']['sample_rate'])
    wf.writeframes(b''.join(audio_data))
    wf.close()
