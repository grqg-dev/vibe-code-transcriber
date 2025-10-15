#!/usr/bin/env python3
"""
Voice Transcriber - Push-to-talk terminal transcription tool
Hold dictation key to record, release to transcribe with Whisper
"""

import pyaudio
import wave
import whisper
import pyperclip
import tempfile
import os
import sys
from pynput import keyboard
from pynput.keyboard import Key, KeyCode, Controller
import threading
import time
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from queue import Queue
import numpy as np
import argparse
import subprocess
import platform


class VoiceTranscriber:
    def __init__(self, config_path="config.yaml", real_time_mode=False):
        # Load configuration
        self.config = self.load_config(config_path)

        self.is_recording = False
        self.audio_data = []
        self.audio = None
        self.record_start_time = None

        # Real-time mode settings
        self.real_time_mode = real_time_mode
        self.data_queue = Queue()
        self.phrase_time = None
        self.last_transcribed_text = ""
        self.transcription_thread = None

        # Initialize PyAudio with error handling
        try:
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            print(f"❌ Error initializing audio: {e}")
            print("Please check that your microphone is connected and accessible.")
            sys.exit(1)

        # Audio settings from config
        self.CHUNK = self.config['audio']['chunk_size']
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = self.config['audio']['channels']
        self.RATE = self.config['audio']['sample_rate']

        # Whisper model (will be loaded on first use)
        self.model = None
        self.model_name = self.config['whisper_model']

        # Thread for recording
        self.record_thread = None
        self.stream = None

        # Hotkey for recording from config
        self.record_key = KeyCode.from_vk(self.config['hotkey_code'])

        # Keyboard controller for auto-paste
        self.kb_controller = Controller()

        # Feature flags
        self.auto_paste = self.config['auto_paste']
        self.audio_feedback = self.config['audio_feedback']

        # Volume attenuation
        self.attenuate_volume = self.config.get('attenuate_volume', True)
        self.attenuation_percent = self.config.get('attenuation_percent', 10)
        self.original_volume = None

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            config_file = Path(__file__).parent / config_path
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"⚠️  Config file not found at {config_path}, using defaults")
            return self.get_default_config()
        except Exception as e:
            print(f"⚠️  Error loading config: {e}, using defaults")
            return self.get_default_config()

    def get_default_config(self):
        """Return default configuration"""
        return {
            'whisper_model': 'base.en',
            'auto_paste': True,
            'audio_feedback': True,
            'hotkey_code': 176,
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'chunk_size': 1024
            }
        }

    def play_beep(self, frequency=800, duration=0.1):
        """Play a beep sound for audio feedback"""
        if not self.audio_feedback:
            return

        try:
            import numpy as np
            sample_rate = 44100
            samples = int(sample_rate * duration)
            t = np.linspace(0, duration, samples, False)
            wave_data = np.sin(frequency * 2 * np.pi * t)
            # Much quieter beep (about 10% volume)
            audio_data = (wave_data * 3000).astype(np.int16)

            # Play using PyAudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=sample_rate,
                          output=True)
            stream.write(audio_data.tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            # Silently fail if beep doesn't work
            pass

    def get_system_volume(self):
        """Get current system volume (macOS only)"""
        if platform.system() != 'Darwin':
            return None

        try:
            result = subprocess.run(
                ['osascript', '-e', 'output volume of (get volume settings)'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return None

    def set_system_volume(self, volume):
        """Set system volume (macOS only, 0-100)"""
        if platform.system() != 'Darwin':
            return False

        try:
            volume = max(0, min(100, int(volume)))  # Clamp to 0-100
            subprocess.run(
                ['osascript', '-e', f'set volume output volume {volume}'],
                capture_output=True,
                timeout=1
            )
            return True
        except Exception:
            return False

    def attenuate_system_volume(self):
        """Lower system volume to attenuation_percent of original"""
        if not self.attenuate_volume:
            return

        self.original_volume = self.get_system_volume()
        if self.original_volume is not None:
            target_volume = int(self.original_volume * (self.attenuation_percent / 100.0))
            self.set_system_volume(target_volume)

    def restore_system_volume(self):
        """Restore system volume to original level"""
        if not self.attenuate_volume or self.original_volume is None:
            return

        self.set_system_volume(self.original_volume)
        self.original_volume = None

    def check_microphone_access(self):
        """Check if microphone is accessible"""
        try:
            test_stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            test_stream.close()
            return True
        except Exception as e:
            return False

    def load_whisper_model(self):
        """Load Whisper model (this will download on first run)"""
        if self.model is None:
            print(f"⏳ Loading Whisper model '{self.model_name}'...")
            print("   (This may take a moment on first run)")
            try:
                self.model = whisper.load_model(self.model_name)
                print("✅ Model loaded successfully!\n")
            except Exception as e:
                print(f"❌ Error loading Whisper model: {e}")
                print("   Check your internet connection and try again.")
                sys.exit(1)

    def start_recording(self):
        """Start recording audio from microphone"""
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_data = []
        self.record_start_time = time.time()
        self.last_transcribed_text = ""

        # Play start beep, then attenuate volume
        def beep_then_attenuate():
            self.play_beep(800, 0.1)
            self.attenuate_system_volume()

        threading.Thread(target=beep_then_attenuate).start()

        if self.real_time_mode:
            print("\n" + "─" * 60)
            print("🔴 REAL-TIME RECORDING... (release key to stop)")
            print("─" * 60)
        else:
            print("\n" + "─" * 60)
            print("🔴 RECORDING... (release key to stop)")
            print("─" * 60)
        sys.stdout.flush()

        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

            # Record in a separate thread
            self.record_thread = threading.Thread(target=self._record_audio)
            self.record_thread.start()

            # Start real-time transcription thread if in real-time mode
            if self.real_time_mode:
                self.transcription_thread = threading.Thread(target=self._real_time_transcribe)
                self.transcription_thread.start()

        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            print("   Check System Preferences → Privacy & Security → Microphone")
            self.is_recording = False

    def _record_audio(self):
        """Internal method to record audio in a thread"""
        while self.is_recording:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                if self.real_time_mode:
                    # Put data in queue for real-time processing
                    self.data_queue.put(data)
                else:
                    # Store all data for batch processing
                    self.audio_data.append(data)
            except Exception as e:
                print(f"❌ Error during recording: {e}")
                break

    def _real_time_transcribe(self):
        """Real-time transcription thread - processes audio as it comes in"""
        # Load model if not already loaded
        self.load_whisper_model()

        phrase_bytes = bytes()
        record_timeout = 1.5  # How often to transcribe (in seconds)
        min_audio_length = 0.5  # Minimum audio length to transcribe (in seconds)

        print("💡 Starting real-time transcription...")
        sys.stdout.flush()

        while self.is_recording or not self.data_queue.empty():
            try:
                now = datetime.utcnow()

                # Check if we have audio data in the queue
                if not self.data_queue.empty():
                    # Update phrase time
                    if self.phrase_time is None:
                        self.phrase_time = now

                    # Get all available audio from queue
                    audio_chunks = []
                    while not self.data_queue.empty():
                        audio_chunks.append(self.data_queue.get())

                    # Combine and add to accumulated audio
                    audio_data = b''.join(audio_chunks)
                    phrase_bytes += audio_data

                    # Check if enough time has passed to transcribe
                    time_since_phrase_start = (now - self.phrase_time).total_seconds()
                    audio_duration = len(phrase_bytes) / (2 * self.CHANNELS * self.RATE)  # 2 bytes per sample

                    if audio_duration >= min_audio_length and time_since_phrase_start >= record_timeout:
                        # Convert bytes to numpy array
                        audio_np = np.frombuffer(phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                        # Transcribe
                        try:
                            result = self.model.transcribe(audio_np, fp16=False)
                            new_text = result['text'].strip()

                            # Type only the new characters
                            if new_text and new_text != self.last_transcribed_text:
                                self._type_new_text(new_text)
                                self.last_transcribed_text = new_text

                        except Exception as e:
                            print(f"\n⚠️  Transcription error: {e}")

                        # Reset for next phrase
                        self.phrase_time = now
                        phrase_bytes = bytes()

                else:
                    # No data available, sleep briefly
                    time.sleep(0.1)

            except Exception as e:
                print(f"\n❌ Real-time transcription error: {e}")
                import traceback
                traceback.print_exc()
                break

        print("\n✅ Real-time transcription completed")
        sys.stdout.flush()

    def _type_new_text(self, new_text):
        """Type only the new portion of text to avoid duplicates"""
        # Find the common prefix between old and new text
        common_length = 0
        for i in range(min(len(self.last_transcribed_text), len(new_text))):
            if self.last_transcribed_text[i] == new_text[i]:
                common_length += 1
            else:
                break

        # Extract only the new characters
        new_chars = new_text[common_length:]

        if new_chars:
            # Type character by character
            for char in new_chars:
                try:
                    self.kb_controller.type(char)
                    time.sleep(0.01)  # Small delay between characters for smooth typing
                except Exception as e:
                    print(f"\n⚠️  Typing error: {e}")
                    break

    def stop_recording(self):
        """Stop recording and transcribe"""
        if not self.is_recording:
            return

        self.is_recording = False
        duration = time.time() - self.record_start_time

        # Wait for recording thread to finish
        if self.record_thread:
            self.record_thread.join()

        # Stop and close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # Restore system volume
        self.restore_system_volume()

        # Play stop beep
        threading.Thread(target=lambda: self.play_beep(600, 0.1)).start()

        print(f"⏹️  Stopped recording ({duration:.1f}s)")

        if self.real_time_mode:
            # Wait for transcription thread to finish processing
            if self.transcription_thread:
                print("⏳ Finishing transcription...")
                sys.stdout.flush()
                self.transcription_thread.join(timeout=5.0)  # Wait up to 5 seconds
                print("─" * 60 + "\n")
        else:
            # Regular mode - transcribe after recording
            print("⏳ Processing transcription...")
            sys.stdout.flush()

            # Save to temporary file
            if self.audio_data:
                self.transcribe_audio()
            else:
                print("❌ No audio recorded\n")

    def transcribe_audio(self):
        """Transcribe the recorded audio using Whisper"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name

            # Write audio data to file
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.audio_data))
            wf.close()

            try:
                # Load model if not already loaded
                self.load_whisper_model()

                # Transcribe
                start_time = time.time()
                result = self.model.transcribe(temp_filename, fp16=False)
                transcribe_time = time.time() - start_time
                text = result["text"].strip()

                if text:
                    # Print the transcription with nice formatting
                    print("\n" + "═" * 60)
                    print("📝 TRANSCRIPTION")
                    print("═" * 60)
                    print(f"{text}")
                    print("═" * 60)
                    print(f"⏱️  Transcribed in {transcribe_time:.1f}s")

                    # Copy to clipboard
                    try:
                        pyperclip.copy(text)
                        print("✅ Copied to clipboard")
                    except Exception as e:
                        print(f"⚠️  Could not copy to clipboard: {e}")

                    # Auto-paste if enabled
                    if self.auto_paste:
                        try:
                            time.sleep(0.2)
                            with self.kb_controller.pressed(Key.cmd):
                                self.kb_controller.press('v')
                                self.kb_controller.release('v')
                            print("✅ Auto-pasted")
                        except Exception as e:
                            print(f"⚠️  Could not auto-paste: {e}")
                            print("   Enable Accessibility permissions in System Preferences")

                    print("─" * 60 + "\n")
                else:
                    print("⚠️  No speech detected in audio\n")

            except Exception as e:
                print(f"❌ Error transcribing: {e}\n")
                import traceback
                traceback.print_exc()

            finally:
                # Clean up temp file
                try:
                    os.remove(temp_filename)
                except:
                    pass

    def on_press(self, key):
        """Callback for key press events"""
        if key == self.record_key:
            self.start_recording()

    def on_release(self, key):
        """Callback for key release events"""
        if key == self.record_key:
            self.stop_recording()

    def run(self):
        """Main run loop"""
        print("\n" + "═" * 60)
        if self.real_time_mode:
            print("🎤  VOICE TRANSCRIBER - REAL-TIME MODE")
        else:
            print("🎤  VOICE TRANSCRIBER")
        print("═" * 60)

        # Show configuration
        print(f"\n📋 Configuration:")
        print(f"   Model: {self.model_name}")
        if self.real_time_mode:
            print(f"   Mode: Real-time typing")
        else:
            print(f"   Auto-paste: {'Enabled' if self.auto_paste else 'Disabled'}")
        print(f"   Audio feedback: {'Enabled' if self.audio_feedback else 'Disabled'}")

        print("\n📖 Instructions:")
        print("   • Hold dictation key to record")
        if self.real_time_mode:
            print("   • Text will type in real-time as you speak")
        else:
            print("   • Release to transcribe and paste")
        print("   • Press Ctrl+C to quit")
        print("\n💡 Tip: Run detect_key.py to find your key code")

        # Check microphone access
        print("\n🔍 Checking microphone access...")
        if self.check_microphone_access():
            print("✅ Microphone is accessible")
        else:
            print("❌ Cannot access microphone!")
            print("   Go to: System Preferences → Privacy & Security → Microphone")
            print("   Enable access for Terminal (or your terminal app)")
            sys.exit(1)

        print("\n" + "═" * 60)
        print("✅ Ready! Press and hold the dictation key to start...")
        print("═" * 60 + "\n")

        # Start keyboard listener
        try:
            with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
            ) as listener:
                listener.join()
        except Exception as e:
            print(f"\n❌ Keyboard listener error: {e}")
            print("   You may need to grant Accessibility permissions")
            print("   Go to: System Preferences → Privacy & Security → Accessibility")

        # Cleanup
        if self.audio:
            self.audio.terminate()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Voice Transcriber - Push-to-talk terminal transcription tool'
    )
    parser.add_argument(
        '--real-time',
        action='store_true',
        help='Enable real-time typing mode (types as you speak)'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    args = parser.parse_args()

    try:
        transcriber = VoiceTranscriber(
            config_path=args.config,
            real_time_mode=args.real_time
        )
        transcriber.run()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
