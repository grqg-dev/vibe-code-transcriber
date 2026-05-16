# 🎤 Voice Transcriber

A powerful push-to-talk terminal utility that uses NVIDIA's Parakeet ASR models (via NeMo) to transcribe speech to text on macOS. Features local processing, auto-paste, audio feedback, and full customization.

## ✨ Features

- **🎯 Push-to-talk recording**: Hold dictation key to record, release to transcribe
- **🔒 100% Local**: Runs entirely on your machine with NVIDIA Parakeet (no API costs, no data sent to cloud)
- **📋 Auto-clipboard & Auto-paste**: Transcriptions automatically copied and pasted where your cursor is
- **🔊 Audio feedback**: Beeps when starting/stopping recording
- **⚙️ Fully customizable**: YAML config for models, hotkeys, and features
- **✅ Smart error handling**: Checks permissions and provides helpful error messages
- **📊 Beautiful terminal UI**: Clean, informative output with recording time and stats

## 🚀 Quick Start

### Installation

```bash
cd voice-transcriber
./install.sh
```

The install script will:
- Install system dependencies (ffmpeg, portaudio)
- Set up Python virtual environment
- Install all Python packages
- Optionally create a `transcribe` command alias

### Run

```bash
# Option 1: Direct run
./run.sh

# Option 2: If you created the alias
transcribe
```

### Controls

- **Dictation Key** (without Fn): Hold to record, release to transcribe
- **ESC**: Quit the application

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Parakeet model (any nvidia/parakeet-* model on Hugging Face / NGC)
parakeet_model: "nvidia/parakeet-tdt-0.6b-v2"

# Auto-paste after transcription
auto_paste: true

# Audio beep feedback
audio_feedback: true

# Custom hotkey (run detect_key.py to find key codes)
hotkey_code: 176

# Audio settings (Parakeet expects 16kHz mono)
audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024
```

### Parakeet Models

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `nvidia/parakeet-tdt_ctc-110m` | ~110M | Fastest | Quick notes |
| `nvidia/parakeet-tdt-0.6b-v2` | ~600M | Fast | **Default - best balance** |
| `nvidia/parakeet-ctc-1.1b` | ~1.1B | Slower | Higher accuracy |
| `nvidia/parakeet-rnnt-1.1b` | ~1.1B | Slowest | Maximum accuracy |

All Parakeet models are English-only. The first run will download the model from Hugging Face (cached under `~/.cache/huggingface/`).

## 🛠️ Advanced

### Custom Hotkey

Run the key detector to find your preferred key:

```bash
source venv/bin/activate
python detect_key.py
```

Press your desired key, note the number, and update `hotkey_code` in `config.yaml`.

### Build Standalone Binary

Create a distributable binary (no Python required):

```bash
./build.sh
```

The binary will be created in `dist/VoiceTranscriber` (~500MB).

### Distribution

To share with others:
1. Run `./build.sh` to create binary
2. Copy `dist/VoiceTranscriber` and `config.yaml`
3. Recipients just run `./VoiceTranscriber`

## 🔒 Permissions

macOS will prompt for:

1. **Microphone Access**: System Preferences → Privacy & Security → Microphone
2. **Accessibility**: System Preferences → Privacy & Security → Accessibility (for auto-paste)

## 📝 How It Works

1. Press and hold dictation key
2. Speak into your microphone (you'll hear a beep)
3. Release key when done (another beep)
4. Parakeet transcribes locally
5. Text appears in terminal, copies to clipboard, and auto-pastes
6. Continue working!

## 🐛 Troubleshooting

**Audio beeps don't work?**
- Set `audio_feedback: false` in config.yaml

**Auto-paste not working?**
- Grant Accessibility permissions in System Preferences
- Or set `auto_paste: false` in config.yaml and paste manually

**No audio recorded?**
- Check microphone permissions
- Try different microphone in System Preferences → Sound

**Import errors?**
- Make sure you're using the virtual environment: `source venv/bin/activate`
- Or run via `./run.sh` which handles this automatically

**Parakeet model download fails?**
- Check internet connection
- Try a smaller model (e.g. `nvidia/parakeet-tdt_ctc-110m`) in config.yaml
- Hugging Face Hub access may be required — run `huggingface-cli login` if you hit auth errors

## 📦 Requirements

- macOS
- Python 3.9+
- ffmpeg (installed by install.sh)
- Microphone access

## 🏗️ Project Structure

```
voice-transcriber/
├── transcribe.py       # Main application
├── config.yaml         # Configuration file
├── detect_key.py       # Utility to find key codes
├── install.sh          # Installation script
├── build.sh            # Build standalone binary
├── run.sh              # Run the app
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎯 Use Cases

- **Coding**: Dictate code comments or documentation
- **Writing**: Quickly capture thoughts and ideas
- **Meetings**: Transcribe quick notes without typing
- **Messaging**: Dictate messages in any app
- **Accessibility**: Alternative input method

## 📄 License

MIT License - Feel free to use, modify, and distribute!

## 🙏 Credits

Built with:
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) - Speech recognition toolkit
- [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) - ASR models
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) - Audio recording
- [pynput](https://github.com/moses-palmer/pynput) - Keyboard control
- [PyInstaller](https://www.pyinstaller.org/) - Binary packaging

---

Made with ❤️ for efficient voice transcription
