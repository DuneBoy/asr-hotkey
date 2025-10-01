# Parakeet Speech-to-Text - Background Service 

A simple push-to-talk speech-to-text system for Linux/Wayland using the multi-language [parakeet ASR model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) from NVIDIA and the [onnx-asr](https://github.com/istupakov/onnx-asr) library. 

⚠️ This script was vibecoded during lunch break out of personal need, don't expect it to be free from errors. I'm just sharing it so you can save an hour or two. Tested on omarchy.

- Press F12 to start recording, press again to transcribe and paste.
- Super fast on low end hardware with only CPU
- Multilanguage - you can even switch language while speaking.
- Everything runs local, forever forever
- Model is kept in memory for fast response
- Use any model from the model that onnx-asr supports by changing the model name the python file.
- All credits go to the onnx-asr project and NVIDIA, thanks for making local ASR so great!

## Requirements

- Python 3.12+
- uv (Python package manager)
- ydotool (for keyboard automation)
- Hyprland (or another Wayland compositor)
- wl-copy (wayland clipboard utility)
- PyAudio dependencies

## Installation

### 1. Install system dependencies

```bash
# Arch Linux
sudo pacman -S python-pyaudio ydotool wl-clipboard

# Ubuntu/Debian
sudo apt install python3 python3-pyaudio ydotool wl-clipboard
```

### 2. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Clone and setup the project

```bash
git clone <repository-url> ~/.local/share/parakeet
cd ~/.local/share/parakeet
uv sync
```

This will automatically install all Python dependencies including:
- onnx-asr
- PyGObject
- pydbus
- pyaudio

## Setup Services

### 1. Install ydotoold system service

ydotoold needs to run as root to simulate keyboard input.

```bash
sudo cp ydotoold.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ydotoold.service
sudo systemctl start ydotoold.service
```

Verify it's running:
```bash
sudo systemctl status ydotoold.service
```

### 2. Install Parakeet D-Bus user service

```bash
cp parakeet-dbus.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable parakeet-dbus.service
systemctl --user start parakeet-dbus.service
```

Verify it's running:
```bash
systemctl --user status parakeet-dbus.service
```

The first startup will take a while as it downloads and loads the Parakeet model (~2.8GB memory usage).

## Configure Hyprland Hotkey

Add this line to your Hyprland hotkeys config:

```
bind = , F12, exec, dbus-send --session --dest=com.parakeet.Transcribe --type=method_call /com/parakeet/Transcribe com.parakeet.Transcribe.Toggle
```

## Usage

1. Press **F12** to start recording
2. Speak your text
3. Press **F12** again to stop recording
4. The transcription will be automatically pasted at your cursor position using Ctrl+Shift+V

## Troubleshooting

### Check service logs

Parakeet service:
```bash
journalctl --user -u parakeet-dbus.service -f
```

ydotoold service:
```bash
sudo journalctl -u ydotoold.service -f
```

### Test D-Bus service manually

```bash
# Start recording
dbus-send --session --dest=com.parakeet.Transcribe --type=method_call /com/parakeet/Transcribe com.parakeet.Transcribe.StartRecording

# Stop recording (will transcribe and paste)
dbus-send --session --dest=com.parakeet.Transcribe --type=method_call /com/parakeet/Transcribe com.parakeet.Transcribe.StopRecording
```

### Pasting not working

Make sure ydotoold is running:
```bash
ls -la /run/ydotool/socket
```

You should see a socket file. If not, restart ydotoold:
```bash
sudo systemctl restart ydotoold.service
```

### Recording too short error

Speak for at least 1 second. Very short recordings are ignored.

## Architecture

- **parakeet_dbus.py** - Main D-Bus service that handles recording and transcription
- **ydotoold.service** - System service for keyboard automation (runs as root)
- **parakeet-dbus.service** - User service for the transcription D-Bus interface

## Model Information

Uses the NVIDIA Parakeet TDT 0.6B v3 model via onnx-asr:
- Automatic download on first run
- ~2.8GB memory usage when loaded
- Multilanguage - you can even switch language while speaking.

