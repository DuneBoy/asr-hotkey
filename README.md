# Speech-to-Text - Background Service for Linux/Wayland

A simple push-to-talk speech-to-text system for Linux/Wayland using the multi-language [parakeet ASR model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) from NVIDIA and the [onnx-asr](https://github.com/istupakov/onnx-asr) library. 

⚠️ This script was vibecoded during lunch break out of personal need, don't expect it to be free from errors. I'm just sharing it so you can save an hour or two. Tested on omarchy.

- Press F12 to start recording, press again to transcribe and paste.
- Super fast on low end hardware with only CPU
- Multilanguage - you can even switch language while speaking.
- Everything runs local, forever forever
- Model is kept in memory for fast response
- System tray icon shows status (gray=idle, red=recording, blue=transcribing)
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

### OpenRouter API key (optional)

Some optional features use an LLM via OpenRouter (ask/transform/insert). To enable them, set your API key via a user environment file loaded by the systemd service.

1) Create the environment file:

```bash
mkdir -p ~/.config/parakeet
printf 'OPENROUTER_API_KEY=sk-or-...\n' > ~/.config/parakeet/parakeet.env
chmod 600 ~/.config/parakeet/parakeet.env
```

2) Reload and restart the user service:

```bash
systemctl --user daemon-reload
systemctl --user restart parakeet-dbus.service
```

3) Verify the key is picked up (look for a log line like "API key found"):

```bash
journalctl --user -u parakeet-dbus.service -f
```

If the key is not set, LLM features will gracefully fall back (no external call).

## Configure Hyprland Hotkey

Add this line to your Hyprland hotkeys config:

```
bind = , F12, exec, dbus-send --session --dest=com.parakeet.Transcribe --type=method_call /com/parakeet/Transcribe com.parakeet.Transcribe.Toggle
```

### Optional: Hotkeys for LLM modes

If you enabled the OpenRouter API key, you can bind additional hotkeys for the LLM-powered modes:

```
# Ask mode: speak a question/command; pastes concise answer
bind = , F9,  exec, dbus-send --session --dest=com.parakeet.Transcribe --type=method_call /com/parakeet/Transcribe com.parakeet.Transcribe.ToggleAsk

# Transform mode: speak an instruction to transform current clipboard text; pastes transformed text
bind = , F10, exec, dbus-send --session --dest=com.parakeet.Transcribe --type=method_call /com/parakeet/Transcribe com.parakeet.Transcribe.ToggleTransform

# Insert mode: speak text to insert into clipboard content; pastes merged text
bind = , F11, exec, dbus-send --session --dest=com.parakeet.Transcribe --type=method_call /com/parakeet/Transcribe com.parakeet.Transcribe.ToggleLLM
```

## Usage

1. Press **F12** to start recording
2. Speak your text
3. Press **F12** again to stop recording
4. The transcription will be automatically pasted at your cursor position using Ctrl+Shift+V

## LLM Features (optional)

Parakeet includes three optional LLM-powered workflows that build on top of local speech recognition. These require an OpenRouter API key and internet connectivity. Without a key, these features are skipped gracefully.

- Ask mode (`ToggleAsk`)
  - Speak a question or instruction. The LLM returns a short, direct answer which is pasted.
  - Example: “Translate ‘good morning’ to French.” → pastes “bonjour”.

- Transform mode (`ToggleTransform`)
  - Speak an instruction; the LLM applies it to your current clipboard content and pastes the result.
  - Example: “Summarize in one sentence.”

- Insert mode (`ToggleLLM`)
  - Speak text to insert into the clipboard content. The LLM merges it in-place (or at the end), adjusting punctuation/casing minimally.
  - Example: Insert a short sentence into a paragraph on your clipboard.

### D-Bus commands

You can trigger these directly via D-Bus as well:

```bash
# Ask mode
dbus-send --session --dest=com.parakeet.Transcribe --type=method_call \
  /com/parakeet/Transcribe com.parakeet.Transcribe.StartRecordingAsk
dbus-send --session --dest=com.parakeet.Transcribe --type=method_call \
  /com/parakeet/Transcribe com.parakeet.Transcribe.StopRecordingAsk

# Transform mode
dbus-send --session --dest=com.parakeet.Transcribe --type=method_call \
  /com/parakeet/Transcribe com.parakeet.Transcribe.StartRecordingTransform
dbus-send --session --dest=com.parakeet.Transcribe --type=method_call \
  /com/parakeet/Transcribe com.parakeet.Transcribe.StopRecordingTransform

# Insert mode
dbus-send --session --dest=com.parakeet.Transcribe --type=method_call \
  /com/parakeet/Transcribe com.parakeet.Transcribe.StartRecordingLLM
dbus-send --session --dest=com.parakeet.Transcribe --type=method_call \
  /com/parakeet/Transcribe com.parakeet.Transcribe.StopRecordingLLM
```

### Model and privacy

- Default OpenRouter model: `google/gemini-2.5-flash-preview-09-2025`.
- You may change the model by editing the `model` field in `parakeet_dbus.py` where the OpenRouter request payload is constructed.
- Using these features sends your prompt and relevant text to the selected OpenRouter model provider; avoid sensitive data.

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
