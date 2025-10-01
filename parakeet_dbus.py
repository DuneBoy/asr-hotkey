#!/usr/bin/env python3
import onnx_asr
import pyaudio
import wave
import os
import sys
import subprocess
import contextlib
import shutil
import time
import threading
import gi
gi.require_version('Gtk', '3.0')
try:
    gi.require_version('AyatanaAppIndicator3', '0.1')
    from gi.repository import GLib, Gtk, Gdk, AyatanaAppIndicator3 as AppIndicator
    from pydbus import SessionBus
    HAS_APPINDICATOR = True
except (ImportError, ValueError) as e:
    from gi.repository import GLib, Gtk, Gdk
    from pydbus import SessionBus
    HAS_APPINDICATOR = False
    print(f"Warning: AppIndicator not available ({e}), tray icon will not be shown")

@contextlib.contextmanager
def suppress_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

def create_icon_file(state="idle"):
    """Create a colored icon file for the tray based on state.

    Args:
        state: One of "idle", "recording", or "transcribing"
    """
    from PIL import Image, ImageDraw
    import os

    # Color mapping - bright, clear colors
    colors = {
        "idle": (100, 100, 100),       # Gray
        "recording": (255, 0, 0),       # Bright Red
        "transcribing": (0, 120, 255)   # Bright Blue
    }

    color = colors.get(state, colors["idle"])

    # Create icon directory if it doesn't exist
    icon_dir = os.path.expanduser("~/.local/share/parakeet/icons")
    os.makedirs(icon_dir, exist_ok=True)

    # Icon file path
    icon_path = os.path.join(icon_dir, f"parakeet-{state}.png")

    # Always recreate to ensure correct color
    # Create a 48x48 image
    img = Image.new('RGBA', (48, 48), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw simple filled circle
    draw.ellipse([8, 8, 40, 40], fill=color)

    img.save(icon_path, 'PNG')

    return icon_path

class ParakeetService:
    """
    <node>
        <interface name='com.parakeet.Transcribe'>
            <method name='StartRecording'/>
            <method name='StopRecording'/>
            <method name='Toggle'/>
        </interface>
    </node>
    """

    def __init__(self):
        print("Loading model...")
        self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
        print("Model loaded!")

        # Audio settings
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.MODEL_RATE = 16000
        self.TEMP_FILE = "/tmp/parakeet_recording.wav"
        self.RAW_FILE = "/tmp/parakeet_recording_raw.wav"

        self.is_recording = False
        self.audio_frames = []
        self.stream = None
        self.tray_icon = None

        with suppress_stderr():
            self.p = pyaudio.PyAudio()

        # Find supported sample rate
        print("Detecting supported sample rate...")
        self.supported_rate = None
        for test_rate in [48000, 44100, 32000, 22050, 16000, 8000]:
            try:
                with suppress_stderr():
                    test_stream = self.p.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=test_rate,
                        input=True,
                        frames_per_buffer=self.CHUNK
                    )
                    test_stream.close()
                self.supported_rate = test_rate
                print(f"Using sample rate: {self.supported_rate} Hz")
                break
            except:
                continue

        if not self.supported_rate:
            print("Error: Could not find supported sample rate")
            sys.exit(1)

        # Initialize system tray
        self._setup_tray()

    def _setup_tray(self):
        """Setup system tray icon."""
        if not HAS_APPINDICATOR:
            self.tray_icon = None
            return

        try:
            # Create menu
            menu = Gtk.Menu()

            # Toggle item
            toggle_item = Gtk.MenuItem(label="Toggle Recording")
            toggle_item.connect("activate", lambda _: self._tray_toggle())
            menu.append(toggle_item)

            # Separator
            menu.append(Gtk.SeparatorMenuItem())

            # Quit item
            quit_item = Gtk.MenuItem(label="Quit")
            quit_item.connect("activate", lambda _: self._tray_quit())
            menu.append(quit_item)

            menu.show_all()

            # Create indicator
            icon_path = os.path.expanduser("~/.local/share/parakeet/icons")

            # Pre-create all icons
            create_icon_file("idle")
            create_icon_file("recording")
            create_icon_file("transcribing")

            self.tray_icon = AppIndicator.Indicator.new(
                "parakeet",
                "parakeet-idle",
                AppIndicator.IndicatorCategory.APPLICATION_STATUS
            )
            self.tray_icon.set_icon_theme_path(icon_path)
            self.tray_icon.set_attention_icon("parakeet-transcribing")
            self.tray_icon.set_status(AppIndicator.IndicatorStatus.ACTIVE)
            self.tray_icon.set_menu(menu)
            self.tray_icon.set_title("Parakeet (Idle)")

            print("System tray icon initialized")
        except Exception as e:
            print(f"Warning: Could not create system tray icon: {e}")
            print("Continuing without tray icon...")
            self.tray_icon = None

    def _tray_toggle(self):
        """Toggle recording from tray menu."""
        self.Toggle()

    def _tray_quit(self):
        """Quit application from tray menu."""
        if self.is_recording:
            self.StopRecording()
        if self.tray_icon:
            self.tray_icon.set_status(AppIndicator.IndicatorStatus.PASSIVE)
        os._exit(0)

    def _update_tray_icon(self, state):
        """Update the tray icon to reflect current state.

        Args:
            state: One of "idle", "recording", or "transcribing"
        """
        if self.tray_icon:
            if state == "recording":
                self.tray_icon.set_icon("parakeet-recording")
                self.tray_icon.set_title("Parakeet (Recording)")
            elif state == "transcribing":
                self.tray_icon.set_icon("parakeet-transcribing")
                self.tray_icon.set_title("Parakeet (Transcribing)")
            else:
                self.tray_icon.set_icon("parakeet-idle")
                self.tray_icon.set_title("Parakeet (Idle)")

    def StartRecording(self):
        if self.is_recording:
            print("Already recording")
            return

        print("Starting recording...")
        self.is_recording = True
        self.audio_frames = []
        self._update_tray_icon("recording")

        try:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.supported_rate,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.is_recording = False
            self._update_tray_icon("idle")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.audio_frames.append(in_data)
        return (None, pyaudio.paContinue)

    def StopRecording(self):
        if not self.is_recording:
            print("Not recording")
            return

        print("Stopping recording...")
        self.is_recording = False

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Warning: Error closing stream: {e}")
            self.stream = None

        # Check if we have enough audio
        if not self.audio_frames or len(self.audio_frames) < 5:
            print("Recording too short, ignoring...")
            self._update_tray_icon("idle")
            return

        # Copy audio frames before threading
        audio_data = b''.join(self.audio_frames)

        # Run transcription in a separate thread
        transcribe_thread = threading.Thread(target=self._transcribe, args=(audio_data,), daemon=True)
        transcribe_thread.start()

    def _transcribe(self, audio_data):
        """Transcribe audio in a separate thread."""
        try:
            # Update to transcribing state
            self._update_tray_icon("transcribing")

            # Save raw recording
            wf = wave.open(self.RAW_FILE, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.supported_rate)
            wf.writeframes(audio_data)
            wf.close()

            # Resample if needed
            if self.supported_rate != self.MODEL_RATE:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', self.RAW_FILE, '-ar', str(self.MODEL_RATE), self.TEMP_FILE],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
            else:
                shutil.copy(self.RAW_FILE, self.TEMP_FILE)

            # Transcribe
            print("Transcribing...")
            result = self.model.recognize(self.TEMP_FILE)
            print(f"Result: {result}")

            # Copy to clipboard and paste
            if result and result.strip():
                try:
                    # Copy to clipboard (Wayland)
                    subprocess.run(['wl-copy'], input=result.encode(), check=True)

                    # Paste using ydotool
                    subprocess.run(['ydotool', 'key', '29:1', '42:1', '47:1', '47:0', '42:0', '29:0'], check=True)
                except Exception as e:
                    print(f"Could not paste result: {e}. Result: {result}")
            else:
                print("No transcription result")
        except Exception as e:
            print(f"Error during transcription: {e}")
        finally:
            # Always return to idle state after transcription
            self._update_tray_icon("idle")

    def Toggle(self):
        if self.is_recording:
            self.StopRecording()
        else:
            self.StartRecording()

if __name__ == '__main__':
    print("Starting Parakeet D-Bus service...")
    bus = SessionBus()
    service = ParakeetService()

    bus.publish("com.parakeet.Transcribe", service)
    print("D-Bus service published at com.parakeet.Transcribe")
    print("Ready!")

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nExiting...")
