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
import argparse
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

# Import streaming components
try:
    from streaming_recorder import StreamingRecorder
    from chunk_merger import ChunkMerger
    HAS_STREAMING = True
except ImportError as e:
    HAS_STREAMING = False
    print(f"Warning: Streaming components not available ({e})")

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

    def __init__(self, streaming_mode=False, window_seconds=7.0, slide_seconds=3.0, start_delay_seconds=6.0):
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

        # Streaming mode settings
        self.streaming_mode = streaming_mode and HAS_STREAMING
        self.streaming_recorder = None
        self.chunk_merger = None
        self.transcription_thread = None
        self.is_transcribing = False
        self.chunk_counter = 0

        if self.streaming_mode:
            print(f"Streaming mode enabled (window: {window_seconds}s, slide: {slide_seconds}s)")
            self.streaming_recorder = StreamingRecorder(
                audio_instance=self.p,
                sample_rate=self.MODEL_RATE,
                channels=self.CHANNELS,
                window_seconds=window_seconds,
                slide_seconds=slide_seconds,
                start_delay_seconds=start_delay_seconds,
            )
            self.chunk_merger = ChunkMerger()
        elif streaming_mode and not HAS_STREAMING:
            print("Warning: Streaming mode requested but components not available, using standard mode")

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
        self._update_tray_icon("recording")

        if self.streaming_mode:
            # Use streaming recorder
            self.chunk_counter = 0
            self.chunk_merger.reset()
            self.is_transcribing = True

            # Start the streaming recorder
            self.streaming_recorder.start_recording()

            # Start transcription thread
            self.transcription_thread = threading.Thread(
                target=self._streaming_transcription_loop,
                daemon=True
            )
            self.transcription_thread.start()
        else:
            # Use standard recording
            self.audio_frames = []
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

        if self.streaming_mode:
            # Stop streaming recorder
            audio_data = self.streaming_recorder.stop_recording()

            if not audio_data or len(audio_data) < self.supported_rate * 0.5:  # Less than 0.5s
                print("Recording too short, ignoring...")
                self.is_transcribing = False
                self._update_tray_icon("idle")
                return

            # Signal transcription thread to finish
            self.is_transcribing = False

            # Wait for transcription thread to process remaining chunks
            if self.transcription_thread:
                print("⏳ Processing remaining chunks...")
                self.transcription_thread.join(timeout=30.0)

            # Get final merged result
            self._finalize_transcription()
        else:
            # Standard mode
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

    def _streaming_transcription_loop(self):
        """Process audio chunks with sliding window approach."""
        print("🔄 Streaming processor started")
        print(f"   Window: {self.streaming_recorder.window_seconds}s, Slide: {self.streaming_recorder.slide_seconds}s")
        overlap = self.streaming_recorder.window_seconds - self.streaming_recorder.slide_seconds
        print(f"   Overlap: {overlap}s")

        while self.is_transcribing or not self.streaming_recorder.processing_queue.empty():
            chunk_np = self.streaming_recorder.get_next_chunk()

            if chunk_np is not None:
                self.chunk_counter += 1
                is_last = not self.is_transcribing and self.streaming_recorder.processing_queue.empty()
                self._transcribe_chunk(chunk_np, self.chunk_counter, is_last_chunk=is_last)
            else:
                # No chunk available yet
                time.sleep(0.1)

        print("🔄 Streaming processor stopped")

    def _transcribe_chunk(self, chunk_np, chunk_id, is_last_chunk=False):
        """Transcribe a single audio chunk and merge with existing result."""
        try:
            # Save chunk to file
            chunk_file = self.streaming_recorder.save_chunk_to_file(chunk_np, chunk_id)
            if not chunk_file:
                return None

            # Transcribe
            print(f"  ⚡ Transcribing chunk {chunk_id}...")
            result = self.model.recognize(chunk_file)
            text = result.strip() if result else ""

            # Clean up chunk file
            try:
                os.remove(chunk_file)
            except:
                pass

            if text:
                # Merge with existing result using overlap detection
                merged = self.chunk_merger.add_chunk(text, is_final=is_last_chunk)

                # Show progress
                marker = "🏁" if is_last_chunk else "⚡"
                print(f"  {marker} Chunk {chunk_id}: {text[:50]}...")
                print(f"     Merged ({len(merged.split())} words): ...{merged[-80:]}")

                return text
            return None

        except Exception as e:
            print(f"Error transcribing chunk {chunk_id}: {e}")
            return None

    def _finalize_transcription(self):
        """Finalize and output the complete transcription."""
        # Update to transcribing state
        self._update_tray_icon("transcribing")

        # Get the merged result
        final_text = self.chunk_merger.get_result()

        if not final_text:
            print("No text transcribed")
            self._update_tray_icon("idle")
            return

        print(f"\n{'=' * 60}")
        print(f"📝 Final Transcription:")
        print(f"{'=' * 60}")
        print(final_text)
        print(f"{'=' * 60}\n")

        # Copy to clipboard and paste (same as standard mode)
        try:
            # Copy to clipboard (Wayland) with trailing space
            subprocess.run(['wl-copy'], input=(final_text + ' ').encode(), check=True)
            print("✓ Copied to clipboard")

            # Paste using ydotool
            subprocess.run(['ydotool', 'key', '29:1', '42:1', '47:1', '47:0', '42:0', '29:0'], check=True)
            print("✓ Pasted")
        except Exception as e:
            print(f"Could not paste result: {e}. Result: {final_text}")

        # Return to idle state
        self._update_tray_icon("idle")

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
                    # Copy to clipboard (Wayland) with trailing space
                    subprocess.run(['wl-copy'], input=(result + ' ').encode(), check=True)

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Parakeet D-Bus transcription service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard mode (default)
  python parakeet_dbus.py

  # Streaming mode with sliding window
  python parakeet_dbus.py --stream

  # Custom streaming parameters
  python parakeet_dbus.py --stream --window 7 --slide 3 --delay 6

Streaming Mode:
  - Processes audio in overlapping chunks during recording
  - Lower latency after stopping (most transcription already done)
  - Supports infinite recording length
  - Default: 7s windows with 3s slide (4s overlap)
        """
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable streaming mode with sliding window transcription'
    )
    parser.add_argument(
        '--window',
        type=float,
        default=7.0,
        help='Chunk window size in seconds (default: 7.0)'
    )
    parser.add_argument(
        '--slide',
        type=float,
        default=3.0,
        help='Slide interval in seconds (default: 3.0, overlap = window - slide)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=6.0,
        help='Delay before starting streaming in seconds (default: 6.0)'
    )

    args = parser.parse_args()

    print("Starting Parakeet D-Bus service...")
    bus = SessionBus()
    service = ParakeetService(
        streaming_mode=args.stream,
        window_seconds=args.window,
        slide_seconds=args.slide,
        start_delay_seconds=args.delay
    )

    bus.publish("com.parakeet.Transcribe", service)
    print("D-Bus service published at com.parakeet.Transcribe")
    print("Ready!")

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nExiting...")
