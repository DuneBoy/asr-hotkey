#!/usr/bin/env python3
import argparse
import contextlib
import enum
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

import onnx_asr
import pyaudio

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

    colors = {
        "idle": (100, 100, 100),       # Gray
        "recording": (255, 0, 0),       # Bright Red
        "transcribing": (0, 120, 255)   # Bright Blue
    }

    color = colors.get(state, colors["idle"])

    icon_dir = os.path.expanduser("~/AUR/asr-hotkey/icons")
    os.makedirs(icon_dir, exist_ok=True)

    icon_path = os.path.join(icon_dir, f"parakeet-{state}.png")

    img = Image.new('RGBA', (48, 48), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 8, 40, 40], fill=color)
    img.save(icon_path, 'PNG')

    return icon_path


class ServiceState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "transcribing"


class TranscriptionMode(enum.Enum):
    NORMAL = "normal"
    LLM_INSERT = "llm_insert"
    LLM_TRANSFORM = "llm_transform"
    LLM_ASK = "llm_ask"


class TranscriptionSession:
    """Encapsulates a single end-to-end transcription session.

    Coordinates recording and transcription in a single background thread to
    avoid multi-threaded races. Reports state transitions and final results via
    callbacks supplied by the service.
    """

    def __init__(
        self,
        *,
        service_ref,
        mode: TranscriptionMode,
        streaming: bool,
        window_seconds: float | None = None,
        slide_seconds: float | None = None,
        start_delay_seconds: float | None = None,
    ):
        self._service = service_ref
        self.mode = mode
        self.streaming = streaming and HAS_STREAMING

        # Streaming parameters
        self.window_seconds = window_seconds or 7.0
        self.slide_seconds = slide_seconds or 3.0
        self.start_delay_seconds = start_delay_seconds or 6.0

        # Runtime state
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # For standard mode
        self._frames: list[bytes] = []

        # For streaming mode
        self._streaming_recorder: StreamingRecorder | None = None
        self._chunk_merger: ChunkMerger | None = None

    # ---- Public API ----
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="TranscriptionSession", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

    # ---- Internals ----
    def _run(self) -> None:
        try:
            # Recording phase
            self._service._on_session_state_change(ServiceState.RECORDING)
            if self.streaming:
                self._run_streaming()
            else:
                self._run_standard()
        except Exception as e:
            print(f"Session error: {e}")
            # Ensure we return to idle
            try:
                self._service._on_session_state_change(ServiceState.IDLE)
            except Exception:
                pass

    def _run_standard(self) -> None:
        # Blockingly record using a dedicated thread loop to avoid callback races
        pa = self._service.p
        fmt = self._service.FORMAT
        channels = self._service.CHANNELS
        rate = self._service.supported_rate
        chunk = self._service.CHUNK

        stream = None
        try:
            stream = pa.open(
                format=fmt,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk,
            )
            while not self._stop_event.is_set():
                data = stream.read(chunk, exception_on_overflow=False)
                self._frames.append(data)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    print(f"Warning: error closing stream: {e}")

        # Processing phase
        if not self._frames or len(self._frames) < 5:
            print("Recording too short, ignoring...")
            self._service._on_session_state_change(ServiceState.IDLE)
            return

        self._service._on_session_state_change(ServiceState.PROCESSING)

        audio_data = b"".join(self._frames)
        # Save raw recording
        wf = wave.open(self._service.RAW_FILE, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(pa.get_sample_size(fmt))
        wf.setframerate(rate)
        wf.writeframes(audio_data)
        wf.close()

        # Resample if needed
        if rate != self._service.MODEL_RATE:
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', self._service.RAW_FILE, '-ar', str(self._service.MODEL_RATE), self._service.TEMP_FILE],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
            except Exception as e:
                print(f"Error during resampling: {e}")
                self._service._on_session_state_change(ServiceState.IDLE)
                return
        else:
            shutil.copy(self._service.RAW_FILE, self._service.TEMP_FILE)

        # Recognize
        print("Transcribing...")
        result = self._service.model.recognize(self._service.TEMP_FILE)
        result = result.strip() if result else ""
        self._service._on_session_final(result, self.mode)

    def _run_streaming(self) -> None:
        # Initialize streaming helpers
        self._streaming_recorder = StreamingRecorder(
            audio_instance=self._service.p,
            sample_rate=self._service.MODEL_RATE,
            channels=self._service.CHANNELS,
            window_seconds=self.window_seconds,
            slide_seconds=self.slide_seconds,
            start_delay_seconds=self.start_delay_seconds,
        )
        self._chunk_merger = ChunkMerger()

        # Start audio
        self._streaming_recorder.start_recording()
        print("🔄 Streaming processor started")
        print(f"   Window: {self.window_seconds}s, Slide: {self.slide_seconds}s")
        overlap = self.window_seconds - self.slide_seconds
        print(f"   Overlap: {overlap}s")

        # Consume chunks while recording or until queue drains
        while not self._stop_event.is_set() or not self._streaming_recorder.processing_queue.empty():
            chunk_np = self._streaming_recorder.get_next_chunk()
            if chunk_np is None:
                time.sleep(0.1)
                continue
            # Transcribe this chunk
            chunk_file = self._streaming_recorder.save_chunk_to_file(chunk_np, int(time.time()))
            if not chunk_file:
                continue
            try:
                text = self._service.model.recognize(chunk_file)
                text = text.strip() if text else ""
            finally:
                try:
                    os.remove(chunk_file)
                except Exception:
                    pass
            if text:
                merged = self._chunk_merger.add_chunk(text, is_final=False)
                print(f"  ⚡ Chunk: {text[:50]}...")
                print(f"     Merged ({len(merged.split())} words): ...{merged[-80:]}")

        print("🔄 Streaming processor stopped")

        # Stop recording and finalize
        _ = self._streaming_recorder.stop_recording()
        # If nothing processed, bail out
        final_text = self._chunk_merger.get_result() if self._chunk_merger else ""
        if not final_text:
            print("No text transcribed")
            self._service._on_session_state_change(ServiceState.IDLE)
            return

        # Move to processing (for UI purposes) right before finalization
        self._service._on_session_state_change(ServiceState.PROCESSING)
        self._service._on_session_final(final_text, self.mode)

class ParakeetService:
    """
    <node>
        <interface name='com.parakeet.Transcribe'>
            <method name='StartRecording'/>
            <method name='StopRecording'/>
            <method name='Toggle'/>
            <method name='StartRecordingLLM'/>
            <method name='StopRecordingLLM'/>
            <method name='ToggleLLM'/>
            <method name='StartRecordingTransform'/>
            <method name='StopRecordingTransform'/>
            <method name='ToggleTransform'/>
            <method name='StartRecordingAsk'/>
            <method name='StopRecordingAsk'/>
            <method name='ToggleAsk'/>
        </interface>
    </node>
    """

    def __init__(self, streaming_mode=False, window_seconds=7.0, slide_seconds=3.0, start_delay_seconds=6.0):
        print("Loading model... This may take a few seconds...")
        
        # Check available ONNX Runtime providers
        import onnxruntime as rt
        available_providers = rt.get_available_providers()
        print(f"Available ONNX Runtime providers: {available_providers}")
        
        # Prioritize GPU (CUDA) over CPU for inference
        self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("Model loaded!")
        
        # Verify which provider is being used
        if 'CUDAExecutionProvider' in available_providers:
            print("✓ GPU (CUDA) acceleration enabled")
        else:
            print("⚠ Warning: CUDA not available, falling back to CPU")

        # Audio settings
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.MODEL_RATE = 16000
        self.TEMP_FILE = "/tmp/parakeet_recording.wav"
        self.RAW_FILE = "/tmp/parakeet_recording_raw.wav"

        # Core state
        self._lock = threading.RLock()
        self._state: ServiceState = ServiceState.IDLE
        self._session: TranscriptionSession | None = None

        # UI
        self.tray_icon = None

        with suppress_stderr():
            self.p = pyaudio.PyAudio()

        # Streaming configuration
        self.streaming_mode = bool(streaming_mode and HAS_STREAMING)
        self._window_seconds = window_seconds
        self._slide_seconds = slide_seconds
        self._start_delay_seconds = start_delay_seconds
        if streaming_mode and not HAS_STREAMING:
            print("Warning: Streaming mode requested but components not available, falling back to standard mode")

        # Detect usable input rate once
        print("Detecting supported sample rate...")
        self.supported_rate = None
        for test_rate in [48000, 44100, 32000, 22050, 16000, 8000]:
            try:
                with suppress_stderr():
                    s = self.p.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=test_rate,
                        input=True,
                        frames_per_buffer=self.CHUNK,
                    )
                    s.close()
                self.supported_rate = test_rate
                print(f"Using sample rate: {self.supported_rate} Hz")
                break
            except Exception:
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
            menu = Gtk.Menu()
            toggle_item = Gtk.MenuItem(label="Toggle Recording")
            toggle_item.connect("activate", lambda _: self._tray_toggle())
            menu.append(toggle_item)
            menu.append(Gtk.SeparatorMenuItem())
            quit_item = Gtk.MenuItem(label="Quit")
            quit_item.connect("activate", lambda _: self._tray_quit())
            menu.append(quit_item)
            menu.show_all()

            icon_path = os.path.expanduser("~/AUR/asr-hotkey/icons")
            create_icon_file("idle")
            create_icon_file("recording")
            create_icon_file("transcribing")

            self.tray_icon = AppIndicator.Indicator.new(
                "parakeet",
                "parakeet-idle",
                AppIndicator.IndicatorCategory.APPLICATION_STATUS,
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
        with self._lock:
            if self._state == ServiceState.RECORDING and self._session:
                self._session.request_stop()
        if self.tray_icon:
            self.tray_icon.set_status(AppIndicator.IndicatorStatus.PASSIVE)
        os._exit(0)

    def _update_tray_icon(self, state: ServiceState):
        """Update tray icon based on state."""
        if not self.tray_icon:
            return
        if state == ServiceState.RECORDING:
            self.tray_icon.set_icon("parakeet-recording")
            self.tray_icon.set_title("Parakeet (Recording)")
        elif state == ServiceState.PROCESSING:
            self.tray_icon.set_icon("parakeet-transcribing")
            self.tray_icon.set_title("Parakeet (Transcribing)")
        else:
            self.tray_icon.set_icon("parakeet-idle")
            self.tray_icon.set_title("Parakeet (Idle)")

    # ---- Session callbacks ----
    def _on_session_state_change(self, new_state: ServiceState) -> None:
        with self._lock:
            # Ignore late callbacks from old sessions
            if not self._session:
                return
            self._state = new_state
            self._update_tray_icon(new_state)

    def _on_session_final(self, text: str, mode: TranscriptionMode) -> None:
        """Handle final text from a session: optional LLM, clipboard, reset state."""
        processed_text = (text or '').strip()
        if not processed_text:
            print("No transcription result")
        else:
            # Optional LLM steps
            print(f"DEBUG: mode={mode}")
            try:
                if mode == TranscriptionMode.LLM_ASK:
                    processed_text = self._ask_llm(processed_text)
                elif mode == TranscriptionMode.LLM_TRANSFORM:
                    processed_text = self._transform_with_llm(processed_text)
                elif mode == TranscriptionMode.LLM_INSERT:
                    processed_text = self._process_with_llm(processed_text)
            except Exception as e:
                print(f"LLM processing error: {e}")

            # Copy to clipboard and paste
            try:
                subprocess.run(['wl-copy'], input=(processed_text + ' ').encode(), check=True)
                print("✓ Copied to clipboard")
                subprocess.run(['ydotool', 'key', '29:1', '42:1', '47:1', '47:0', '42:0', '29:0'], check=True)
                print("✓ Pasted")
            except Exception as e:
                print(f"Could not paste result: {e}. Result: {processed_text}")

        # Reset service state
        with self._lock:
            self._session = None
            self._state = ServiceState.IDLE
            self._update_tray_icon(self._state)

    # ---- D-Bus methods ----
    def StartRecording(self):
        self._start_session(TranscriptionMode.NORMAL)

    def StopRecording(self):
        with self._lock:
            if self._state != ServiceState.RECORDING or not self._session:
                print("Not recording")
                return
            print("Stopping recording...")
            self._session.request_stop()

    # ---- Core control helpers ----
    def _start_session(self, mode: TranscriptionMode) -> None:
        with self._lock:
            if self._state != ServiceState.IDLE or self._session is not None:
                print(f"Busy (state={self._state.value}), ignoring start request")
                return
            print("Starting recording...")
            self._session = TranscriptionSession(
                service_ref=self,
                mode=mode,
                streaming=self.streaming_mode,
                window_seconds=self._window_seconds,
                slide_seconds=self._slide_seconds,
                start_delay_seconds=self._start_delay_seconds,
            )
            self._state = ServiceState.RECORDING
            self._update_tray_icon(self._state)
            self._session.start()

    def _ask_llm(self, question):
        """Ask the LLM a direct question and get a concise answer.

        Args:
            question: The question/instruction to ask

        Returns:
            The answer from the LLM, or the original question if processing fails
        """
        try:
            print("\n" + "="*60)
            print("💭 LLM ASK MODE DEBUG")
            print("="*60)

            # Get API key
            print("\n🔑 Checking API key...")
            api_key = os.environ.get('OPENROUTER_API_KEY')
            if not api_key:
                print("❌ OPENROUTER_API_KEY not set, returning question as-is")
                return question
            print(f"   API key found: {api_key[:20]}...")

            # Format the prompt
            prompt = f"""Answer the following question or instruction concisely and directly. Output ONLY the answer and NOTHING ELSE. No explanations, no preamble, no postamble.

If asking for a command, provide only the command.
If asking for a translation, provide only the translated word/phrase.
If asking for information, provide a brief 1-2 sentence answer.

Question/Instruction: {question}"""

            # Prepare API request
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/parakeet-transcribe",
            }

            data = {
                "model": "google/gemini-2.5-flash-preview-09-2025",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            print("\n🌐 Calling OpenRouter API...")
            print(f"   URL: {url}")
            print(f"   Model: {data['model']}")
            print(f"   Question: {question}")

            # Make API request
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                print(f"   Response status: {response.status}")
                response_data = json.loads(response.read().decode('utf-8'))
                print(f"   Response keys: {list(response_data.keys())}")
                answer = response_data['choices'][0]['message']['content'].strip()
                print(f"\n✅ LLM ask complete")
                print(f"   Answer length: {len(answer)}")
                print(f"   Answer: {answer[:200]}..." if len(answer) > 200 else f"   Answer: {answer}")
                print("="*60 + "\n")
                return answer

        except urllib.error.URLError as e:
            print(f"Error calling OpenRouter API: {e}")
            print("Falling back to original question")
            return question
        except KeyError as e:
            print(f"Error parsing API response: {e}")
            print("Falling back to original question")
            return question
        except Exception as e:
            print(f"Unexpected error in LLM ask: {e}")
            print("Falling back to original question")
            return question

    def _transform_with_llm(self, instruction):
        """Transform clipboard content using transcription as instruction.

        Args:
            instruction: The transcribed instruction (e.g., "correct the spelling")

        Returns:
            The transformed text from the LLM, or the original clipboard if processing fails
        """
        try:
            print("\n" + "="*60)
            print("🔄 LLM TRANSFORM MODE DEBUG")
            print("="*60)

            # Get current clipboard content
            print("📋 Reading clipboard...")
            result = subprocess.run(['wl-paste'], capture_output=True, text=True, check=False)
            clipboard_text = result.stdout if result.returncode == 0 else ""
            print(f"   Return code: {result.returncode}")
            print(f"   Clipboard content length: {len(clipboard_text)}")
            print(f"   Clipboard content: {clipboard_text[:100]}..." if len(clipboard_text) > 100 else f"   Clipboard content: {clipboard_text}")

            # If clipboard is empty, return instruction as-is
            if not clipboard_text.strip():
                print("⚠️  Clipboard empty, returning instruction as-is")
                return instruction

            # Get API key
            print("\n🔑 Checking API key...")
            api_key = os.environ.get('OPENROUTER_API_KEY')
            if not api_key:
                print("❌ OPENROUTER_API_KEY not set, returning clipboard as-is")
                return clipboard_text
            print(f"   API key found: {api_key[:20]}...")

            # Format the prompt
            prompt = f"""You will receive an instruction and a text. Apply the instruction to the text or answer the question asked about the text, give short and concise answers in 1 - 2 sentences. Output ONLY the transformed text OR your answer and NOTHING ELSE. No explanations, no preamble, no postamble.

Instruction: {instruction}

Text:
{clipboard_text}"""

            # Prepare API request
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/parakeet-transcribe",
            }

            data = {
                "model": "google/gemini-2.5-flash-preview-09-2025",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            print("\n🌐 Calling OpenRouter API...")
            print(f"   URL: {url}")
            print(f"   Model: {data['model']}")
            print(f"   Instruction: {instruction}")

            # Make API request
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                print(f"   Response status: {response.status}")
                response_data = json.loads(response.read().decode('utf-8'))
                print(f"   Response keys: {list(response_data.keys())}")
                transformed_text = response_data['choices'][0]['message']['content'].strip()
                print(f"\n✅ LLM transform complete")
                print(f"   Transformed text length: {len(transformed_text)}")
                print(f"   Transformed text: {transformed_text[:200]}..." if len(transformed_text) > 200 else f"   Transformed text: {transformed_text}")
                print("="*60 + "\n")
                return transformed_text

        except urllib.error.URLError as e:
            print(f"Error calling OpenRouter API: {e}")
            print("Falling back to original clipboard")
            return clipboard_text
        except KeyError as e:
            print(f"Error parsing API response: {e}")
            print("Falling back to original clipboard")
            return clipboard_text
        except Exception as e:
            print(f"Unexpected error in LLM transform: {e}")
            print("Falling back to original clipboard")
            return clipboard_text

    def _process_with_llm(self, transcription):
        """Process transcription with LLM to insert it into clipboard content.

        Args:
            transcription: The transcribed text to insert

        Returns:
            The processed text from the LLM, or the original transcription if processing fails
        """
        try:
            print("\n" + "="*60)
            print("🤖 LLM PROCESSING DEBUG")
            print("="*60)

            # Get current clipboard content
            print("📋 Reading clipboard...")
            result = subprocess.run(['wl-paste'], capture_output=True, text=True, check=False)
            existing_text = result.stdout if result.returncode == 0 else ""
            print(f"   Return code: {result.returncode}")
            print(f"   Clipboard content length: {len(existing_text)}")
            print(f"   Clipboard content: {existing_text[:100]}..." if len(existing_text) > 100 else f"   Clipboard content: {existing_text}")

            # If clipboard is empty, just return the transcription
            if not existing_text.strip():
                print("⚠️  Clipboard empty, returning transcription as-is")
                return transcription

            # Get API key from environment or use hardcoded fallback
            print("\n🔑 Checking API key...")
            api_key = os.environ.get('OPENROUTER_API_KEY')
            if not api_key:
                print("❌ OPENROUTER_API_KEY not set, returning transcription as-is")
                return transcription
            print(f"   API key found: {api_key[:20]}...")

            # Format the prompt
            prompt = f"""I'm giving you a short piece of text, as well as another short text that should be inserted into the first text. If you find an underline with spaces around in the text, the insertion should go there, if not, the insertion should go at the end of the text. Make the insertion fit into the text or at the end. Especially regarding punctuation marks and capitalization, you can also make small changes to words around the insertion to make it fit better, but only if necessary, keep the first text as close to the original as possible.
The insertion is a transcript that is machine created and might contain words that were misunderstood. Sometimes there are specialized terms that can be corrected with the context from the rest of the text, check the existing text closely for names or specialized terms that could be wrong in the transcript. Fix that and repair other obvious artifacts of the transcription but keep as close to the original as possible.
Keep the overall tone academic, exclamation marks are unusual. Output only the final text and NOTHING ELSE.

<text>
{existing_text}
</text>

<insertion>
{transcription}
</insertion>"""

            # Prepare API request
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/parakeet-transcribe",
            }

            data = {
                "model": "google/gemini-2.5-flash-preview-09-2025",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            print("\n🌐 Calling OpenRouter API...")
            print(f"   URL: {url}")
            print(f"   Model: {data['model']}")
            print(f"   Transcription to insert: {transcription[:100]}...")

            # Make API request
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                print(f"   Response status: {response.status}")
                response_data = json.loads(response.read().decode('utf-8'))
                print(f"   Response keys: {list(response_data.keys())}")
                processed_text = response_data['choices'][0]['message']['content'].strip()
                print(f"\n✅ LLM processing complete")
                print(f"   Processed text length: {len(processed_text)}")
                print(f"   Processed text: {processed_text[:200]}..." if len(processed_text) > 200 else f"   Processed text: {processed_text}")
                print("="*60 + "\n")
                return processed_text

        except urllib.error.URLError as e:
            print(f"Error calling OpenRouter API: {e}")
            print("Falling back to original transcription")
            return transcription
        except KeyError as e:
            print(f"Error parsing API response: {e}")
            print("Falling back to original transcription")
            return transcription
        except Exception as e:
            print(f"Unexpected error in LLM processing: {e}")
            print("Falling back to original transcription")
            return transcription

    # Legacy helper removed in favor of TranscriptionSession

    def Toggle(self):
        with self._lock:
            if self._state == ServiceState.RECORDING:
                self.StopRecording()
            else:
                self.StartRecording()

    def StartRecordingLLM(self):
        """Start recording with LLM insert mode enabled."""
        print("🤖 LLM insert mode ENABLED")
        self._start_session(TranscriptionMode.LLM_INSERT)

    def StopRecordingLLM(self):
        """Stop recording (applies if currently recording)."""
        self.StopRecording()

    def ToggleLLM(self):
        """Toggle recording with LLM insert mode."""
        with self._lock:
            if self._state == ServiceState.RECORDING:
                self.StopRecording()
            else:
                self.StartRecordingLLM()

    def StartRecordingTransform(self):
        """Start recording with LLM transform mode enabled."""
        print("🔄 Transform mode ENABLED")
        self._start_session(TranscriptionMode.LLM_TRANSFORM)

    def StopRecordingTransform(self):
        """Stop recording (applies if currently recording)."""
        self.StopRecording()

    def ToggleTransform(self):
        """Toggle recording with LLM transform mode."""
        with self._lock:
            if self._state == ServiceState.RECORDING:
                self.StopRecordingTransform()
            else:
                self.StartRecordingTransform()

    def StartRecordingAsk(self):
        """Start recording with LLM ask mode enabled."""
        print("💭 Ask mode ENABLED")
        self._start_session(TranscriptionMode.LLM_ASK)

    def StopRecordingAsk(self):
        """Stop recording (applies if currently recording)."""
        self.StopRecording()

    def ToggleAsk(self):
        """Toggle recording with LLM ask mode."""
        with self._lock:
            if self._state == ServiceState.RECORDING:
                self.StopRecordingAsk()
            else:
                self.StartRecordingAsk()

if __name__ == '__main__':
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
    parser.add_argument('--stream', action='store_true', help='Enable streaming mode with sliding window transcription')
    parser.add_argument('--window', type=float, default=7.0, help='Chunk window size in seconds (default: 7.0)')
    parser.add_argument('--slide', type=float, default=3.0, help='Slide interval in seconds (default: 3.0)')
    parser.add_argument('--delay', type=float, default=6.0, help='Delay before starting streaming in seconds (default: 6.0)')

    args = parser.parse_args()

    print("Starting Parakeet D-Bus service...")
    bus = SessionBus()
    service = ParakeetService(
        streaming_mode=args.stream,
        window_seconds=args.window,
        slide_seconds=args.slide,
        start_delay_seconds=args.delay,
    )

    bus.publish("com.parakeet.Transcribe", service)
    print("D-Bus service published at com.parakeet.Transcribe")
    print("Ready!")

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nExiting...")
