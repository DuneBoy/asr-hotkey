"""
Streaming audio recorder with sliding window processing.
Processes audio chunks in real-time during recording.
"""
import wave
import pyaudio
import queue
import time
import numpy as np
from collections import deque
from datetime import datetime


class StreamingRecorder:
    def __init__(
        self,
        audio_instance,
        sample_rate=16000,
        channels=1,
        window_seconds=7.0,
        slide_seconds=3.0,
        start_delay_seconds=6.0,
    ):
        """
        Initialize streaming recorder with sliding window.

        Args:
            audio_instance: Existing PyAudio instance to use
            sample_rate: Audio sample rate (default: 16000)
            channels: Number of audio channels (default: 1)
            window_seconds: Size of each chunk window (default: 7.0)
            slide_seconds: Interval between chunk starts (default: 3.0)
            start_delay_seconds: Delay before starting to process chunks (default: 6.0)
        """
        self.audio = audio_instance
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = 1024
        self.format = pyaudio.paInt16

        self.stream = None
        self.is_recording = False

        # Streaming parameters
        self.window_seconds = window_seconds
        self.slide_seconds = slide_seconds
        self.start_delay_seconds = start_delay_seconds

        # Calculate samples
        self.window_samples = int(sample_rate * window_seconds)
        self.slide_samples = int(sample_rate * slide_seconds)
        self.start_delay_samples = int(sample_rate * start_delay_seconds)

        # Buffer to hold all recorded audio
        self.all_frames = []
        self.audio_buffer = deque(maxlen=self.window_samples)
        self.samples_recorded = 0

        # Queue for chunks ready to process
        self.processing_queue = queue.Queue()

        self.last_process_time = 0

    def start_recording(self):
        """Start recording audio with streaming."""
        if self.is_recording:
            return

        self.all_frames = []
        self.audio_buffer.clear()
        self.samples_recorded = 0
        self.is_recording = True
        self.last_process_time = time.time()

        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self._audio_callback,
            )
            self.stream.start_stream()
            print(f"🎤 Recording... (streaming after {self.start_delay_seconds}s)")
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            return

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream - runs in separate thread."""
        if status:
            print(f"Audio status: {status}")

        if not self.is_recording:
            return (None, pyaudio.paComplete)

        # Store all frames for final save
        self.all_frames.append(in_data)

        # Convert to numpy for processing
        audio_np = np.frombuffer(in_data, dtype=np.int16)

        # Add to circular buffer
        self.audio_buffer.extend(audio_np)
        self.samples_recorded += len(audio_np)

        # Check if we should process a chunk
        if self.samples_recorded >= self.start_delay_samples:
            current_time = time.time()
            time_since_last = current_time - self.last_process_time

            # Process at slide_seconds intervals
            if time_since_last >= self.slide_seconds:
                self.last_process_time = current_time

                # Only process if we have a full window
                if len(self.audio_buffer) >= self.window_samples:
                    # Copy current window
                    window_data = np.array(list(self.audio_buffer))
                    # Take the last window_samples
                    window_chunk = window_data[-self.window_samples :]

                    # Add to processing queue
                    self.processing_queue.put(window_chunk)

        return (in_data, pyaudio.paContinue)

    def get_next_chunk(self):
        """Get next audio chunk for processing (non-blocking)."""
        try:
            return self.processing_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_recording(self):
        """Stop recording and return audio data."""
        if not self.is_recording:
            return None

        self.is_recording = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if not self.all_frames:
            print("No audio recorded")
            return None

        # Return the raw audio frames
        return b''.join(self.all_frames)

    def save_chunk_to_file(self, chunk_np, chunk_id):
        """Save a chunk to a temporary WAV file for transcription."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/tmp/parakeet_chunk_{timestamp}_{chunk_id}.wav"

        try:
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(chunk_np.astype(np.int16).tobytes())

            return output_path
        except Exception as e:
            print(f"Error saving chunk: {e}")
            return None
