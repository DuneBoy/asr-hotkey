"""
Audio recording module with functional, testable components.

This module provides pure functions and well-defined classes for audio recording,
device detection, and audio file manipulation.
"""
import contextlib
import os
import subprocess
import sys
import wave
from dataclasses import dataclass
from typing import List

import pyaudio


# Constants for temp files
RAW_RECORDING_PATH = "/tmp/parakeet_recording_raw.wav"
PROCESSED_RECORDING_PATH = "/tmp/parakeet_recording.wav"


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
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


@dataclass(frozen=True)
class AudioConfig:
    """Immutable configuration for audio recording."""
    chunk_size: int = 2048
    format: int = pyaudio.paInt16
    channels: int = 1
    model_rate: int = 16000
    supported_rate: int = 16000
    device_index: int | None = None


@dataclass(frozen=True)
class AudioDeviceInfo:
    """Information about an audio input device."""
    name: str
    index: int
    max_input_channels: int
    default_sample_rate: float


@dataclass(frozen=True)
class RecordingResult:
    """Result of a recording session with metadata."""
    audio_data: bytes
    sample_rate: int
    channels: int
    duration_seconds: float
    frame_count: int


def detect_audio_device(audio_instance: pyaudio.PyAudio) -> AudioDeviceInfo:
    """
    Detect and return information about the default input device.
    
    Args:
        audio_instance: PyAudio instance to query
        
    Returns:
        AudioDeviceInfo with device details
        
    Raises:
        RuntimeError: If no default input device is found
    """
    try:
        device_info = audio_instance.get_default_input_device_info()
        return AudioDeviceInfo(
            name=device_info.get('name', 'Unknown'),
            index=device_info.get('index', -1),
            max_input_channels=device_info.get('maxInputChannels', 0),
            default_sample_rate=device_info.get('defaultSampleRate', 0.0)
        )
    except Exception as e:
        raise RuntimeError(f"Could not detect default input device: {e}")


def detect_supported_rate(
    audio_instance: pyaudio.PyAudio,
    config: AudioConfig,
    rates_to_test: List[int] | None = None
) -> int:
    """
    Detect the first supported sample rate from a list of candidates.
    
    Args:
        audio_instance: PyAudio instance to test with
        config: Audio configuration to use for testing
        rates_to_test: List of sample rates to test (defaults to common rates)
        
    Returns:
        First supported sample rate from the list
        
    Raises:
        RuntimeError: If no supported rate is found
    """
    if rates_to_test is None:
        rates_to_test = [48000, 44100, 32000, 22050, 16000, 8000]
    
    for test_rate in rates_to_test:
        try:
            with suppress_stderr():
                stream = audio_instance.open(
                    format=config.format,
                    channels=config.channels,
                    rate=test_rate,
                    input=True,
                    frames_per_buffer=config.chunk_size,
                    input_device_index=config.device_index,
                )
                stream.close()
            return test_rate
        except Exception:
            continue
    
    raise RuntimeError("Could not find any supported sample rate")


def save_audio_to_wav(
    recording: RecordingResult,
    output_path: str,
    audio_instance: pyaudio.PyAudio
) -> str:
    """
    Save audio data to a WAV file.
    
    Args:
        recording: RecordingResult containing audio data and metadata
        output_path: Path where WAV file should be saved
        audio_instance: PyAudio instance for getting sample size
        
    Returns:
        Path to the saved file (same as output_path)
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(recording.channels)
            wf.setsampwidth(audio_instance.get_sample_size(pyaudio.paInt16))
            wf.setframerate(recording.sample_rate)
            wf.writeframes(recording.audio_data)
        return output_path
    except Exception as e:
        raise IOError(f"Failed to save audio to {output_path}: {e}")


def resample_audio(
    input_path: str,
    output_path: str,
    target_rate: int
) -> str:
    """
    Resample an audio file to a target sample rate using ffmpeg.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output resampled file
        target_rate: Target sample rate in Hz
        
    Returns:
        Path to the resampled file (same as output_path)
        
    Raises:
        RuntimeError: If ffmpeg fails or is not available
    """
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-ar', str(target_rate), output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to resample audio: {e}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found - please install ffmpeg")


class StandardRecorder:
    """
    Encapsulates standard blocking audio recording with clear lifecycle.
    
    Usage:
        recorder = StandardRecorder(config, audio_instance)
        recorder.start_recording()
        while recording:
            recorder.read_chunk()
        result = recorder.stop_recording()
    """
    
    def __init__(self, config: AudioConfig, audio_instance: pyaudio.PyAudio):
        """
        Initialize the recorder.
        
        Args:
            config: Audio configuration to use
            audio_instance: PyAudio instance for stream management
        """
        self.config = config
        self.audio = audio_instance
        self._stream = None
        self._frames: List[bytes] = []
        self._is_recording = False
    
    def start_recording(self) -> None:
        """
        Start recording audio.
        
        Raises:
            RuntimeError: If already recording or stream cannot be opened
        """
        if self._is_recording:
            raise RuntimeError("Already recording")
        
        try:
            self._stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.supported_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
                input_device_index=self.config.device_index,
            )
            self._frames = []
            self._is_recording = True
        except Exception as e:
            raise RuntimeError(f"Failed to start recording: {e}")
    
    def read_chunk(self) -> bytes:
        """
        Read one chunk of audio data and append to internal buffer.
        
        Returns:
            The chunk of audio data read
            
        Raises:
            RuntimeError: If not currently recording
        """
        if not self._is_recording or self._stream is None:
            raise RuntimeError("Not currently recording")
        
        data = self._stream.read(self.config.chunk_size, exception_on_overflow=False)
        self._frames.append(data)
        return data
    
    def stop_recording(self) -> RecordingResult:
        """
        Stop recording and return the result.
        
        Returns:
            RecordingResult containing all recorded audio and metadata
            
        Raises:
            RuntimeError: If not currently recording
        """
        if not self._is_recording:
            raise RuntimeError("Not currently recording")
        
        self._is_recording = False
        
        # Close stream
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                print(f"Warning: error closing stream: {e}")
            finally:
                self._stream = None
        
        # Combine all frames
        audio_data = b"".join(self._frames)
        frame_count = len(self._frames)
        
        # Calculate duration
        bytes_per_sample = self.audio.get_sample_size(self.config.format)
        samples_per_frame = self.config.chunk_size
        total_samples = frame_count * samples_per_frame
        duration = total_samples / self.config.supported_rate
        
        return RecordingResult(
            audio_data=audio_data,
            sample_rate=self.config.supported_rate,
            channels=self.config.channels,
            duration_seconds=duration,
            frame_count=frame_count
        )
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
