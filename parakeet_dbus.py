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
from gi.repository import GLib
from pydbus import SessionBus

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

    def StartRecording(self):
        if self.is_recording:
            print("Already recording")
            return

        print("Starting recording...")
        self.is_recording = True
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
            return

        # Save and transcribe
        try:
            # Save raw recording
            wf = wave.open(self.RAW_FILE, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.supported_rate)
            wf.writeframes(b''.join(self.audio_frames))
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
