"""
Unit tests for audio_recorder module.

Tests marked with TODO require actual audio files to be provided.
"""
import os
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch
import pyaudio

from audio_recorder import (
    AudioConfig,
    AudioDeviceInfo,
    RecordingResult,
    detect_audio_device,
    detect_supported_rate,
    save_audio_to_wav,
    resample_audio,
    StandardRecorder,
)


class TestAudioConfig(unittest.TestCase):
    """Test AudioConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = AudioConfig()
        self.assertEqual(config.chunk_size, 2048)
        self.assertEqual(config.format, pyaudio.paInt16)
        self.assertEqual(config.channels, 1)
        self.assertEqual(config.model_rate, 16000)
        self.assertEqual(config.supported_rate, 16000)
        self.assertIsNone(config.device_index)
    
    def test_immutable(self):
        """Test that AudioConfig is immutable."""
        config = AudioConfig()
        with self.assertRaises(AttributeError):
            config.chunk_size = 4096
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = AudioConfig(
            chunk_size=4096,
            channels=2,
            supported_rate=48000,
            device_index=5
        )
        self.assertEqual(config.chunk_size, 4096)
        self.assertEqual(config.channels, 2)
        self.assertEqual(config.supported_rate, 48000)
        self.assertEqual(config.device_index, 5)


class TestAudioDeviceInfo(unittest.TestCase):
    """Test AudioDeviceInfo dataclass."""
    
    def test_creation(self):
        """Test creating device info."""
        info = AudioDeviceInfo(
            name="Test Device",
            index=0,
            max_input_channels=2,
            default_sample_rate=44100.0
        )
        self.assertEqual(info.name, "Test Device")
        self.assertEqual(info.index, 0)
        self.assertEqual(info.max_input_channels, 2)
        self.assertEqual(info.default_sample_rate, 44100.0)
    
    def test_immutable(self):
        """Test that AudioDeviceInfo is immutable."""
        info = AudioDeviceInfo("Test", 0, 2, 44100.0)
        with self.assertRaises(AttributeError):
            info.name = "Changed"


class TestRecordingResult(unittest.TestCase):
    """Test RecordingResult dataclass."""
    
    def test_creation(self):
        """Test creating recording result."""
        result = RecordingResult(
            audio_data=b"test data",
            sample_rate=16000,
            channels=1,
            duration_seconds=5.0,
            frame_count=100
        )
        self.assertEqual(result.audio_data, b"test data")
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.channels, 1)
        self.assertEqual(result.duration_seconds, 5.0)
        self.assertEqual(result.frame_count, 100)
    
    def test_immutable(self):
        """Test that RecordingResult is immutable."""
        result = RecordingResult(b"test", 16000, 1, 5.0, 100)
        with self.assertRaises(AttributeError):
            result.sample_rate = 48000


class TestDetectAudioDevice(unittest.TestCase):
    """Test detect_audio_device function."""
    
    def test_success(self):
        """Test successful device detection."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_audio.get_default_input_device_info.return_value = {
            'name': 'Test Microphone',
            'index': 0,
            'maxInputChannels': 2,
            'defaultSampleRate': 44100.0
        }
        
        info = detect_audio_device(mock_audio)
        
        self.assertEqual(info.name, 'Test Microphone')
        self.assertEqual(info.index, 0)
        self.assertEqual(info.max_input_channels, 2)
        self.assertEqual(info.default_sample_rate, 44100.0)
    
    def test_missing_fields(self):
        """Test device detection with missing fields."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_audio.get_default_input_device_info.return_value = {}
        
        info = detect_audio_device(mock_audio)
        
        self.assertEqual(info.name, 'Unknown')
        self.assertEqual(info.index, -1)
        self.assertEqual(info.max_input_channels, 0)
        self.assertEqual(info.default_sample_rate, 0.0)
    
    def test_no_device(self):
        """Test when no device is available."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_audio.get_default_input_device_info.side_effect = Exception("No device")
        
        with self.assertRaises(RuntimeError) as cm:
            detect_audio_device(mock_audio)
        
        self.assertIn("Could not detect default input device", str(cm.exception))


class TestDetectSupportedRate(unittest.TestCase):
    """Test detect_supported_rate function."""
    
    def test_first_rate_works(self):
        """Test when first rate in list works."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_stream = Mock()
        mock_audio.open.return_value = mock_stream
        
        config = AudioConfig()
        rates = [48000, 44100, 16000]
        
        rate = detect_supported_rate(mock_audio, config, rates)
        
        self.assertEqual(rate, 48000)
    
    def test_fallback_to_second_rate(self):
        """Test when first rate fails but second works."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_stream = Mock()
        
        # First call fails, second succeeds
        mock_audio.open.side_effect = [Exception("Not supported"), mock_stream]
        
        config = AudioConfig()
        rates = [48000, 44100]
        
        rate = detect_supported_rate(mock_audio, config, rates)
        
        self.assertEqual(rate, 44100)
    
    def test_no_supported_rate(self):
        """Test when no rate is supported."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_audio.open.side_effect = Exception("Not supported")
        
        config = AudioConfig()
        rates = [48000, 44100]
        
        with self.assertRaises(RuntimeError) as cm:
            detect_supported_rate(mock_audio, config, rates)
        
        self.assertIn("Could not find any supported sample rate", str(cm.exception))
    
    def test_default_rates(self):
        """Test that default rates list is used when none provided."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_stream = Mock()
        mock_audio.open.return_value = mock_stream
        
        config = AudioConfig()
        
        rate = detect_supported_rate(mock_audio, config)
        
        # Should use default list starting with 48000
        self.assertEqual(rate, 48000)


class TestSaveAudioToWav(unittest.TestCase):
    """Test save_audio_to_wav function."""
    
    def test_save_success(self):
        """Test successful save to WAV file."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_audio.get_sample_size.return_value = 2  # 16-bit = 2 bytes
        
        # Create simple audio data (silence)
        audio_data = b'\x00\x00' * 1000
        result = RecordingResult(
            audio_data=audio_data,
            sample_rate=16000,
            channels=1,
            duration_seconds=1.0,
            frame_count=10
        )
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            saved_path = save_audio_to_wav(result, temp_path, mock_audio)
            
            self.assertEqual(saved_path, temp_path)
            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_invalid_path(self):
        """Test save to invalid path."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_audio.get_sample_size.return_value = 2
        
        result = RecordingResult(b'test', 16000, 1, 1.0, 10)
        
        with self.assertRaises(IOError):
            save_audio_to_wav(result, '/invalid/path/file.wav', mock_audio)


class TestResampleAudio(unittest.TestCase):
    """Test resample_audio function."""
    
    @patch('subprocess.run')
    def test_resample_success(self, mock_run):
        """Test successful resampling."""
        mock_run.return_value = Mock(returncode=0)
        
        result = resample_audio('/input.wav', '/output.wav', 16000)
        
        self.assertEqual(result, '/output.wav')
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], 'ffmpeg')
        self.assertIn('16000', args)
    
    @patch('subprocess.run')
    def test_ffmpeg_fails(self, mock_run):
        """Test when ffmpeg fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'ffmpeg')
        
        with self.assertRaises(RuntimeError) as cm:
            resample_audio('/input.wav', '/output.wav', 16000)
        
        self.assertIn("ffmpeg failed", str(cm.exception))
    
    @patch('subprocess.run')
    def test_ffmpeg_not_found(self, mock_run):
        """Test when ffmpeg is not installed."""
        mock_run.side_effect = FileNotFoundError()
        
        with self.assertRaises(RuntimeError) as cm:
            resample_audio('/input.wav', '/output.wav', 16000)
        
        self.assertIn("ffmpeg not found", str(cm.exception))


class TestStandardRecorder(unittest.TestCase):
    """Test StandardRecorder class."""
    
    def test_initialization(self):
        """Test recorder initialization."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        config = AudioConfig()
        
        recorder = StandardRecorder(config, mock_audio)
        
        self.assertEqual(recorder.config, config)
        self.assertEqual(recorder.audio, mock_audio)
        self.assertFalse(recorder.is_recording())
    
    def test_start_recording(self):
        """Test starting recording."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_stream = Mock()
        mock_audio.open.return_value = mock_stream
        
        config = AudioConfig()
        recorder = StandardRecorder(config, mock_audio)
        
        recorder.start_recording()
        
        self.assertTrue(recorder.is_recording())
        mock_audio.open.assert_called_once()
    
    def test_start_when_already_recording(self):
        """Test starting when already recording raises error."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_stream = Mock()
        mock_audio.open.return_value = mock_stream
        
        config = AudioConfig()
        recorder = StandardRecorder(config, mock_audio)
        recorder.start_recording()
        
        with self.assertRaises(RuntimeError) as cm:
            recorder.start_recording()
        
        self.assertIn("Already recording", str(cm.exception))
    
    def test_read_chunk_when_not_recording(self):
        """Test reading chunk when not recording raises error."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        config = AudioConfig()
        recorder = StandardRecorder(config, mock_audio)
        
        with self.assertRaises(RuntimeError) as cm:
            recorder.read_chunk()
        
        self.assertIn("Not currently recording", str(cm.exception))
    
    def test_read_chunk_success(self):
        """Test reading chunk successfully."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_stream = Mock()
        mock_stream.read.return_value = b'audio data'
        mock_audio.open.return_value = mock_stream
        
        config = AudioConfig()
        recorder = StandardRecorder(config, mock_audio)
        recorder.start_recording()
        
        chunk = recorder.read_chunk()
        
        self.assertEqual(chunk, b'audio data')
        mock_stream.read.assert_called_once_with(config.chunk_size, exception_on_overflow=False)
    
    def test_stop_recording(self):
        """Test stopping recording."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        mock_stream = Mock()
        mock_stream.read.return_value = b'\x00\x00' * 2048
        mock_audio.open.return_value = mock_stream
        mock_audio.get_sample_size.return_value = 2
        
        config = AudioConfig(supported_rate=16000, chunk_size=2048)
        recorder = StandardRecorder(config, mock_audio)
        
        recorder.start_recording()
        recorder.read_chunk()
        recorder.read_chunk()
        result = recorder.stop_recording()
        
        self.assertIsInstance(result, RecordingResult)
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.channels, 1)
        self.assertEqual(result.frame_count, 2)
        self.assertFalse(recorder.is_recording())
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
    
    def test_stop_when_not_recording(self):
        """Test stopping when not recording raises error."""
        mock_audio = Mock(spec=pyaudio.PyAudio)
        config = AudioConfig()
        recorder = StandardRecorder(config, mock_audio)
        
        with self.assertRaises(RuntimeError) as cm:
            recorder.stop_recording()
        
        self.assertIn("Not currently recording", str(cm.exception))


class TestIntegrationWithRealAudio(unittest.TestCase):
    """
    Integration tests that require actual audio recording.
    
    TODO: Provide test audio files to complete these tests.
    """
    
    @unittest.skip("TODO: Requires actual audio hardware - fill in test implementation")
    def test_record_and_save_real_audio(self):
        """
        Test recording actual audio and saving to file.
        
        TODO: Implement this test with real PyAudio instance.
        Steps:
        1. Create real PyAudio instance
        2. Detect device and rate
        3. Create StandardRecorder
        4. Record for 2 seconds
        5. Save to WAV
        6. Verify file exists and has correct format
        """
        pass
    
    @unittest.skip("TODO: Requires test audio file - provide path to a test WAV file")
    def test_resample_real_audio(self):
        """
        Test resampling actual audio file.
        
        TODO: Provide path to test audio file (e.g., TEST_AUDIO_PATH)
        Steps:
        1. Use an existing audio file at TEST_AUDIO_PATH
        2. Resample to different rate
        3. Verify output file exists
        4. Verify output has correct sample rate
        """
        # TEST_AUDIO_PATH = "/path/to/test/audio.wav"  # TODO: Set this
        # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        #     output_path = f.name
        # 
        # try:
        #     result = resample_audio(TEST_AUDIO_PATH, output_path, 16000)
        #     self.assertTrue(os.path.exists(result))
        #     # TODO: Add verification of sample rate
        # finally:
        #     if os.path.exists(output_path):
        #         os.unlink(output_path)
        pass


if __name__ == '__main__':
    unittest.main()
