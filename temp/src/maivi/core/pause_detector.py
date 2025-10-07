"""
Pause detection for creating paragraph breaks in transcription.
"""
import numpy as np


class PauseDetector:
    """Detect pauses/silence in audio for paragraph formatting."""

    def __init__(self,
                 silence_threshold_db=-40.0,
                 min_pause_duration=1.0,
                 sample_rate=16000):
        """
        Args:
            silence_threshold_db: Audio level below this is considered silence (dB)
            min_pause_duration: Minimum pause length to trigger paragraph break (seconds)
            sample_rate: Audio sample rate
        """
        self.silence_threshold_db = silence_threshold_db
        self.min_pause_duration = min_pause_duration
        self.sample_rate = sample_rate
        self.min_pause_samples = int(min_pause_duration * sample_rate)

    def _calculate_db(self, audio_chunk):
        """Calculate RMS amplitude in dB."""
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        if rms == 0:
            return -100.0  # Very quiet
        return 20 * np.log10(rms)

    def detect_pause(self, audio_chunk):
        """
        Check if audio chunk contains a significant pause.

        Returns:
            bool: True if pause detected, False otherwise
        """
        if len(audio_chunk) < self.min_pause_samples:
            return False

        # Calculate dB level
        db_level = self._calculate_db(audio_chunk)

        # Check if below silence threshold
        if db_level < self.silence_threshold_db:
            return True

        return False

    def find_pauses_in_window(self, audio_data, chunk_size_seconds=0.1):
        """
        Find all pauses in an audio window.

        Args:
            audio_data: numpy array of audio samples
            chunk_size_seconds: Size of chunks to analyze for silence

        Returns:
            list: List of (start_time, duration) tuples for each pause
        """
        chunk_samples = int(chunk_size_seconds * self.sample_rate)
        pauses = []

        i = 0
        pause_start = None

        while i < len(audio_data):
            chunk = audio_data[i:i + chunk_samples]
            if len(chunk) < chunk_samples:
                break

            db_level = self._calculate_db(chunk)

            if db_level < self.silence_threshold_db:
                # Start of pause
                if pause_start is None:
                    pause_start = i / self.sample_rate
            else:
                # End of pause
                if pause_start is not None:
                    pause_end = i / self.sample_rate
                    pause_duration = pause_end - pause_start

                    if pause_duration >= self.min_pause_duration:
                        pauses.append((pause_start, pause_duration))

                    pause_start = None

            i += chunk_samples

        # Handle pause at end
        if pause_start is not None:
            pause_end = len(audio_data) / self.sample_rate
            pause_duration = pause_end - pause_start

            if pause_duration >= self.min_pause_duration:
                pauses.append((pause_start, pause_duration))

        return pauses


class ParagraphFormatter:
    """Format transcription with paragraph breaks based on pauses."""

    def __init__(self, pause_detector):
        self.pause_detector = pause_detector
        self.pending_text = []
        self.last_chunk_had_pause = False

    def add_chunk(self, text, audio_chunk=None, had_pause=False):
        """
        Add transcription chunk with optional pause detection.

        Args:
            text: Transcribed text
            audio_chunk: Audio data (optional, for pause detection)
            had_pause: Whether this chunk had a pause before it

        Returns:
            str: Formatted text with paragraph breaks
        """
        if not text or not text.strip():
            return self.get_result()

        # Detect pause from audio if provided
        if audio_chunk is not None and self.pause_detector.detect_pause(audio_chunk):
            had_pause = True

        # Add paragraph break if there was a pause
        if had_pause and self.pending_text:
            # Add newline to separate paragraphs
            self.pending_text.append("\n")

        # Add the new text
        self.pending_text.append(text.strip())
        self.last_chunk_had_pause = had_pause

        return self.get_result()

    def get_result(self):
        """Get the formatted result with paragraph breaks."""
        return " ".join(self.pending_text)

    def reset(self):
        """Reset the formatter."""
        self.pending_text = []
        self.last_chunk_had_pause = False
