# Audio Recording Refactoring Plan

## ✅ STATUS: COMPLETED

All audio recording functionality has been successfully extracted into a dedicated, testable module!

## Goal
Extract all audio recording functionality into a dedicated, testable module that separates concerns and makes the code more functional and deterministic.

## Current State Analysis

### Files Involved
1. **parakeet_dbus.py** - Contains:
   - `TranscriptionSession._run_standard()` - Standard blocking recording
   - PyAudio initialization and device detection in `ParakeetService.__init__`
   - Audio format constants (CHUNK, FORMAT, CHANNELS, etc.)
   
2. **streaming_recorder.py** - Contains:
   - `StreamingRecorder` class - Streaming audio with sliding window
   
## Proposed Changes

### 1. Create New File: `audio_recorder.py`

This will be the core audio recording module with pure, testable functions.

#### Classes/Functions to Create:

##### A. `AudioConfig` (dataclass)
**Purpose**: Immutable configuration for audio recording
```python
@dataclass(frozen=True)
class AudioConfig:
    chunk_size: int = 2048
    format: int = pyaudio.paInt16
    channels: int = 1
    model_rate: int = 16000
    supported_rate: int = 16000
    device_index: int | None = None
```

##### B. `AudioDeviceInfo` (dataclass)
**Purpose**: Information about audio device
```python
@dataclass(frozen=True)
class AudioDeviceInfo:
    name: str
    index: int
    max_input_channels: int
    default_sample_rate: float
```

##### C. `RecordingResult` (dataclass)
**Purpose**: Result of a recording session
```python
@dataclass(frozen=True)
class RecordingResult:
    audio_data: bytes
    sample_rate: int
    channels: int
    duration_seconds: float
    frame_count: int
```

##### D. `detect_audio_device()` -> AudioDeviceInfo
**Purpose**: Pure function to detect and return default input device info
- Takes PyAudio instance
- Returns device information
- No side effects (logging done by caller)

##### E. `detect_supported_rate()` -> int
**Purpose**: Pure function to find supported sample rate
- Takes PyAudio instance, config, list of rates to test
- Returns first working rate
- Raises exception if none found

##### F. `StandardRecorder` class
**Purpose**: Encapsulate standard (non-streaming) recording logic
- Constructor takes AudioConfig and PyAudio instance
- `start_recording()` - Opens stream
- `read_chunk()` - Reads one chunk of data
- `stop_recording()` -> RecordingResult - Closes stream and returns result
- Stateful but with clear lifecycle

##### G. `save_audio_to_wav()` -> str
**Purpose**: Pure function to save audio data to WAV file
- Takes RecordingResult, output path, PyAudio instance
- Returns file path
- No mutations

##### H. `resample_audio()` -> str
**Purpose**: Pure function to resample audio file
- Takes input path, output path, target rate
- Uses ffmpeg subprocess
- Returns output path

### 2. Update `streaming_recorder.py`

#### Changes:
- Accept `AudioConfig` instead of individual parameters
- Return `RecordingResult` instead of raw bytes
- Make `save_chunk_to_file()` accept explicit parameters (don't use self.audio)
- Remove internal PyAudio reference where possible

### 3. Update `parakeet_dbus.py`

#### Changes:
- Import from `audio_recorder` module
- Replace inline audio logic with calls to new functions/classes
- `ParakeetService.__init__()`:
  - Use `detect_audio_device()` for device detection
  - Use `detect_supported_rate()` for rate detection
  - Store `AudioConfig` instance
- `TranscriptionSession._run_standard()`:
  - Use `StandardRecorder` class
  - Use `save_audio_to_wav()` function
  - Use `resample_audio()` function
- `TranscriptionSession._run_streaming()`:
  - Pass `AudioConfig` to StreamingRecorder

### 4. Benefits of This Refactoring

1. **Testability**: Pure functions can be tested without audio hardware
2. **Separation of Concerns**: Audio logic separated from D-Bus service logic
3. **Reusability**: Recording logic can be used in other contexts
4. **Determinism**: Immutable configs and clear data flow
5. **Easier Mocking**: Can inject test audio data without PyAudio
6. **Clear Contracts**: Dataclasses make inputs/outputs explicit

### 5. Testing Strategy (Future)

With this refactoring, we can:
- Mock PyAudio for unit tests
- Test audio processing with synthetic data
- Test recording lifecycle without actual recording
- Verify resampling logic with known inputs

## Execution Order

1. Create `audio_recorder.py` with all dataclasses and functions
2. Update `streaming_recorder.py` to use new structures
3. Update `parakeet_dbus.py` to use new module
4. Test with actual recording to verify functionality
5. Clean up any dead code

## Questions for Review

1. Should we also extract the transcription model loading into a separate module?
2. Do you want dependency injection for PyAudio instance or keep it as-is?
3. Should we add type hints throughout for better IDE support?
4. Any specific test scenarios you want to prioritize?
5. Should we keep temp file paths configurable or hardcoded?

## Notes

- All changes maintain backward compatibility with existing behavior
- No changes to D-Bus interface
- No changes to LLM processing logic
- Focus is purely on audio recording abstraction

---

## ✅ IMPLEMENTATION COMPLETE

### What Was Delivered

#### 1. **audio_recorder.py** (295 lines)
- ✅ 3 immutable dataclasses: `AudioConfig`, `AudioDeviceInfo`, `RecordingResult`
- ✅ 4 pure functions: `detect_audio_device()`, `detect_supported_rate()`, `save_audio_to_wav()`, `resample_audio()`
- ✅ 1 stateful class: `StandardRecorder` with clean lifecycle
- ✅ Constants for temp file paths
- ✅ Full docstrings and type hints

#### 2. **test_audio_recorder.py** (397 lines)
- ✅ 28 unit tests total
- ✅ 26 tests passing with mocks
- ✅ 2 integration tests skipped (ready for you to add audio files)
- ✅ Test coverage for all functions and edge cases
- ✅ Clear TODO comments for audio-dependent tests

#### 3. **streaming_recorder.py** (Updated)
- ✅ Now accepts `AudioConfig` instead of individual parameters
- ✅ Returns `RecordingResult` instead of raw bytes
- ✅ Uses constants for temp file paths
- ✅ Better type hints and documentation

#### 4. **parakeet_dbus.py** (Updated)
- ✅ Removed `suppress_stderr()` (now imported from audio_recorder)
- ✅ Removed inline audio recording logic from `_run_standard()`
- ✅ Uses `StandardRecorder` class for recording
- ✅ Uses pure functions for device/rate detection, saving, and resampling
- ✅ Stores `AudioConfig` instance instead of scattered constants
- ✅ Much cleaner and more maintainable

### Test Results

```
Ran 28 tests in 0.004s
OK (skipped=2)
```

All tests pass! The 2 skipped tests are ready for real audio:
1. `test_record_and_save_real_audio` - Needs real PyAudio hardware
2. `test_resample_real_audio` - Needs test WAV file path

### Benefits Achieved

✅ **Testability**: All audio logic can be tested with mocks  
✅ **Separation of Concerns**: Audio completely isolated from D-Bus service  
✅ **Reusability**: Audio module can be used in other projects  
✅ **Determinism**: Immutable configs, pure functions, clear data flow  
✅ **Maintainability**: Much easier to understand and modify  
✅ **Type Safety**: Full type hints throughout

### Next Steps (Future)

When you're ready to refactor transcription/AI logic:
1. Create a new `REFACTORING_PLAN_TRANSCRIPTION.md`
2. Extract model loading and LLM processing
3. Create similar testable structures

The audio refactoring is **complete and ready to use**! 🎉
