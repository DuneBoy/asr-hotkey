# Parakeet Streaming Mode

## Overview

Streaming mode enables real-time transcription using a sliding window approach. Audio is processed in overlapping chunks during recording, reducing latency and enabling infinite recording length.

## How It Works

### Sliding Window Architecture

1. **Recording Phase**
   - Audio is continuously captured at 16kHz
   - A circular buffer maintains the last 7 seconds of audio (configurable)
   - After an initial delay (default 6s), chunks are extracted every 3 seconds (configurable)

2. **Processing Phase**
   - Each 7-second chunk is transcribed in parallel during recording
   - Chunks have 4 seconds of overlap (window - slide = 7s - 3s)
   - A separate thread handles transcription while recording continues

3. **Merging Phase**
   - ChunkMerger intelligently combines overlapping transcriptions
   - Detects duplicate text in overlap regions
   - Removes redundancy and appends only new content
   - Uses word-based matching for reliability

### Benefits

- **Lower Latency**: Most transcription happens during recording
- **Infinite Length**: No memory limit, process as you record
- **Better Quality**: Larger context windows (7s) vs real-time streaming
- **Graceful Fallback**: Short recordings (< 6s) use standard single-pass mode

## Usage

### Start with Streaming Mode

```bash
# Basic streaming mode
uv run python parakeet_dbus.py --stream

# Custom parameters
uv run python parakeet_dbus.py --stream --window 10 --slide 5 --delay 8
```

### Start with Standard Mode (Default)

```bash
uv run python parakeet_dbus.py
```

### Command-Line Options

- `--stream`: Enable streaming mode
- `--window SECONDS`: Chunk window size (default: 7.0)
  - Larger = better context, slower processing
  - Must be larger than slide interval
- `--slide SECONDS`: Slide interval (default: 3.0)
  - Smaller = more overlap, smoother merging
  - Overlap = window - slide
- `--delay SECONDS`: Delay before streaming starts (default: 6.0)
  - Should be ≥ window size to build up buffer

### Recommended Settings

**Balanced (default):**
```bash
--stream --window 7 --slide 3 --delay 6
```
- 4 seconds overlap
- Good balance of quality and speed

**High Quality (slow):**
```bash
--stream --window 10 --slide 5 --delay 10
```
- 5 seconds overlap
- Better context, more accurate
- Requires faster CPU

**Fast (lower quality):**
```bash
--stream --window 5 --slide 2 --delay 5
```
- 3 seconds overlap
- Faster processing
- May miss some words

## Files Added

1. **streaming_recorder.py**
   - Handles audio capture with sliding window
   - Manages circular buffer and chunk queue
   - Adapted from maivi project for Parakeet

2. **chunk_merger.py**
   - Merges overlapping transcription chunks
   - Uses word-based overlap detection
   - No external dependencies (pure Python)

## D-Bus Interface

The D-Bus interface remains unchanged:
- `com.parakeet.Transcribe.StartRecording()`
- `com.parakeet.Transcribe.StopRecording()`
- `com.parakeet.Transcribe.Toggle()`

The mode (standard vs streaming) is set when starting the service.

## System Tray Icons

- **Gray**: Idle
- **Red**: Recording (in both modes)
- **Blue**: Transcribing/processing

In streaming mode, the blue state is typically very brief since most transcription happens during recording.

## Troubleshooting

### "No overlap found" warnings
- Increase overlap by reducing `--slide` value
- Increase `--window` for more context

### High CPU usage
- Reduce `--window` size
- Increase `--slide` interval (less overlap)
- Use standard mode for short recordings

### Choppy transcription
- Increase `--window` for better context
- Decrease `--slide` for more overlap
- Check CPU isn't maxed out during recording

## Technical Details

### Memory Usage
- Circular buffer: ~window_seconds × 32KB (for 16kHz mono)
- Chunk queue: ~3-5 chunks pending (each ~window_seconds × 32KB)
- Total extra memory: ~1-2 MB for default settings

### CPU Requirements
- Must transcribe chunks faster than they're generated
- Rule of thumb: transcription should take < slide_seconds
- Test with: record for 30s and observe lag

### Dependencies
- `numpy`: For audio buffer manipulation
- `pyaudio`: Audio capture (already required)
- `onnx-asr`: Transcription model (already required)
- No additional dependencies needed

## Comparison: Standard vs Streaming

| Feature | Standard Mode | Streaming Mode |
|---------|--------------|----------------|
| Recording length | Limited by memory | Infinite |
| Latency after stop | High (full transcribe) | Low (mostly done) |
| CPU during record | Low | High |
| Memory usage | Full audio in RAM | Sliding window only |
| Best for | Short recordings | Long recordings |
| Complexity | Simple | Complex |

## Future Enhancements

Possible additions (not yet implemented):
- [ ] Hybrid mode: auto-switch based on recording length
- [ ] Progressive clipboard updates during recording
- [ ] Pause detection for paragraph breaks
- [ ] Real-time UI display
- [ ] Configurable tray icon states for streaming
