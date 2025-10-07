#!/usr/bin/env python3
"""
Entry point for Maivi - My AI Voice Input.
Launches the Qt GUI by default.
"""
import argparse
import sys


def main():
    """Main entry point for Maivi GUI."""
    parser = argparse.ArgumentParser(
        description="🎤 Maivi - My AI Voice Input\n"
                    "Real-time voice-to-text transcription with hotkey support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  maivi                                  # Start with defaults
  maivi --auto-paste                     # Auto-paste transcribed text
  maivi --window 10 --slide 5            # Custom chunk timing
  maivi --no-toggle                      # Hold mode (hold Alt+Q to record)
  maivi --keep-recordings 10             # Keep last 10 recordings
  maivi --keep-recordings 0              # Keep all recordings
  maivi --keep-recordings -1             # Delete immediately after transcription
  maivi --reprocess ~/.local/share/maivi/recordings/recording_20251005_123456.wav

Controls:
  Alt+Q     Start/stop recording (toggle mode)
  Esc       Exit application

How it works:
  Maivi records audio in overlapping chunks (default: 7s chunks, 3s intervals).
  The overlap ensures smooth merging without cutting words mid-syllable.
  Transcribed text is automatically copied to your clipboard.

Recording storage:
  - Linux: ~/.local/share/maivi/recordings/
  - macOS: ~/Library/Application Support/maivi/recordings/
  - Windows: %APPDATA%\maivi\recordings\
  Default: keeps last 3 recordings (use --keep-recordings to change).

For more info: https://github.com/MaximeRivest/maivi
        """
    )

    # Audio parameters
    parser.add_argument(
        "--window",
        type=float,
        default=7.0,
        metavar="SECONDS",
        help="Audio chunk window size in seconds (default: 7.0). "
             "Larger = better quality but slower processing."
    )
    parser.add_argument(
        "--slide",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="Slide interval in seconds (default: 3.0). "
             "Smaller = more overlap, higher CPU usage. "
             "Must be > window × 0.36 to avoid queue buildup."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Processing start delay in seconds (default: 2.0). "
             "Time to wait before starting transcription."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Speed adjustment factor (default: 1.0, experimental)."
    )

    # Behavior options
    parser.add_argument(
        "--auto-paste",
        action="store_true",
        help="Automatically paste transcribed text (default: copy to clipboard only)."
    )
    parser.add_argument(
        "--no-toggle",
        action="store_true",
        help="Use hold mode instead of toggle (hold Alt+Q to record)."
    )

    # Recording management
    parser.add_argument(
        "--keep-recordings",
        type=int,
        default=3,
        metavar="N",
        help="Keep last N recordings (default: 3). "
             "Use 0 to keep all, -1 to delete immediately after transcription."
    )
    parser.add_argument(
        "--reprocess",
        type=str,
        metavar="FILE",
        help="Reprocess an existing WAV file and exit."
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="Maivi v0.3.0"
    )

    args = parser.parse_args()

    import warnings

    # Suppress SyntaxWarnings from dependencies
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    # Handle --reprocess mode
    if args.reprocess:
        from pathlib import Path
        import nemo.collections.asr as nemo_asr
        import os

        wav_file = Path(args.reprocess)
        if not wav_file.exists():
            print(f"❌ Error: File not found: {wav_file}")
            sys.exit(1)

        print(f"\n🎤 Maivi - Reprocessing mode")
        print(f"   File: {wav_file}\n")
        print("📥 Loading AI model...")

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        model = model.cpu()
        model.eval()

        print("✓ Model loaded\n")
        print(f"🎯 Transcribing {wav_file.name}...")

        output = model.transcribe([str(wav_file)], timestamps=False)
        text = output[0].text.strip()

        if text:
            print(f"\n📋 Transcription:\n{text}\n")

            # Copy to clipboard
            import pyperclip
            pyperclip.copy(text)
            print("✓ Copied to clipboard!\n")
        else:
            print("⚠️  No speech detected in file\n")

        sys.exit(0)

    # Normal GUI mode
    # Show startup message immediately (before slow imports)
    print("\n🎤 Maivi - My AI Voice Input v0.3.0")
    print("   Starting up... (this may take ~30 seconds)\n")

    from maivi.gui.qt_gui import QtSTTServer

    server = QtSTTServer(
        auto_paste=args.auto_paste,
        window_seconds=args.window,
        slide_seconds=args.slide,
        start_delay_seconds=args.delay,
        speed=args.speed,
        toggle_mode=not args.no_toggle,
        keep_recordings=args.keep_recordings,
    )
    sys.exit(server.run())


if __name__ == "__main__":
    main()
