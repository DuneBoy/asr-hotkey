"""
Cross-platform streaming UI for live transcription display.
Shows scrolling text in terminal as transcription happens.
"""
import threading
import sys
import time


class StreamingUITerminal:
    """
    Terminal-based streaming display (thread-safe).
    Shows scrolling text in the console with recording indicator.
    """

    def __init__(self, width_chars: int = 40):
        """
        Initialize terminal UI.

        Args:
            width_chars: Width of display in characters (default: 40)
        """
        self.width_chars = width_chars
        self.enabled = True
        self.last_text = ""
        self.lock = threading.Lock()
        self.started = False
        self.word_count = 0

        # ANSI color codes
        self.RED = '\033[91m'
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        self.RESET = '\033[0m'
        self.BOLD = '\033[1m'

    def start(self):
        """Start terminal display."""
        if self.started:
            return
        self.started = True

        # Print prominent header
        with self.lock:
            print(f"\n{self.BOLD}{self.GREEN}{'═' * (self.width_chars + 6)}{self.RESET}")
            print(f"{self.BOLD}{self.GREEN}║ {self.RED}●{self.RESET} {self.BOLD}{'RECORDING - LIVE TRANSCRIPTION'.center(self.width_chars)} {self.GREEN}║{self.RESET}")
            print(f"{self.BOLD}{self.GREEN}{'═' * (self.width_chars + 6)}{self.RESET}")
            print(f"{self.GREEN}║{self.RESET} {self.YELLOW}Listening...{self.RESET}".ljust(self.width_chars + 20) + f"{self.GREEN}║{self.RESET}")
            sys.stdout.flush()

    def update_text(self, text: str):
        """
        Update terminal display (thread-safe).

        Args:
            text: Full transcription text so far
        """
        if not self.started:
            return

        with self.lock:
            # Count words
            self.word_count = len(text.split())

            # Get last N characters for scrolling display
            display_text = text[-self.width_chars:] if len(text) > self.width_chars else text
            padded = display_text.ljust(self.width_chars)

            if padded != self.last_text:
                # Move cursor up one line and clear it, then print new text
                sys.stdout.write('\r\033[K')  # Clear current line
                print(f"{self.GREEN}║{self.RESET} {self.BOLD}{padded}{self.RESET} {self.GREEN}║{self.RESET} {self.BLUE}[{self.word_count} words]{self.RESET}", end='', flush=True)
                self.last_text = padded

    def stop(self):
        """Finish terminal display."""
        if not self.started:
            return

        with self.lock:
            print()  # New line
            print(f"{self.BOLD}{self.GREEN}{'═' * (self.width_chars + 6)}{self.RESET}")
            print(f"{self.GREEN}║{self.RESET} {self.BOLD}✓ Recording complete{self.RESET} - {self.BLUE}{self.word_count} words captured{self.RESET}".ljust(self.width_chars + 40) + f"{self.GREEN}║{self.RESET}")
            print(f"{self.BOLD}{self.GREEN}{'═' * (self.width_chars + 6)}{self.RESET}\n")
            sys.stdout.flush()
            self.started = False


def create_streaming_ui(width_chars: int = 40, prefer_gui: bool = False) -> StreamingUITerminal:
    """
    Factory function to create streaming UI.

    Args:
        width_chars: Width in characters (default: 40)
        prefer_gui: Ignored - always returns terminal UI (thread-safe)

    Returns:
        StreamingUITerminal instance (thread-safe terminal display)

    Note: Terminal display is thread-safe and works reliably across platforms.
    """
    return StreamingUITerminal(width_chars)
