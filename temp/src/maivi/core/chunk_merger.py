"""
Simple overlap-based chunk merger - much more reliable than word trimming.

Strategy:
- Use fixed-size chunks (7s) with large overlap (4s)
- Merge by finding overlap text in the existing result
- No word trimming, no Levenshtein distance, no complex fallbacks
- Just simple string matching on the overlap region
"""


class SimpleChunkMerger:
    """Merge overlapping transcription chunks by matching overlap regions."""

    def __init__(self):
        self.result = ""  # Final merged text
        self.last_chunk_text = ""  # Previous chunk for overlap detection

    def reset(self):
        """Reset the merger."""
        self.result = ""
        self.last_chunk_text = ""

    def add_chunk(self, text: str, is_final: bool = False) -> str:
        """
        Add a new chunk and merge with existing result.

        Args:
            text: Transcribed text from this chunk
            is_final: True if this is the last chunk

        Returns:
            The merged result so far
        """
        if not text or not text.strip():
            return self.result

        text = text.strip()

        # First chunk - just use it
        if not self.result:
            self.result = text
            self.last_chunk_text = text
            return self.result

        # Find overlap by looking for the beginning of new chunk in previous chunk
        # Since chunks overlap by ~4s, we expect the first ~half of new chunk
        # to match the last ~half of previous chunk

        overlap_found = False
        new_text = text

        # Try to find where the new chunk starts in our existing result
        # Split into words for more reliable matching
        new_words = text.split()
        result_words = self.result.split()

        # Try matching with increasingly smaller overlap sizes
        # Start with first 60% of new chunk (should be in overlap region)
        for overlap_fraction in [0.6, 0.5, 0.4, 0.3, 0.2]:
            overlap_word_count = max(3, int(len(new_words) * overlap_fraction))
            overlap_words = new_words[:overlap_word_count]
            overlap_text = " ".join(overlap_words).lower()

            # Look for this overlap in the existing result
            result_lower = self.result.lower()

            # Try to find exact match
            if overlap_text in result_lower:
                # Found it! Find where it ends
                overlap_end_pos = result_lower.find(overlap_text) + len(overlap_text)

                # Everything after the overlap in new chunk is genuinely new
                new_part = " ".join(new_words[overlap_word_count:])

                if new_part:
                    self.result = self.result[:overlap_end_pos].strip() + " " + new_part

                overlap_found = True
                break

        # If no overlap found, try fuzzy word-by-word matching
        if not overlap_found:
            # Find the longest sequence of words from start of new chunk
            # that appear somewhere in the result
            best_match_len = 0
            best_match_pos = 0

            for start_len in range(min(10, len(new_words)), 2, -1):
                search_words = new_words[:start_len]
                search_text = " ".join(search_words).lower()

                result_lower = self.result.lower()
                if search_text in result_lower:
                    best_match_len = start_len
                    best_match_pos = result_lower.find(search_text) + len(search_text)
                    break

            if best_match_len > 0:
                # Found overlap
                new_part = " ".join(new_words[best_match_len:])
                if new_part:
                    self.result = self.result[:best_match_pos].strip() + " " + new_part
                overlap_found = True

        # If still no overlap (shouldn't happen with 4s overlap), just append with separator
        if not overlap_found:
            print(f"⚠️  Warning: No overlap found, appending with gap marker")
            self.result = self.result.strip() + " ... " + text

        self.last_chunk_text = text
        return self.result

    def get_result(self) -> str:
        """Get the final merged result."""
        return self.result.strip()
