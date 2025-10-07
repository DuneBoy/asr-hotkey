#!/usr/bin/env python3
"""
Simple test script for streaming components.
Tests the chunk merger with sample transcriptions.
"""

from chunk_merger import ChunkMerger


def test_chunk_merger():
    """Test the chunk merger with overlapping text samples."""
    print("=" * 60)
    print("Testing ChunkMerger")
    print("=" * 60)

    merger = ChunkMerger()

    # Simulate overlapping transcription chunks
    # These simulate 7s chunks with 4s overlap
    chunks = [
        "Hello this is a test of the streaming transcription system",
        "test of the streaming transcription system it should work well with overlapping chunks",
        "it should work well with overlapping chunks that contain duplicate words",
        "that contain duplicate words at the beginning and end of each segment"
    ]

    print("\nProcessing chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        is_final = (i == len(chunks))
        result = merger.add_chunk(chunk, is_final=is_final)

        print(f"Chunk {i}: {chunk}")
        print(f"Merged:  {result}")
        print()

    print("=" * 60)
    print("Final Result:")
    print("=" * 60)
    print(merger.get_result())
    print()

    # Test reset
    merger.reset()
    print("✓ Merger reset successful")

    # Test with no overlap (should add separator)
    print("\n" + "=" * 60)
    print("Testing chunks with no overlap")
    print("=" * 60)

    merger.reset()
    result1 = merger.add_chunk("First sentence here")
    print(f"Chunk 1: {result1}")

    result2 = merger.add_chunk("Completely different text")
    print(f"Chunk 2: {result2}")
    print()

    # Test with empty chunks
    print("=" * 60)
    print("Testing edge cases")
    print("=" * 60)

    merger.reset()
    result = merger.add_chunk("")
    print(f"Empty chunk: '{result}'")

    result = merger.add_chunk("   ")
    print(f"Whitespace chunk: '{result}'")

    result = merger.add_chunk("Now some actual text")
    print(f"After empty: '{result}'")
    print()

    print("✓ All tests completed!")


if __name__ == "__main__":
    test_chunk_merger()
