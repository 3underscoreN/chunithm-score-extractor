from pathlib import Path

from chunithm_score_extractor import ChunithmExtractor

extractor = ChunithmExtractor(device="gpu:0")
result = extractor.extract(Path("image.png").read_bytes())

print(f"Extracted Score: {result.score}")
print(f"Extracted Song Name: {result.song_name}")
print(f"Extracted Difficulty: {result.difficulty.value}")
