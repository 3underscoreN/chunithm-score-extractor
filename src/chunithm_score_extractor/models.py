from dataclasses import dataclass
from enum import StrEnum


class Difficulty(StrEnum):
    """
    Represents the difficulty level of a song.

    Attributes
    ----------
    BAS : str
        Basic difficulty.
    ADV : str
        Advanced difficulty.
    EXP : str
        Expert difficulty.
    MAS : str
        Master difficulty.
    ULT : str
        Ultimate difficulty.
    WORLDEND : str
        World's End difficulty.
    """
    BAS = "BAS"
    ADV = "ADV"
    EXP = "EXP"
    MAS = "MAS"
    ULT = "ULT"
    WORLDEND = "WORLDEND"


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """
    Represents the result of extracting a song score from an image.

    Attributes
    ----------
    score : int
        The extracted score for the song.
    song_name : str
        The name of the song.
    difficulty : Difficulty
        The difficulty level of the song.
    """
    score: int
    song_name: str
    difficulty: Difficulty
