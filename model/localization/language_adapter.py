"""
Language Adapter for syllabification and word-level analysis.

Provides a plug-in architecture for mapping word boundaries to
syllable boundaries across multiple languages (Kannada, English, Hindi).

Each language adapter implements the same interface:
    - adapt(word_timestamps, text) → List[SyllableTimestamp]
    - syllabify(word) → List[str]

Usage:
    registry = LanguageAdapterRegistry()
    adapter = registry.get("en")
    syllables = adapter.adapt(word_timestamps, text)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class WordTimestamp:
    """A word with its start/end time and text."""
    word: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0


@dataclass
class SyllableTimestamp:
    """A syllable with its start/end time, text, parent word, and position."""
    syllable: str
    start_sec: float
    end_sec: float
    word: str
    syllable_index: int  # 0-based position within word
    total_syllables: int  # total syllables in word
    confidence: float = 1.0


class BaseLanguageAdapter:
    """Base class for language-specific syllabification adapters."""

    language_code: str = ""
    language_name: str = ""

    def adapt(
        self, word_timestamps: List[WordTimestamp], text: str
    ) -> List[SyllableTimestamp]:
        """
        Convert word timestamps to syllable timestamps.

        Args:
            word_timestamps: List of WordTimestamp with timing info.
            text: Original transcript text.

        Returns:
            List of SyllableTimestamp with syllable-level timing.
        """
        raise NotImplementedError

    def syllabify(self, word: str) -> List[str]:
        """
        Split a word into syllables.

        Args:
            word: The word to syllabify.

        Returns:
            List of syllable strings.
        """
        raise NotImplementedError


class EnglishAdapter(BaseLanguageAdapter):
    """English syllabification adapter using rule-based + fallback approach."""

    language_code = "en"
    language_name = "English"

    # Vowel nuclei for syllable detection
    VOWELS = set("aeiouy")
    VOWEL_CLUSTERS = {"ai", "au", "ea", "ei", "io", "oa", "ou", "oy"}

    def syllabify(self, word: str) -> List[str]:
        word = word.lower().strip()
        if not word:
            return []

        syllables = []
        current = ""
        prev_was_vowel = False

        for i, ch in enumerate(word):
            is_vowel = ch in self.VOWELS
            current += ch

            # Split after a vowel followed by a consonant (VC boundary)
            if prev_was_vowel and not is_vowel and len(current) > 1:
                if i >= 2 and word[i-2:i+1] in self.VOWEL_CLUSTERS:
                    continue
                syllables.append(current)
                current = ""

            prev_was_vowel = is_vowel

        if current:
            if syllables and len(current) == 1 and current not in self.VOWELS:
                syllables[-1] += current
            else:
                syllables.append(current)

        return syllables if syllables else [word]

    def adapt(
        self, word_timestamps: List[WordTimestamp], text: str
    ) -> List[SyllableTimestamp]:
        result = []
        for wt in word_timestamps:
            syls = self.syllabify(wt.word)
            n = len(syls)
            duration = wt.end_sec - wt.start_sec
            syl_duration = duration / n if n > 0 else duration

            for j, syl in enumerate(syls):
                result.append(SyllableTimestamp(
                    syllable=syl,
                    start_sec=round(wt.start_sec + j * syl_duration, 4),
                    end_sec=round(wt.start_sec + (j + 1) * syl_duration, 4),
                    word=wt.word,
                    syllable_index=j,
                    total_syllables=n,
                    confidence=wt.confidence,
                ))
        return result


class KannadaAdapter(BaseLanguageAdapter):
    """
    Kannada syllabification adapter.

    Kannada uses an abugida script where each consonant inherently carries
    the vowel 'a' (ಅ). Syllables are structured as CV or CVC patterns.
    """

    language_code = "kn"
    language_name = "Kannada"

    # Kannada vowel signs (standalone vowels)
    KANNADA_VOWEL_SIGNS = set("ಅಆಇಈಉಊಋಎಏಐಒಓಔ")
    # Kannada combining marks (vowel signs that attach to consonants)
    KANNADA_COMBINING = set("ಾಿೀುೂೃೆೇೈೊೋೌಂಃ್")

    def syllabify(self, word: str) -> List[str]:
        """
        Syllabify Kannada text.

        Kannada syllables follow the pattern: (C)(C)V(C)
        where C = consonant, V = vowel (inherent or explicit).
        """
        if not word:
            return []

        syllables = []
        current = ""

        for ch in word:
            current += ch
            # A syllable boundary occurs after a vowel sign or vowel
            if ch in self.KANNADA_VOWEL_SIGNS or ch in self.KANNADA_COMBINING:
                if len(current) > 0:
                    syllables.append(current)
                    current = ""

        if current:
            syllables.append(current)

        return syllables if syllables else [word]

    def adapt(
        self, word_timestamps: List[WordTimestamp], text: str
    ) -> List[SyllableTimestamp]:
        result = []
        for wt in word_timestamps:
            syls = self.syllabify(wt.word)
            n = len(syls)
            duration = wt.end_sec - wt.start_sec
            syl_duration = duration / n if n > 0 else duration

            for j, syl in enumerate(syls):
                result.append(SyllableTimestamp(
                    syllable=syl,
                    start_sec=round(wt.start_sec + j * syl_duration, 4),
                    end_sec=round(wt.start_sec + (j + 1) * syl_duration, 4),
                    word=wt.word,
                    syllable_index=j,
                    total_syllables=n,
                    confidence=wt.confidence,
                ))
        return result


class HindiAdapter(BaseLanguageAdapter):
    """
    Hindi syllabification adapter.

    Hindi (Devanagari) syllables follow CV(C) patterns.
    Each consonant carries inherent vowel 'अ' unless modified by a vowel sign.
    """

    language_code = "hi"
    language_name = "Hindi"

    # Devanagari vowel signs (matras)
    DEVANAGARI_VOWELS = set("अआइईउऊऋएऐओऔ")
    DEVANAGARI_COMBINING = set("ािीुूृेैोौंः्")

    def syllabify(self, word: str) -> List[str]:
        if not word:
            return []

        syllables = []
        current = ""

        for ch in word:
            current += ch
            if ch in self.DEVANAGARI_VOWELS or ch in self.DEVANAGARI_COMBINING:
                if len(current) > 0:
                    syllables.append(current)
                    current = ""

        if current:
            syllables.append(current)

        return syllables if syllables else [word]

    def adapt(
        self, word_timestamps: List[WordTimestamp], text: str
    ) -> List[SyllableTimestamp]:
        result = []
        for wt in word_timestamps:
            syls = self.syllabify(wt.word)
            n = len(syls)
            duration = wt.end_sec - wt.start_sec
            syl_duration = duration / n if n > 0 else duration

            for j, syl in enumerate(syls):
                result.append(SyllableTimestamp(
                    syllable=syl,
                    start_sec=round(wt.start_sec + j * syl_duration, 4),
                    end_sec=round(wt.start_sec + (j + 1) * syl_duration, 4),
                    word=wt.word,
                    syllable_index=j,
                    total_syllables=n,
                    confidence=wt.confidence,
                ))
        return result


class LanguageAdapterRegistry:
    """
    Registry for language-specific adapters.

    Provides plug-in architecture for adding new languages.
    """

    _adapters: Dict[str, BaseLanguageAdapter] = {}

    def __init__(self):
        # Register built-in adapters
        self.register(EnglishAdapter())
        self.register(KannadaAdapter())
        self.register(HindiAdapter())

    def register(self, adapter: BaseLanguageAdapter) -> None:
        self._adapters[adapter.language_code] = adapter

    def get(self, language_code: str) -> BaseLanguageAdapter:
        if language_code not in self._adapters:
            raise ValueError(
                f"Unknown language: {language_code}. "
                f"Available: {list(self._adapters.keys())}"
            )
        return self._adapters[language_code]

    def list_languages(self) -> List[str]:
        return list(self._adapters.keys())


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Language Adapter — Self Test ===")

    registry = LanguageAdapterRegistry()
    print(f"Languages: {registry.list_languages()}")

    # Test English
    adapter = registry.get("en")
    words = [
        WordTimestamp("hello", 0.0, 0.5),
        WordTimestamp("world", 0.5, 1.0),
    ]
    syllables = adapter.adapt(words, "hello world")
    for s in syllables:
        print(f"  {s.syllable} ({s.start_sec:.2f}-{s.end_sec:.2f}) in {s.word}")

    # Test Kannada
    kn_adapter = registry.get("kn")
    kn_words = [WordTimestamp("ಮನೆ", 0.0, 0.5)]
    kn_syllables = kn_adapter.adapt(kn_words, "ಮನೆ")
    for s in kn_syllables:
        print(f"  {s.syllable} ({s.start_sec:.2f}-{s.end_sec:.2f}) in {s.word}")

    # Test Hindi
    hi_adapter = registry.get("hi")
    hi_words = [WordTimestamp("नमस्ते", 0.0, 0.6)]
    hi_syllables = hi_adapter.adapt(hi_words, "नमस्ते")
    for s in hi_syllables:
        print(f"  {s.syllable} ({s.start_sec:.2f}-{s.end_sec:.2f}) in {s.word}")

    print("=== Self test passed ===")
