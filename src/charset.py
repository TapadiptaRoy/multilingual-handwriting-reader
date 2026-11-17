"""
charset.py
The character set for the handwriting recognizer: a numbered list of every
character the model can output, with index 0 reserved for the CTC blank.
"""

import json
from pathlib import Path

import pandas as pd

BLANK_INDEX = 0


class Charset:
    """Converts between text and lists of numbers. Index 0 is the CTC blank."""

    def __init__(self, characters):
        self.characters = list(characters)
        # Real characters are numbered from 1, because 0 is the blank
        self.char_to_index = {char: i + 1 for i, char in enumerate(self.characters)}

    def __len__(self):
        """Number of model outputs: every character, plus the blank."""
        return len(self.characters) + 1

    def encode(self, text):
        """Turn a word into a list of numbers."""
        return [self.char_to_index[char] for char in text]

    def decode(self, indices):
        """Turn a list of numbers back into a word, skipping blanks."""
        return "".join(self.characters[i - 1] for i in indices if i != BLANK_INDEX)

    def covers(self, text):
        """True if every character in the text is in this character set."""
        return all(char in self.char_to_index for char in text)

    def save(self, path):
        """Save the character list to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.characters, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """Create a Charset from a saved JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))


def build_charset(manifest, language):
    """Build the character set from one language's TRAINING labels only."""
    is_train = (manifest["language"] == language) & (manifest["split"] == "train")
    train_labels = manifest.loc[is_train, "label"]
    characters = sorted(set("".join(train_labels)))
    return Charset(characters)


if __name__ == "__main__":
    # This test only runs when the file is run directly, not when it is imported.
    import unicodedata

    from src.utils import load_config

    config = load_config()
    rec_cfg = config["recognizer"]
    language = rec_cfg["language"]

    manifest = pd.read_csv(
        config["preprocess"]["processed_manifest_path"],
        encoding="utf-8",
        keep_default_na=False,
    )
    lang_rows = manifest[manifest["language"] == language]

    # 1. Build and save the character set
    charset = build_charset(manifest, language)
    charset.save(rec_cfg["charset_path"])
    print(f"{language}: {len(charset.characters)} characters + blank = {len(charset)} model outputs")
    print(f"Saved to: {rec_cfg['charset_path']}")

    # 2. Encode and decode a few real words
    print("\nExamples:")
    train_labels = lang_rows.loc[lang_rows["split"] == "train", "label"]
    for word in train_labels.head(5):
        encoded = charset.encode(word)
        decoded = charset.decode(encoded)
        status = "OK" if decoded == word else "MISMATCH"
        print(f"  {word} -> {encoded} -> {decoded}  [{status}]")

    # 3. Check every training label survives a round trip
    failures = sum(charset.decode(charset.encode(word)) != word for word in train_labels)
    print(f"\nRound-trip failures on all {len(train_labels):,} training labels: {failures}")

    # 4. Check that loading the saved file gives the same character set
    reloaded = Charset.load(rec_cfg["charset_path"])
    print(f"Reloaded from file matches: {reloaded.characters == charset.characters}")

    # 5. How many validation and test labels contain characters unseen in training?
    for split in ["val", "test"]:
        labels = lang_rows.loc[lang_rows["split"] == split, "label"]
        uncovered = labels[~labels.map(charset.covers)]
        print(f"\n{split}: {len(uncovered)} of {len(labels):,} labels contain characters unseen in training")
        unseen_chars = set("".join(uncovered)) - set(charset.characters)
        for char in sorted(unseen_chars):
            print(f"    {char}  U+{ord(char):04X}  {unicodedata.name(char, 'UNKNOWN')}")