import os
import re
from cyberpunk_dictionary import CYBERPUNK_TO_SCIFI

INPUT_DIR = "cyberpunk_pages"
OUTPUT_DIR = "cyberpunk_pages_modified"


def replace_with_dictionary(text, dictionary):
    """Replace all occurrences from dictionary keys with their values.
    
    Longer keys are replaced first to avoid partial replacements
    (e.g., 'Johnny Silverhand' before 'V' so 'V' in 'Silverhand' isn't touched).
    """
    # Sort by length (longest first) to avoid partial matches
    sorted_terms = sorted(dictionary.keys(), key=len, reverse=True)

    for term in sorted_terms:
        replacement = dictionary[term]
        # Use word boundaries for single-word terms, plain replace for multi-word
        if " " in term:
            text = text.replace(term, replacement)
        else:
            # For single words, use regex with word boundaries
            pattern = r'\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, replacement, text)

    return text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    print(f"Found {len(txt_files)} files in '{INPUT_DIR}'")
    print(f"Dictionary has {len(CYBERPUNK_TO_SCIFI)} replacements")
    print(f"Output directory: '{OUTPUT_DIR}'\n")

    processed = 0
    for filename in sorted(txt_files):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            original_text = f.read()

        modified_text = replace_with_dictionary(original_text, CYBERPUNK_TO_SCIFI)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(modified_text)

        # Count actual replacements made
        changes = sum(1 for key in CYBERPUNK_TO_SCIFI if key in original_text)
        print(f"[{processed + 1}/{len(txt_files)}] {filename} → {changes} terms replaced")
        processed += 1

    print(f"\nDone! Processed {processed} files.")
    print(f"Originals preserved in: '{INPUT_DIR}'")
    print(f"Modified files saved to: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
