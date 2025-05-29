import json
import random
import re
from collections import Counter


def contains_emoji(text):
    """
    Check if the input text contains emojis.

    Args:
        text (str): Input text to check

    Returns:
        bool: True if emojis are present, False otherwise
    """
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"   # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"   # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"   # Flags
        "\U00002702-\U000027B0"   # Dingbats
        "\U000024C2-\U0001F251]+"  # Enclosed characters
        , flags=re.UNICODE
    )
    return bool(emoji_pattern.search(text))


def contains_html(text):
    """
    Check if the input text contains HTML tags.

    Args:
        text (str): Input text to check

    Returns:
        bool: True if HTML tags are present, False otherwise
    """
    html_pattern = re.compile(r'<[^>]+>')
    return bool(html_pattern.search(text))


def normalize_text(text):
    """
    Normalize text by converting to lowercase and collapsing whitespace.

    Args:
        text (str): Input text

    Returns:
        str: Normalized text
    """
    return ' '.join(text.lower().strip().split())


def process_jsonl_dataset(file_path, output_path, max_words=50, rating_threshold=3, keep_emojis=False):
    """
    Process a JSONL dataset by cleaning, labeling, and balancing it.

    Args:
        file_path (str): Path to the input JSONL dataset
        output_path (str): Path to save the processed JSONL dataset
        max_words (int): Maximum number of words allowed in text
        rating_threshold (float): Threshold for labeling positive (1) vs. negative (0)
        keep_emojis (bool): Whether to retain entries with emojis
    """
    # Load JSONL data
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = []
            for i, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON at line {i}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    print(f"Original dataset size: {len(data)}")

    # Process and label data
    labeled_data = []
    for entry in data:
        # Check for required fields
        if 'rating' not in entry or 'text' not in entry:
            continue
        text = entry['text']
        rating = entry['rating']

        # Validate rating type
        if not isinstance(rating, (int, float)):
            continue

        # Apply cleaning filters: remove emojis (if not keeping), HTML, empty text, or overly long text
        if (
            (not keep_emojis and contains_emoji(text)) or
            contains_html(text) or
            text.strip() == "" or
            len(text.split()) > max_words
        ):
            continue

        # Normalize text and assign label based on rating
        text = normalize_text(text)
        label = 1 if rating > rating_threshold else 0
        labeled_data.append({'label': label, 'text': text})

    print(f"Labeled dataset size after cleaning: {len(labeled_data)}")
    # Count label distribution
    label_counts = Counter(entry['label'] for entry in labeled_data)
    print(f"Label distribution before balancing: {label_counts}")

    # Balance dataset by sampling equal numbers of positive and negative examples
    min_count = min(label_counts.values(), default=0)
    if min_count == 0:
        print("Error: One or more label classes have no samples after cleaning.")
        return

    positive_samples = [entry for entry in labeled_data if entry['label'] == 1]
    negative_samples = [entry for entry in labeled_data if entry['label'] == 0]
    balanced_data = random.sample(positive_samples, min_count) + random.sample(negative_samples, min_count)
    random.shuffle(balanced_data)
    print(f"Balanced dataset size: {len(balanced_data)}")

    # Save processed dataset to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in balanced_data:
            json.dump(entry, f)
            f.write('\n')
    print(f"Processed dataset saved to: {output_path}")


# Example usage of the processing function
file_path = 'data/dl/Digital_Music.jsonl'  # Input JSONL file
output_path = 'data/raw/processed_Digital_Music.jsonl'  # Output processed file
# Process the dataset with specified parameters
process_jsonl_dataset(file_path, output_path, max_words=50, rating_threshold=3, keep_emojis=False)
