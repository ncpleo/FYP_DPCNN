import json
import random
import re
from collections import Counter

def contains_emoji(text):
    """Check if a string contains emojis."""
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # other symbols
        "\U000024C2-\U0001F251"  # enclosed characters
        "]+",
        flags=re.UNICODE,
    )
    return bool(emoji_pattern.search(text))

def contains_html(text):
    """Check if a string contains HTML tags."""
    html_pattern = re.compile(r'<[^>]+>')
    return bool(html_pattern.search(text))

def process_jsonl_dataset(file_path, output_path):
    """
    Process a JSONL dataset by labeling, balancing, and cleaning it.

    Args:
        file_path (str): Path to the input JSONL dataset.
        output_path (str): Path to save the processed JSONL dataset.
    """
    # Check if the file exists
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    print(f"Original dataset size: {len(data)}")

    # Label data based on "rating"
    labeled_data = []
    for entry in data:
        if 'rating' in entry and 'text' in entry:
            text = entry['text']
            # Skip entries with emojis, HTML tags, empty text, or text longer than 50 words
            if (
                contains_emoji(text) or
                contains_html(text) or
                text.strip() == "" or
                len(text.split()) > 50
            ):
                continue

            label = 1 if entry['rating'] > 3 else 0
            labeled_data.append({'label': label, 'text': text})

    print(f"Labeled dataset size after cleaning: {len(labeled_data)}")

    # Count label distribution
    label_counts = Counter(entry['label'] for entry in labeled_data)
    print(f"Label distribution before balancing: {label_counts}")

    # Balance the dataset
    min_count = min(label_counts.values())
    balanced_data = []

    # Separate data by label
    positive_samples = [entry for entry in labeled_data if entry['label'] == 1]
    negative_samples = [entry for entry in labeled_data if entry['label'] == 0]

    # Randomly sample to balance
    balanced_data.extend(random.sample(positive_samples, min_count))
    balanced_data.extend(random.sample(negative_samples, min_count))

    # Shuffle the balanced dataset
    random.shuffle(balanced_data)
    print(f"Balanced dataset size: {len(balanced_data)}")

    # Save cleaned and balanced data to a new JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in balanced_data:
            json.dump(entry, f)
            f.write('\n')

    print(f"Processed dataset saved to: {output_path}")

# Example usage
file_path = 'data/dl/Digital_Music.jsonl'  # Replace with your input file path
output_path = 'data/raw/processed_Digital_Music.jsonl'  # Replace with your output file path

process_jsonl_dataset(file_path, output_path)
