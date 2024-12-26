import os
import json
import random

def split_dataset(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Splits the dataset into training, validation, and test sets.
    
    Args:
        source_dir (str): Directory containing the source .jsonl files.
        train_dir (str): Directory to save training data.
        val_dir (str): Directory to save validation data.
        test_dir (str): Directory to save test data.
        split_ratio (tuple): Ratios for train, validation, and test sets.
    """
    # Create directories for training, validation, and test sets
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List all .jsonl files in the source directory
    files = [f for f in os.listdir(source_dir) if f.endswith('.jsonl')]
    
    if not files:
        print("No .jsonl files found in the source directory.")
        return

    all_data = []

    # Read and collect all lines from each .jsonl file
    for file in files:
        with open(os.path.join(source_dir, file), 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(line.strip())  # Collect each line

    # Shuffle the collected data
    random.shuffle(all_data)

    # Calculate the split indices
    total_size = len(all_data)
    train_size = int(total_size * split_ratio[0])
    val_size = int(total_size * split_ratio[1])

    # Split into training, validation, and test sets
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]

    # Write training data to train directory
    with open(os.path.join(train_dir, 'train.jsonl'), 'w', encoding='utf-8') as train_file:
        for item in train_data:
            train_file.write(item + '\n')

    # Write validation data to validation directory
    with open(os.path.join(val_dir, 'val.jsonl'), 'w', encoding='utf-8') as val_file:
        for item in val_data:
            val_file.write(item + '\n')

    # Write test data to test directory
    with open(os.path.join(test_dir, 'test.jsonl'), 'w', encoding='utf-8') as test_file:
        for item in test_data:
            test_file.write(item + '\n')

    print(f'Split {total_size} files into {len(train_data)} for training, {len(val_data)} for validation, and {len(test_data)} for testing.')

# Usage
source_directory = 'data/raw'  # Directory containing your .jsonl files
train_directory = 'data/train_split'
val_directory = 'data/val_split'
test_directory = 'data/test_split'

split_dataset(source_directory, train_directory, val_directory, test_directory, split_ratio=(0.7, 0.15, 0.15))