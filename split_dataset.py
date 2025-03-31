import os
import json
import random
from collections import defaultdict

def split_dataset(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Splits the dataset into training, validation, and testing sets with balanced labels.
    
    Args:
        source_dir (str): Directory containing the source .jsonl files.
        train_dir (str): Directory to save training data.
        val_dir (str): Directory to save validation data.
        test_dir (str): Directory to save test data.
        split_ratio (tuple): Ratios for train, validation, and test sets (must sum to 1).
    """
    # Validate split_ratio
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    # Create directories for training, validation, and test sets
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List all .jsonl files in the source directory
    files = [f for f in os.listdir(source_dir) if f.endswith('.jsonl')]
    
    if not files:
        print("No .jsonl files found in the source directory.")
        return

    # Group data by label
    label_groups = defaultdict(list)
    for file in files:
        with open(os.path.join(source_dir, file), 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    label = data.get('label')  # Assumes each JSON object has a 'label' field
                    if label is not None:
                        label_groups[label].append(line.strip())
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {file}")
                    continue

    # Shuffle and split each label group
    train_data = []
    val_data = []
    test_data = []
    label_counts_train = defaultdict(int)
    label_counts_val = defaultdict(int)
    label_counts_test = defaultdict(int)

    for label, items in label_groups.items():
        random.shuffle(items)
        total_size = len(items)
        
        # Calculate split sizes
        train_size = int(total_size * split_ratio[0])
        val_size = int(total_size * split_ratio[1])
        test_size = total_size - train_size - val_size  # Ensure all items are used
        
        # Split data for this label
        train_items = items[:train_size]
        val_items = items[train_size:train_size + val_size]
        test_items = items[train_size + val_size:]
        
        # Add to respective sets
        train_data.extend(train_items)
        val_data.extend(val_items)
        test_data.extend(test_items)
        
        # Count labels
        label_counts_train[label] += len(train_items)
        label_counts_val[label] += len(val_items)
        label_counts_test[label] += len(test_items)

    # Shuffle the final sets
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Write training data
    with open(os.path.join(train_dir, 'train.jsonl'), 'w', encoding='utf-8') as train_file:
        for item in train_data:
            train_file.write(item + '\n')

    # Write validation data
    with open(os.path.join(val_dir, 'val.jsonl'), 'w', encoding='utf-8') as val_file:
        for item in val_data:
            val_file.write(item + '\n')

    # Write test data
    with open(os.path.join(test_dir, 'test.jsonl'), 'w', encoding='utf-8') as test_file:
        for item in test_data:
            test_file.write(item + '\n')

    # Print statistics
    total_size = len(train_data) + len(val_data) + len(test_data)
    print(f'Split {total_size} items into {len(train_data)} for training, '
          f'{len(val_data)} for validation, and {len(test_data)} for testing.')
    
    print("\nTraining set label counts:")
    for label, count in sorted(label_counts_train.items()):
        print(f"  {label}: {count}")
    
    print("\nValidation set label counts:")
    for label, count in sorted(label_counts_val.items()):
        print(f"  {label}: {count}")
    
    print("\nTesting set label counts:")
    for label, count in sorted(label_counts_test.items()):
        print(f"  {label}: {count}")

# Usage
source_directory = 'data/raw'  # Directory containing your .jsonl files
train_directory = 'data/train_split'
val_directory = 'data/val_split'
test_directory = 'data/test_split'

split_dataset(source_directory, train_directory, val_directory, test_directory, split_ratio=(0.7, 0.15, 0.15))