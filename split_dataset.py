import os
import json
import random
from collections import defaultdict


def split_dataset(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Split a JSONL dataset into training, validation, and test sets with balanced label distributions.

    Args:
        source_dir (str): Directory containing the source .jsonl files.
        train_dir (str): Directory to save training data.
        val_dir (str): Directory to save validation data.
        test_dir (str): Directory to save test data.
        split_ratio (tuple): Ratios for train, validation, and test sets (must sum to 1).
    """
    # Validate that split ratios sum to 1
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    # Create output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of .jsonl files in the source directory
    files = [f for f in os.listdir(source_dir) if f.endswith('.jsonl')]
    
    # Check if any .jsonl files were found
    if not files:
        print("No .jsonl files found in the source directory.")
        return

    # Group data by label to ensure balanced splits
    label_groups = defaultdict(list)
    for file in files:
        with open(os.path.join(source_dir, file), 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    label = data.get('label')  # Extract label from JSON object
                    if label is not None:
                        label_groups[label].append(line.strip())  # Store raw JSON line
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {file}")
                    continue

    # Initialize lists for split data
    train_data = []
    val_data = []
    test_data = []
    # Initialize counters for label distribution statistics
    label_counts_train = defaultdict(int)
    label_counts_val = defaultdict(int)
    label_counts_test = defaultdict(int)

    # Split data for each label group
    for label, items in label_groups.items():
        random.shuffle(items)  # Shuffle items to ensure random splits
        total_size = len(items)
        
        # Calculate sizes for each split
        train_size = int(total_size * split_ratio[0])
        val_size = int(total_size * split_ratio[1])
        test_size = total_size - train_size - val_size  # Use remainder for test set
        
        # Assign items to train, validation, and test sets
        train_items = items[:train_size]
        val_items = items[train_size:train_size + val_size]
        test_items = items[train_size + val_size:]
        
        # Add items to respective sets
        train_data.extend(train_items)
        val_data.extend(val_items)
        test_data.extend(test_items)
        
        # Update label counts
        label_counts_train[label] += len(train_items)
        label_counts_val[label] += len(val_items)
        label_counts_test[label] += len(test_items)

    # Shuffle final sets to mix labels
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Write training data to file
    with open(os.path.join(train_dir, 'train.jsonl'), 'w', encoding='utf-8') as train_file:
        for item in train_data:
            train_file.write(item + '\n')

    # Write validation data to file
    with open(os.path.join(val_dir, 'val.jsonl'), 'w', encoding='utf-8') as val_file:
        for item in val_data:
            val_file.write(item + '\n')

    # Write test data to file
    with open(os.path.join(test_dir, 'test.jsonl'), 'w', encoding='utf-8') as test_file:
        for item in test_data:
            test_file.write(item + '\n')

    # Print split statistics
    total_size = len(train_data) + len(val_data) + len(test_data)
    print(f'Split {total_size} items into {len(train_data)} for training, '
          f'{len(val_data)} for validation, and {len(test_data)} for testing.')
    
    # Print label distribution for training set
    print("\nTraining set label counts:")
    for label, count in sorted(label_counts_train.items()):
        print(f"  {label}: {count}")
    
    # Print label distribution for validation set
    print("\nValidation set label counts:")
    for label, count in sorted(label_counts_val.items()):
        print(f"  {label}: {count}")
    
    # Print label distribution for testing set
    print("\nTesting set label counts:")
    for label, count in sorted(label_counts_test.items()):
        print(f"  {label}: {count}")


# Example usage of the split_dataset function
source_directory = 'data/raw'  # Directory containing raw .jsonl files
train_directory = 'data/train_split'  # Output directory for training data
val_directory = 'data/val_split'    # Output directory for validation data
test_directory = 'data/test_split'   # Output directory for test data

# Call the function to split the dataset with a 70-15-15 ratio
split_dataset(source_directory, train_directory, val_directory, test_directory, split_ratio=(0.7, 0.15, 0.15))
