import os
import csv
import glob

DATA_DIR = os.path.join("..", "data", "custom_landmarks")

def inspect_dataset():
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Directory not found: {os.path.abspath(DATA_DIR)}")
        return

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print("[WARNING] No CSV files found in data/custom_landmarks!")
        return

    print("==========================================")
    print("      DATASET INTEGRITY INSPECTOR         ")
    print("==========================================")
    print(f"Total CSV Files Found: {len(csv_files)}\n")

    class_counts = {}
    valid_files = 0
    corrupted_files = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        with open(file_path, mode="r") as f:
            reader = list(csv.reader(f))
            
            # Check 1: Header row + 30 frames = 31 rows
            if len(reader) != 31:
                corrupted_files.append((filename, f"Invalid row count ({len(reader)} rows, expected 31)"))
                continue
            
            # Check 2: Column count (63 landmark values + 1 label = 64 columns)
            header = reader[0]
            if len(header) != 64:
                corrupted_files.append((filename, f"Invalid column count ({len(header)} cols, expected 64)"))
                continue

            # Check 3: Read class label from last row
            label = reader[1][-1]
            class_counts[label] = class_counts.get(label, 0) + 1
            valid_files += 1

    print("📊 CLASS DISTRIBUTION SUMMARY:")
    print("------------------------------------------")
    for gesture, count in class_counts.items():
        print(f"  • {gesture:<15} : {count} samples")
    print("------------------------------------------")
    print(f"Total Valid Samples: {valid_files} / {len(csv_files)}\n")

    if corrupted_files:
        print("❌ CORRUPTED / BAD FILES DETECTED:")
        for name, reason in corrupted_files:
            print(f"  - {name}: {reason}")
    else:
        print("✅ ALL FILES PASSED INTEGRITY & FORMAT CHECKS!")
        print("   Your dataset structure is 100% valid for Deep Learning!")
    print("==========================================\n")

if __name__ == "__main__":
    inspect_dataset()