import os
import csv
import glob
import numpy as np

RAW_DATA_DIR = os.path.join("..", "data", "custom_landmarks")
PROCESSED_DATA_DIR = os.path.join("..", "data", "processed")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

GESTURE_MAP = {
    "idle": 0,
    "point": 1,
    "left_click": 2,
    "right_click": 3,
    "scroll_up": 4,
    "scroll_down": 5
}

def normalize_landmarks_hybrid(sequence):
    """
    Hybrid Normalization:
    - 63 values: Finger positions relative to Wrist (Shape invariant to location)
    - 3 values : Absolute Wrist Screen Position (Preserves Top vs Bottom for Scroll)
    Total per frame = 66 features
    """
    hybrid_sequence = []
    
    for frame in sequence:
        landmarks = frame.reshape(21, 3) # (21, 3)
        
        wrist_abs = landmarks[0].copy() # Original screen position of wrist (x0, y0, z0)
        
        # 1. Shape Normalization (Relative to wrist)
        centered = landmarks - wrist_abs
        distances = np.linalg.norm(centered, axis=1)
        max_dist = np.max(distances)
        
        if max_dist > 0:
            shape_normalized = centered / max_dist
        else:
            shape_normalized = centered
            
        # Flatten relative landmarks (63 values)
        relative_flat = shape_normalized.flatten()
        
        # 2. Append Absolute Wrist Position (3 values)
        combined_frame = np.hstack((relative_flat, wrist_abs)) # 63 + 3 = 66 features
        
        hybrid_sequence.append(combined_frame)
        
    return np.array(hybrid_sequence)

def process_all_files():
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    
    X_data = [] # Landmark sequences
    y_data = [] # Numeric labels

    print("==========================================")
    print("      HYBRID DATASET PREPROCESSING        ")
    print("==========================================")

    for file_path in csv_files:
        with open(file_path, mode="r") as f:
            reader = list(csv.reader(f))
            data_rows = reader[1:]
            
            if len(data_rows) != 30:
                continue # Skip incomplete sequences
            
            raw_coords = np.array([row[:-1] for row in data_rows], dtype=np.float32)
            label_str = data_rows[0][-1]
            
            if label_str not in GESTURE_MAP:
                continue
                
            norm_coords = normalize_landmarks_hybrid(raw_coords)
            
            X_data.append(norm_coords)
            y_data.append(GESTURE_MAP[label_str])

    X_data = np.array(X_data, dtype=np.float32) # Shape: (N, 30, 66)
    y_data = np.array(y_data, dtype=np.int64)   # Shape: (N,)

    output_path = os.path.join(PROCESSED_DATA_DIR, "gesture_dataset.npz")
    np.savez_compressed(output_path, X=X_data, y=y_data)

    print(f"✅ Processed {len(X_data)} sequences successfully!")
    print(f"📦 Feature Array Shape (X) : {X_data.shape}  [Samples, Frames, 66 Features]")
    print(f"🏷️  Label Array Shape (y)   : {y_data.shape}")
    print(f"💾 Saved to: {os.path.abspath(output_path)}")
    print("==========================================\n")

if __name__ == "__main__":
    process_all_files()