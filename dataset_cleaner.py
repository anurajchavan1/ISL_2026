"""
dataset_cleaner.py

Cleans raw_videos before extraction:
  1. Removes classes with fewer than MIN_VIDEOS videos
  2. Removes folders named 'Extra' or 'extra'
  3. Detects and removes duplicate video files within each class
     (same file size = likely duplicate)
  4. Prints a full report before deleting anything
  5. Asks for confirmation before making any changes

Run this ONCE before extract_features.py.
"""

import os
import shutil
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================
RAW_VIDEOS_DIR = os.path.join(os.getcwd(), 'raw_videos')
MIN_VIDEOS     = 10   # classes with fewer videos than this are removed
VALID_EXT      = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


# ==========================================
# SCAN
# ==========================================
def scan_dataset():
    classes_to_remove  = []   # (path, reason)
    classes_to_keep    = []   # (path, word, video_count)
    duplicates_to_remove = [] # (path, reason)

    for cat in sorted(os.listdir(RAW_VIDEOS_DIR)):
        cat_path = os.path.join(RAW_VIDEOS_DIR, cat)
        if not os.path.isdir(cat_path):
            continue

        for word in sorted(os.listdir(cat_path)):
            word_path = os.path.join(cat_path, word)
            if not os.path.isdir(word_path):
                continue

            # Remove Extra/extra junk folders
            if word.lower().strip() == 'extra':
                classes_to_remove.append((word_path, 'junk folder (Extra)'))
                continue

            # Get all videos
            videos = [f for f in os.listdir(word_path)
                      if f.lower().endswith(VALID_EXT)]

            # Detect duplicates by file size
            size_map = defaultdict(list)
            for v in videos:
                size = os.path.getsize(os.path.join(word_path, v))
                size_map[size].append(v)

            dups = []
            for size, files in size_map.items():
                if len(files) > 1:
                    # Keep first, mark rest as duplicates
                    for dup in files[1:]:
                        dups.append(os.path.join(word_path, dup))

            if dups:
                for d in dups:
                    duplicates_to_remove.append((d, f'duplicate in {word}'))

            # Count unique videos (after removing dups)
            unique_count = len(videos) - len(dups)

            if unique_count < MIN_VIDEOS:
                classes_to_remove.append(
                    (word_path,
                     f'only {unique_count} unique videos (min={MIN_VIDEOS})')
                )
            else:
                classes_to_keep.append((word_path, word, unique_count))

    return classes_to_keep, classes_to_remove, duplicates_to_remove


# ==========================================
# REPORT
# ==========================================
def print_report(keep, remove, dups):
    print("\n" + "="*60)
    print("DATASET CLEANING REPORT")
    print("="*60)

    print(f"\n✓ Classes to KEEP ({len(keep)}):")
    for path, word, count in sorted(keep, key=lambda x: x[1]):
        print(f"    {word:<40} {count:>3} videos")

    print(f"\n✗ Classes to REMOVE ({len(remove)}):")
    for path, reason in remove:
        name = os.path.basename(path)
        print(f"    {name:<40} {reason}")

    print(f"\n⚠ Duplicate files to REMOVE ({len(dups)}):")
    for path, reason in dups:
        print(f"    {os.path.basename(path):<40} {reason}")

    print(f"\nSummary:")
    print(f"  Classes kept    : {len(keep)}")
    print(f"  Classes removed : {len(remove)}")
    print(f"  Duplicates removed: {len(dups)}")
    total_kept = sum(c for _, _, c in keep)
    print(f"  Videos after cleaning: ~{total_kept}")
    print()


# ==========================================
# EXECUTE
# ==========================================
def clean(keep, remove, dups):
    # Remove duplicate files first
    for path, _ in dups:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed duplicate: {path}")

    # Remove low-sample and junk class folders
    for path, reason in remove:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"  Removed class: {os.path.basename(path)} ({reason})")

    # Remove empty category folders
    for cat in os.listdir(RAW_VIDEOS_DIR):
        cat_path = os.path.join(RAW_VIDEOS_DIR, cat)
        if os.path.isdir(cat_path) and not os.listdir(cat_path):
            shutil.rmtree(cat_path)
            print(f"  Removed empty category: {cat}")

    print("\nCleaning complete.")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("Scanning dataset...")
    keep, remove, dups = scan_dataset()
    print_report(keep, remove, dups)

    if not remove and not dups:
        print("Dataset is already clean. Nothing to do.")
        exit()

    confirm = input("Proceed with cleaning? This cannot be undone. (yes/no): ")
    if confirm.strip().lower() == 'yes':
        clean(keep, remove, dups)
    else:
        print("Aborted. No changes made.")