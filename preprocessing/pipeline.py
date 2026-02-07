"""
Main preprocessing pipeline orchestrator.
Coordinates all preprocessing steps for PPG signals from multiple datasets.
Supports supervised learning by extracting and synchronizing labels.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .loaders import load_ppg_dalia, load_wesad, load_bidmc, resample_signal, load_all_subjects
from .filters import apply_preprocessing_filters
from .normalization import normalize_windows
from .windowing import segment_signal, reject_low_quality_windows, create_train_val_test_split
from .quality import batch_quality_assessment


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for PPG signals.
    """
    
    def __init__(self, 
                 target_fs: float = 64.0,
                 window_duration: float = 10.0,
                 overlap: float = 0.5,
                 quality_threshold: float = 0.5,
                 normalization_method: str = 'zscore',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 output_dir: str = "preprocessed_data",
                 random_seed: int = 42):
        self.target_fs = target_fs
        self.window_duration = window_duration
        self.overlap = overlap
        self.quality_threshold = quality_threshold
        self.normalization_method = normalization_method
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_subjects': 0,
            'total_windows': 0,
            'accepted_windows': 0,
            'rejected_windows': 0,
            'datasets': {}
        }
    
    def process_single_subject(self, signal: np.ndarray, fs: float, 
                               labels: Optional[Any], metadata: Dict, 
                               acc_signal: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Process a single subject's data through the complete pipeline.
        """
        subject_id = metadata['subject_id']
        print(f"\nProcessing {subject_id}...")
        
        # Standardize labels to dict if needed
        labels_dict = None
        if labels is not None:
            if isinstance(labels, dict):
                labels_dict = labels
            else:
                labels_dict = {'label': labels}
        
        # Step 1: Pre-resampling filtering (for clinical data like BIDMC)
        if metadata.get('dataset') == 'BIDMC':
            try:
                signal = apply_preprocessing_filters(
                    signal, 
                    fs, 
                    remove_powerline=True,
                    powerline_freq=50.0
                )
                print(f"  ✓ Powerline removed at native {fs} Hz")
            except Exception as e:
                print(f"  [WARNING] Pre-resampling filtering failed: {e}")

        # Step 2: Resample to target frequency
        if fs != self.target_fs:
            signal = resample_signal(signal, fs, self.target_fs)
            if acc_signal is not None:
                acc_signal = resample_signal(acc_signal, fs, self.target_fs)
            print(f"  ✓ Resampled: {fs} Hz -> {self.target_fs} Hz")
        
        # Step 3: Apply remaining filtering (Band-pass 0.5-8 Hz)
        try:
            signal = apply_preprocessing_filters(
                signal, 
                self.target_fs, 
                acc_sig=acc_signal,
                use_wavelet=False,
                remove_powerline=metadata.get('dataset') == 'BIDMC' and fs < 100 # Redundant check
            )
            print(f"  [OK] Filtered: Band-pass 0.5-8 Hz applied")
        except Exception as e:
            print(f"  [ERROR] Filtering failed for {subject_id}: {e}")
            return None
        
        # Step 4: Segment into windows
        try:
            windows, indices, seg_labels = segment_signal(
                signal, 
                self.target_fs, 
                window_duration=self.window_duration,
                overlap=self.overlap,
                labels=labels_dict
            )
            if len(windows) == 0:
                return None
            print(f"  [OK] Segmented: {len(windows)} windows created")
        except Exception as e:
            print(f"  [ERROR] Segmentation failed for {subject_id}: {e}")
            return None
        
        # Step 5: Assess quality and reject low-quality windows
        try:
            filtered_windows, quality_mask, filtered_labels = reject_low_quality_windows(
                windows, 
                self.target_fs,
                quality_threshold=self.quality_threshold,
                labels=seg_labels
            )
            
            rejection_rate = 1 - np.mean(quality_mask) if len(quality_mask) > 0 else 1.0
            
            if len(filtered_windows) == 0:
                print(f"  [SKIP] No windows passed quality threshold for {subject_id}")
                return None
        except Exception as e:
            print(f"  [ERROR] Quality assessment failed for {subject_id}: {e}")
            return None
        
        # Step 6: Normalize windows
        normalized_windows = normalize_windows(filtered_windows, method=self.normalization_method)
        print(f"  [OK] Normalized: {self.normalization_method} applied")
        
        # Update statistics
        self.stats['total_windows'] += len(windows)
        self.stats['accepted_windows'] += len(filtered_windows)
        self.stats['rejected_windows'] += len(windows) - len(filtered_windows)
        
        return {
            'windows': normalized_windows,
            'labels': filtered_labels,
            'subject_id': subject_id,
            'metadata': metadata
        }
    
    def process_dataset(self, dataset_name: str, base_path: Optional[str] = None) -> List[Dict]:
        """Process all subjects from a dataset."""
        print(f"\n{'='*60}\nProcessing {dataset_name.upper()}\n{'='*60}")
        
        all_subjects = load_all_subjects(dataset_name, base_path)
        processed_subjects = []
        
        for signal, fs, labels, metadata in all_subjects:
            result = self.process_single_subject(signal, fs, labels, metadata)
            if result is not None:
                processed_subjects.append(result)
                self.stats['total_subjects'] += 1
        
        self.stats['datasets'][dataset_name] = {
            'n_subjects': len(processed_subjects),
            'n_windows': sum(len(s['windows']) for s in processed_subjects)
        }
        return processed_subjects
    
    def create_splits_and_save(self, processed_data: List[Dict], dataset_name: str):
        """Create splits and save X and y files."""
        print(f"\nCreating splits for {dataset_name}...")
        
        all_windows = []
        all_labels = {}
        all_subject_ids = []
        
        # Collect data
        for subject_data in processed_data:
            windows = subject_data['windows']
            labels = subject_data['labels']
            subject_id = subject_data['subject_id']
            
            all_windows.append(windows)
            all_subject_ids.extend([subject_id] * len(windows))
            
            if labels:
                for k, v in labels.items():
                    if k not in all_labels: all_labels[k] = []
                    all_labels[k].append(v)
        
        all_windows = np.vstack(all_windows)
        all_subject_ids = np.array(all_subject_ids)
        combined_labels = {k: np.concatenate(v) for k, v in all_labels.items()} if all_labels else None
        
        splits = create_train_val_test_split(
            all_windows,
            labels=combined_labels,
            subject_ids=all_subject_ids,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            random_seed=self.random_seed
        )
        
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name in ['train', 'val', 'test']:
            split_dir = dataset_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            windows = splits[split_name]['windows']
            labels = splits[split_name]['labels']
            s_ids = splits[split_name]['subject_ids']
            
            windows_reshaped = windows[..., np.newaxis]  # (N, T, 1)
            np.save(split_dir / 'signals.npy', windows_reshaped.astype(np.float32))
            
            # Save labels
            if labels:
                if 'hr' in labels:
                    np.save(split_dir / 'labels_hr.npy', labels['hr'].astype(np.float32))
                if 'rr' in labels:
                    np.save(split_dir / 'labels_rr.npy', labels['rr'].astype(np.float32))
                # For backward compatibility, save primary label as y.npy
                primary = labels.get('hr', labels.get('label'))
                if primary is not None:
                    np.save(split_dir / 'y.npy', primary.astype(np.float32))

            metadata = {
                'n_samples': len(windows),
                'sampling_rate': self.target_fs,
                'window_length': windows.shape[1],
                'subject_ids': s_ids.tolist() if s_ids is not None else None,
                'labels_available': list(labels.keys()) if labels else []
            }
            with open(split_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  [OK] Saved {split_name}: {len(windows)} windows")

    def run(self, datasets: List[str] = ['ppg_dalia', 'wesad', 'bidmc']):
        for d in datasets:
            try:
                processed = self.process_dataset(d)
                if processed:
                    self.create_splits_and_save(processed, d)
            except Exception as e:
                warnings.warn(f"Failed dataset {d}: {e}")
        self.save_configuration()

    def save_configuration(self):
        config = {
            'target_fs': self.target_fs,
            'statistics': self.stats
        }
        with open(self.output_dir / 'preprocessing_config.json', 'w') as f:
            json.dump(config, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=['ppg_dalia', 'wesad', 'bidmc'])
    args = parser.parse_args()
    
    pipeline = PreprocessingPipeline(output_dir="preprocessed_data")
    pipeline.run(datasets=args.datasets)
