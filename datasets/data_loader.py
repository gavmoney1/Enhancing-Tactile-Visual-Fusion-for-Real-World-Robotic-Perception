import os
import re
import tensorflow as tf
from typing import List, Tuple, Dict
from collections import defaultdict
import random

class DataLoader:
    def __init__(self, config):
        self.orig_root = config['data']['orig_root']
        self.mask_root = config['data']['mask_root']
        self.img_size = config['data']['img_size']
        self.batch_size = config['training']['batch_size']
        self.seed = config['training']['seed']
        self.autotune = tf.data.AUTOTUNE

    def _imread(self, path: tf.Tensor) -> tf.Tensor:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # float32 [0,1]
        img = tf.image.resize(img, (self.img_size, self.img_size))
        return img

    def load_pair(self, rel_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        orig_path = tf.strings.join([self.orig_root, "/", rel_path])
        mask_path = tf.strings.join([self.mask_root, "/", rel_path])
        orig = self._imread(orig_path)
        mask = self._imread(mask_path)
        orig.set_shape([self.img_size, self.img_size, 3])
        mask.set_shape([self.img_size, self.img_size, 3])
        return mask, orig
    
    def _extract_video_id(self, fname: str) -> str:
        """
        Extract video identifier from filename or subdirectory.
        Handles:
            (1) matched_frames/20220503_100937/0000001.jpg -> 20220503_100937
            or 
            (2) matched_frames/20220503_100937_0000001.jpg -> 20220503_100937

            Raises:
                ValueError: if a valid video ID cannot be extracted
        """

        # normalize slashes
        fname = fname.replace("\\", "/")

        # For case 1: Search for the 8 pattern digits (date) + _ + 6 digits (time)
        match = re.search(r"\d{8}_\d{6}", fname)
        if match:
            return match.group(0)

        # For case 2: use filename minus last underscore
        base_name = os.path.basename(fname)
        if "_" in base_name:
            return "_".join(base_name.split("_")[:-1])
        
        # Path does not match expected patterns
        raise ValueError(f"Invalid path format for extracting video ID: {fname}")

    def _get_matching_files(self) -> Dict[str, List[str]]:
        """
        Match files between orig_root and mask_root, grouped by video ID.
        """
        matched_files = defaultdict(list)

        for root, _, files in os.walk(self.orig_root):
            for fname in files:
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                rel_path = os.path.relpath(os.path.join(root, fname), self.orig_root)
                video_id = self._extract_video_id(rel_path)
                mask_path = os.path.join(self.mask_root, rel_path)
                if os.path.exists(mask_path):
                    matched_files[video_id].append(rel_path)

        return matched_files

    def create_datasets(self, test_split=0.2, val_split=0.1) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        grouped_files = self._get_matching_files()
        video_ids = list(grouped_files.keys())
        video_ids.sort()
        random.seed(self.seed)
        random.shuffle(video_ids)

        n = len(video_ids)
        n_test = int(test_split * n)
        n_val = int(val_split * n)

        test_videos = video_ids[:n_test]
        val_videos = video_ids[n_test:n_test+n_val]
        train_videos = video_ids[n_test+n_val:]

        # Turn back into file paths
        train_paths = [p for vid in train_videos for p in grouped_files[vid]]
        val_paths = [p for vid in val_videos for p in grouped_files[vid]]
        test_paths = [p for vid in test_videos for p in grouped_files[vid]]

        print(f"Total videos: {n}")
        print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}, Test videos: {len(test_videos)}")

        train_ds = self._make_dataset(train_paths, shuffle=True)
        val_ds = self._make_dataset(val_paths)
        test_ds = self._make_dataset(test_paths)

        for mask, orig in train_ds.take(1):
            print("Mask dtype:", mask.dtype, "shape:", mask.shape)
            print("Orig dtype:", orig.dtype, "shape:", orig.shape)
        return train_ds, val_ds, test_ds

    def _make_dataset(self, paths: List[str], shuffle: bool = False) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.map(self.load_pair, num_parallel_calls=self.autotune)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(self.autotune)
        return ds
