import os
import re
import tensorflow as tf
from typing import List, Tuple, Dict
from collections import defaultdict
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DataLoader:
    def __init__(self, config):
        self.orig_root = config['data']['orig_root']
        self.mask_root = config['data'].get('mask_root', None)  # Make mask_root optional
        self.img_size = config['data']['img_size']
        self.batch_size = config['training']['batch_size']
        self.seed = config['training']['seed']
        self.autotune = tf.data.AUTOTUNE
        self.config = config
        self.val_split = self.config['training'].get('val_split', 0.1)
        self.test_split = self.config['training'].get('test_split', 0.2)

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

    def create_datasets(self):
        """Create train, validation, and test datasets"""
        task = self.config.get('task', 'reconstruction')
        
        if task == 'classification':
            return self._create_classification_datasets()
        else:
            return self._create_reconstruction_datasets()
    
    def _create_reconstruction_datasets(self):
        """Original reconstruction dataset creation"""
        grouped_files = self._get_matching_files()
        video_ids = list(grouped_files.keys())
        video_ids.sort()
        random.seed(self.seed)
        random.shuffle(video_ids)

        n = len(video_ids)
        n_test = int(self.test_split * n)
        n_val = int(self.val_split * n)

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
    
    def _create_classification_datasets(self):
        """Create datasets for classification task"""
        print("Creating classification datasets...")
        
        # For classification, use orig_root for images (no masked images needed)
        image_dir = self.config['data']['orig_root']
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Load labels from labels_path
        labels_path = self.config['data']['labels_path']
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        labels_df = pd.read_csv(labels_path)
        print(f"Loaded {len(labels_df)} labels from {labels_path}")
        
        # Get image paths (sorted to ensure consistent ordering)
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        # Match images with labels
        image_paths = [os.path.join(image_dir, f) for f in image_files]
        
        # Ensure we have a 'label' column
        if 'label' not in labels_df.columns:
            raise ValueError(f"Labels file must have a 'label' column. Found columns: {labels_df.columns.tolist()}")
        
        labels = labels_df['label'].values[:len(image_paths)]
        
        if len(image_paths) != len(labels):
            print(f"Warning: Number of images ({len(image_paths)}) != number of labels ({len(labels)})")
            min_len = min(len(image_paths), len(labels))
            image_paths = image_paths[:min_len]
            labels = labels[:min_len]
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {list(label_encoder.classes_)}")
        
        # Update config with detected number of classes
        if 'classification' not in self.config:
            self.config['classification'] = {}
        self.config['classification']['num_classes'] = num_classes
        self.config['classification']['class_names'] = label_encoder.classes_.tolist()
        
        # Split dataset using seed for reproducibility
        np.random.seed(self.config['training']['seed'])
        
        indices = np.random.permutation(len(image_paths))
        
        total_size = len(image_paths)
        train_size = int((1.0 - self.val_split - self.test_split) * total_size)
        val_size = int(self.val_split * total_size)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_paths = [image_paths[i] for i in train_indices]
        train_labels = encoded_labels[train_indices]
        
        val_paths = [image_paths[i] for i in val_indices]
        val_labels = encoded_labels[val_indices]
        
        test_paths = [image_paths[i] for i in test_indices]
        test_labels = encoded_labels[test_indices]
        
        print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        # Create TF datasets
        train_ds = self._create_classification_dataset(train_paths, train_labels, is_training=True)
        val_ds = self._create_classification_dataset(val_paths, val_labels, is_training=False)
        test_ds = self._create_classification_dataset(test_paths, test_labels, is_training=False)
        
        return train_ds, val_ds, test_ds
    
    def _create_classification_dataset(self, image_paths, labels, is_training=False):
        """Create a single classification dataset"""
        batch_size = self.config['training']['batch_size']
        
        def load_and_preprocess(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, [self.img_size, self.img_size])
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000, seed=self.config['training']['seed'])
        
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def _make_dataset(self, paths: List[str], shuffle: bool = False) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.map(self.load_pair, num_parallel_calls=self.autotune)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(self.autotune)
        return ds
