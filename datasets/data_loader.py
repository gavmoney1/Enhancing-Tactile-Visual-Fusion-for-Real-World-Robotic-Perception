import os
import tensorflow as tf
from typing import List, Tuple

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

    def _get_matching_files(self) -> List[str]:
        orig_files = set(os.listdir(self.orig_root))
        mask_files = set(os.listdir(self.mask_root))
        matched_files = list(orig_files & mask_files)
        matched_files.sort()
        return matched_files

    def create_datasets(self, test_split=0.1, val_split=0.1) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        paths = self._get_matching_files()
        tf.random.set_seed(self.seed)
        paths = tf.random.shuffle(paths)

        n = len(paths)
        n_test = int(test_split * n)
        n_val = int(val_split * n)

        test_paths = paths[:n_test]
        val_paths = paths[n_test:n_test+n_val]
        train_paths = paths[n_test+n_val:]

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
