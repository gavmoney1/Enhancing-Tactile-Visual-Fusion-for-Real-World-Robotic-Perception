import tensorflow as tf
from abc import ABC, abstractmethod

class BaseTransformerModel(ABC):
    """Base class for all transformer models"""
    
    def __init__(self, config):
        self.config = config
        self.img_size = config['data']['img_size']
        self.patch_size = config.get('patch_size', 16)
        self.embed_dim = config.get('embed_dim', 384)
        self.num_heads = config.get('num_heads', 6)
        self.num_layers = config.get('num_layers', 6)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        
    @abstractmethod
    def build_model(self):
        """Build and return the model"""
        pass
    
    def create_patches(self, images):
        """Convert images to patches"""
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        num_patches = (self.img_size // self.patch_size) ** 2
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])
        return patches
    
    def patches_to_image(self, patches):
        """Reconstruct image from patches"""
        batch_size = tf.shape(patches)[0]
        h = w = self.img_size // self.patch_size
        patches = tf.reshape(patches, [batch_size, h, w, self.patch_size, self.patch_size, 3])
        patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
        images = tf.reshape(patches, [batch_size, self.img_size, self.img_size, 3])
        return images
