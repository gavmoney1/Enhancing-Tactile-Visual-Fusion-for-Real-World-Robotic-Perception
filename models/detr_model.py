import tensorflow as tf
from tensorflow.keras import layers
from .base_model import BaseTransformerModel

class QueryTiling(layers.Layer):
    """Tile object queries for batch size"""
    def __init__(self, num_queries, embed_dim):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.object_queries = self.add_weight(
            shape=(1, self.num_queries, self.embed_dim),
            initializer="random_normal",
            name="object_queries"
        )
        
    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.tile(self.object_queries, [batch_size, 1, 1])

class PatchesToImage(layers.Layer):
    """Convert patches back to image format"""
    def __init__(self, img_size, patch_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = int((img_size // patch_size) ** 2)
        
    def call(self, patches):
        batch_size = tf.shape(patches)[0]
        h_patches = w_patches = int(self.num_patches ** 0.5)
        
        # Reshape to image
        outputs = tf.reshape(patches, [batch_size, h_patches, w_patches, 
                                     self.patch_size, self.patch_size, 3])
        outputs = tf.transpose(outputs, [0, 1, 3, 2, 4, 5])
        outputs = tf.reshape(outputs, [batch_size, self.img_size, self.img_size, 3])
        
        return outputs

class PositionalEncoding(layers.Layer):
    """Positional encoding layer"""
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.pos_encoding = self.add_weight(
            shape=(1, self.max_len, self.embed_dim),
            initializer="random_normal",
            name="pos_encoding"
        )
        
    def call(self, x):
        return x + self.pos_encoding

class DETRModel(BaseTransformerModel):
    """DETR-style transformer for image reconstruction"""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_queries = config.get('num_queries', 196)  # 14x14 for 224x224 image
        
    def build_model(self):
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # CNN backbone for feature extraction
        x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
        x = layers.MaxPool2D(2)(x)
        
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.MaxPool2D(2)(x)
        
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.MaxPool2D(2)(x)
        
        # Flatten spatial dimensions but keep feature channels
        h = x.shape[1]
        w = x.shape[2]
        c = x.shape[3]
        x = layers.Reshape((h * w, c))(x)
        
        # Project to embedding dimension
        x = layers.Dense(self.embed_dim)(x)
        
        # Add positional encoding
        x = PositionalEncoding(h * w, self.embed_dim)(x)
        
        queries = QueryTiling(self.num_queries, self.embed_dim)(tf.zeros_like(x))
        queries = QueryTiling(self.num_queries, self.embed_dim)(x)
        
        # Transformer decoder layers
        for i in range(self.num_layers):
            queries = self._decoder_layer(queries, x, name=f'decoder_{i}')
        
        # Output projection
        outputs = layers.Dense(self.patch_size * self.patch_size * 3)(queries)
        
        # Reshape to image using custom layer
        outputs = PatchesToImage(self.img_size, self.patch_size)(outputs)
        outputs = layers.Activation('sigmoid')(outputs)
        
        return tf.keras.Model(inputs, outputs, name="DETR_Reconstruction")
    
    def _decoder_layer(self, queries, encoder_output, name):
        """Single transformer decoder layer"""
        # Self-attention on queries
        queries_norm = layers.LayerNormalization(name=f'{name}_self_norm')(queries)
        self_attn = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.embed_dim // self.num_heads,
            name=f'{name}_self_attn'
        )
        queries = queries + self_attn(queries_norm, queries_norm)
        
        # Cross-attention with encoder output
        queries_norm = layers.LayerNormalization(name=f'{name}_cross_norm')(queries)
        cross_attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads, 
            name=f'{name}_cross_attn'
        )
        queries = queries + cross_attn(queries_norm, encoder_output)
        
        # Feed-forward network
        queries_norm = layers.LayerNormalization(name=f'{name}_ff_norm')(queries)
        ff_dim = int(self.embed_dim * self.mlp_ratio)
        ff = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(self.embed_dim)
        ], name=f'{name}_ff')
        
        queries = queries + ff(queries_norm)
        
        return queries
