import tensorflow as tf
from tensorflow.keras import layers
from .base_model import BaseTransformerModel
import math
from typing import Tuple

class PatchEmbed(layers.Layer):
    """Convert images to patches using convolution"""
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, 
                                 strides=patch_size, name='proj')
        
    def call(self, x):
        x = self.proj(x)
        # Rearrange to tokens sequence
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B, H*W, C])
        return x

class MLP(layers.Layer):
    """MLP with GELU activation"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])
    
    def call(self, x, training=False):
        return self.net(x, training=training)

class TransformerBlock(layers.Layer):
    """Transformer block with pre-norm architecture"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, 
                                            key_dim=dim//num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        
    def call(self, x, training=False):
        x = x + self.attn(self.norm1(x), self.norm1(x), 
                         self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x

class MAEUpModel(tf.keras.Model, BaseTransformerModel):
    """Masked Autoencoder with Upsampling Decoder"""
    
    def __init__(self, config):
        tf.keras.Model.__init__(self)  # Initialize keras.Model first
        BaseTransformerModel.__init__(self, config)  # Then initialize our base class
        
        # Encoder settings
        self.enc_embed_dim = config['encoder']['embed_dim']
        self.enc_depth = config['encoder']['depth']
        self.enc_num_heads = config['encoder']['num_heads']
        self.enc_mlp_ratio = config['encoder']['mlp_ratio']
        self.enc_dropout = config['encoder']['dropout_rate']
        
        # Decoder settings
        self.dec_embed_dim = config['decoder']['embed_dim']
        self.dec_depth = config['decoder']['depth']
        self.dec_num_heads = config['decoder']['num_heads']
        self.dec_mlp_ratio = config['decoder']['mlp_ratio']
        self.dec_dropout = config['decoder']['dropout_rate']
        
        # Masking
        self.mask_ratio = config['training']['mask_ratio']
        
        # Create positional embedding layer instead of weight
        img_size = config['data']['img_size']
        patch_size = config.get('patch_size', 16)
        grid_size = img_size // patch_size
        num_patches = grid_size * grid_size
        
        self.pos_embed = layers.Embedding(
            input_dim=num_patches,
            output_dim=self.enc_embed_dim,
            name="position_embedding"
        )
    
    def build(self, input_shape):
        """Build the model layers - called automatically by Keras"""
        img_size = self.img_size
        grid_size = img_size // self.patch_size
        num_patches = grid_size * grid_size
        
        super().build(input_shape)
    
    def build_model(self):
        """Build and return the complete model"""
        img_size = self.img_size
        
        # Input
        inputs = layers.Input((img_size, img_size, 3))
        
        # Patchify and embed
        x = PatchEmbed(self.patch_size, self.enc_embed_dim)(inputs)
        
        # Add positional embedding
        positions = tf.range(0, tf.shape(x)[1])
        pos_encoding = self.pos_embed(positions)
        x = x + pos_encoding[tf.newaxis, :, :]
        
        # Generate center mask
        grid_size = img_size // self.patch_size
        num_patches = grid_size * grid_size
        mask_len = int(math.sqrt(1.0 - self.mask_ratio) * grid_size)
        start = (grid_size - mask_len) // 2
        visible_patches = tf.reshape(
            tf.range(num_patches), [grid_size, grid_size]
        )[start:start+mask_len, start:start+mask_len]
        visible_patches = tf.reshape(visible_patches, [-1])
        
        # Apply mask by gathering visible patches
        x = tf.gather(x, visible_patches, axis=1)
        
        # Encoder transformer blocks
        for _ in range(self.enc_depth):
            x = TransformerBlock(
                self.enc_embed_dim, 
                self.enc_num_heads,
                self.enc_mlp_ratio,
                self.enc_dropout
            )(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Project to decoder dimension
        x = layers.Dense(self.dec_embed_dim)(x)
        
        # Decoder transformer blocks
        for _ in range(self.dec_depth):
            x = TransformerBlock(
                self.dec_embed_dim,
                self.dec_num_heads,
                self.dec_mlp_ratio,
                self.dec_dropout
            )(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Predict patch pixels
        x = layers.Dense(self.patch_size * self.patch_size * 3)(x)
        
        # Reshape to spatial features
        x = layers.Reshape((mask_len, mask_len, self.patch_size * self.patch_size * 3))(x)
        
        # Fixed upsampling architecture
        x = layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Final convolution to get 3 channels
        outputs = layers.Conv2D(3, kernel_size=1, activation='sigmoid', padding='same')(x)
        
        # Final resize to ensure exact dimensions
        outputs = layers.Resizing(
            height=self.img_size,
            width=self.img_size,
            interpolation='bilinear'
        )(outputs)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="mae_up")
    
    def build_classifier(self, input_shape: Tuple[int, int, int], num_classes: int):
        """Build MAE-based classifier model (encoder only)"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # Patch embedding
        patches = self.create_patches(inputs)
        encoded_patches = layers.Dense(self.enc_embed_dim)(patches)
        
        # Add positional embedding
        num_patches = (self.img_size // self.patch_size) ** 2
        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=self.enc_embed_dim
        )(positions)
        encoded_patches = encoded_patches + pos_embedding
        
        # Encoder transformer blocks
        for _ in range(self.enc_depth):
            # Layer norm 1
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.enc_num_heads,
                key_dim=self.enc_embed_dim // self.enc_num_heads,
                dropout=0.1
            )(x1, x1)
            
            # Skip connection 1
            x2 = layers.Add()([attention_output, encoded_patches])
            
            # Layer norm 2
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            mlp_dim = int(self.enc_embed_dim * self.enc_mlp_ratio)
            x3 = layers.Dense(mlp_dim, activation='gelu')(x3)
            x3 = layers.Dropout(0.1)(x3)
            x3 = layers.Dense(self.enc_embed_dim)(x3)
            x3 = layers.Dropout(0.1)(x3)
            
            # Skip connection 2
            encoded_patches = layers.Add()([x3, x2])
        
        # Classification head
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.GlobalAveragePooling1D()(representation)
        representation = layers.Dropout(0.2)(representation)
        outputs = layers.Dense(num_classes, activation='softmax')(representation)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mae_up_classifier')
        return model
