import tensorflow as tf
from tensorflow.keras import layers
from .base_model import BaseTransformerModel

class WindowPartition(layers.Layer):
    """Partition input into windows"""
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        
    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        x = tf.reshape(x, [B, H//self.window_size, self.window_size, 
                          W//self.window_size, self.window_size, C])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, self.window_size * self.window_size, C])
        return x

class WindowReverse(layers.Layer):
    """Reverse window partition"""
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        
    def call(self, x, H, W):
        B = tf.shape(x)[0] // ((H // self.window_size) * (W // self.window_size))
        C = tf.shape(x)[2]
        
        x = tf.reshape(x, [B, H//self.window_size, W//self.window_size, 
                          self.window_size, self.window_size, C])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H, W, C])
        return x

class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = layers.Dense(dim * 3, use_bias=False)
        self.attn_dropout = layers.Dropout(dropout)
        self.proj = layers.Dense(dim)
        self.proj_dropout = layers.Dropout(dropout)
        self.window_partition = WindowPartition(window_size)
        self.window_reverse = WindowReverse(window_size)
        
    def call(self, x, training=False):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Window partition
        x = self.window_partition(x)
        
        # Self-attention within windows
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [-1, self.window_size * self.window_size, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn)
        attn = self.attn_dropout(attn, training=training)
        
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, self.window_size * self.window_size, C])
        x = self.proj(x)
        x = self.proj_dropout(x, training=training)
        
        # Reverse window partition
        x = self.window_reverse(x, H, W)
        
        return x

class CyclicShift(layers.Layer):
    """Apply cyclic shift to tensor"""
    def __init__(self, shift_size):
        super().__init__()
        self.shift_size = shift_size
        
    def call(self, x):
        if self.shift_size > 0:
            return tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        return x

class CyclicShiftReverse(layers.Layer):
    """Reverse cyclic shift"""
    def __init__(self, shift_size):
        super().__init__()
        self.shift_size = shift_size
        
    def call(self, x):
        if self.shift_size > 0:
            return tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        return x

class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = WindowAttention(dim, window_size, num_heads, dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)
        
        # Cyclic shift layers
        self.cyclic_shift = CyclicShift(shift_size)
        self.cyclic_shift_reverse = CyclicShiftReverse(shift_size)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_hidden_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])
        
    def call(self, x, training=False):
        shortcut = x
        x = self.norm1(x)
        
        # Cyclic shift if needed
        x = self.cyclic_shift(x)
        
        # Window attention
        x = self.attn(x, training=training)
        
        # Reverse cyclic shift
        x = self.cyclic_shift_reverse(x)
        
        # Residual connection
        x = shortcut + self.dropout(x, training=training)
        
        # MLP
        x = x + self.mlp(self.norm2(x), training=training)
        
        return x


class PatchMerging(layers.Layer):
    """Downsample layer between stages (reduces spatial, increases channels)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, x):
        # x: (B, H, W, C)
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Concatenate 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # (B, H/2, W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2C)
        
        return x


class PatchExpanding(layers.Layer):
    """Upsample layer in decoder (increases spatial, reduces channels)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.expand = layers.Dense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, x):
        # x: (B, H, W, C)
        x = self.expand(x)  # (B, H, W, 2C)
        x = self.norm(x)
        
        # Rearrange to upsample spatially
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B, H, W, 2, 2, C // 4])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H * 2, W * 2, C // 4])
        
        return x


class SwinModel(BaseTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.img_size = config.get('img_size', 224)
        self.patch_size = config.get('patch_size', 4)
        self.embed_dim = config.get('embed_dim', 96)
        self.window_size = config.get('window_size', 7)
        self.depths = config.get('depths', [2, 2, 6, 2])
        self.num_heads_list = config.get('num_heads_list', [3, 6, 12, 24])
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
    def build_model(self):
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # Patch Embeddings
        x = layers.Conv2D(self.embed_dim, kernel_size=self.patch_size, 
                         strides=self.patch_size, padding='same')(inputs)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        # Shape: (B, 56, 56, 96) for 224x224 input
        
        # Encoder - Hierarchical stages with skip connections
        skip_connections = []
        num_stages = len(self.depths)
        
        for stage_idx in range(num_stages):
            # Get parameters for this stage
            depth = self.depths[stage_idx]
            num_heads = self.num_heads_list[stage_idx]
            dim = self.embed_dim * (2 ** stage_idx)
            
            # Swin Transformer blocks for this stage
            for block_idx in range(depth):
                shift_size = 0 if (block_idx % 2 == 0) else self.window_size // 2
                x = SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout_rate
                )(x)
            
            # Save skip connection
            skip_connections.append(x)
            
            # Downsample between stages (except last stage)
            if stage_idx < num_stages - 1:
                x = PatchMerging(dim)(x)
        
        # Decoder - Symmetric upsampling with skip connections
        for stage_idx in range(num_stages - 1, 0, -1):
            # Get parameters for this stage
            depth = self.depths[stage_idx - 1]
            num_heads = self.num_heads_list[stage_idx - 1]
            dim = self.embed_dim * (2 ** (stage_idx - 1))
            
            # Upsample
            x = PatchExpanding(x.shape[-1])(x)
            
            # Concatenate with skip connection
            skip = skip_connections[stage_idx - 1]
            x = layers.Concatenate()([x, skip])
            
            # Conv to adjust channels
            x = layers.Conv2D(dim, 1, padding='same')(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Swin Transformer blocks in decoder
            for block_idx in range(depth):
                shift_size = 0 if (block_idx % 2 == 0) else self.window_size // 2
                x = SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout_rate
                )(x)
        
        # Upsample from 56x56 to 224x224
        x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, 
                                   padding='same', activation='gelu')(x)
        x = layers.BatchNormalization()(x)

        
        x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, 
                                   padding='same', activation='gelu')(x)
        x = layers.BatchNormalization()(x)
        
        # Final refinement
        x = layers.Conv2D(32, 3, padding='same', activation='gelu')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Conv2D(3, kernel_size=1, activation='sigmoid')(x)
        
        return tf.keras.Model(inputs, outputs, name="Swin_Reconstruction")