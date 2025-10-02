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
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = layers.Dense(dim * 3, use_bias=False)
        self.proj = layers.Dense(dim)
        self.window_partition = WindowPartition(window_size)
        self.window_reverse = WindowReverse(window_size)
        
    def call(self, x):
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
        
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, self.window_size * self.window_size, C])
        x = self.proj(x)
        
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
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = layers.LayerNormalization()
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = layers.LayerNormalization()
        
        # Cyclic shift layers
        self.cyclic_shift = CyclicShift(shift_size)
        self.cyclic_shift_reverse = CyclicShiftReverse(shift_size)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_hidden_dim, activation='gelu'),
            layers.Dense(dim)
        ])
        
    def call(self, x):
        shortcut = x
        x = self.norm1(x)
        
        # Cyclic shift if needed
        x = self.cyclic_shift(x)
        
        # Window attention
        x = self.attn(x)
        
        # Reverse cyclic shift
        x = self.cyclic_shift_reverse(x)
        
        # Residual connection
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

class SwinModel(BaseTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.window_size = config.get('window_size', 7)
        self.depths = config.get('depths', [2, 2, 6, 2])
        
    def build_model(self):
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # Patch embedding
        x = layers.Conv2D(self.embed_dim, kernel_size=4, strides=4)(inputs)
        x = layers.LayerNormalization()(x)
        
        # Swin Transformer stages
        for stage_idx, depth in enumerate(self.depths):
            for block_idx in range(depth):
                shift_size = 0 if (block_idx % 2 == 0) else self.window_size // 2
                x = SwinTransformerBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    mlp_ratio=self.mlp_ratio
                )(x)
        
        # Reconstruction head
        x = layers.LayerNormalization()(x)
        x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        outputs = layers.Conv2D(3, kernel_size=1, activation='sigmoid')(x)
        
        return tf.keras.Model(inputs, outputs, name="Swin_Reconstruction")
