import tensorflow as tf
from tensorflow.keras import layers
from .base_model import BaseTransformerModel

class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        
        self.q_dense = layers.Dense(embed_dim)
        self.k_dense = layers.Dense(embed_dim)
        self.v_dense = layers.Dense(embed_dim)
        self.out_dense = layers.Dense(embed_dim)
        
    def call(self, x):
        batch_size, seq_len, embed_dim = tf.unstack(tf.shape(x))
        
        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)
        
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        attention = tf.matmul(q, k, transpose_b=True)
        attention = attention / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attention = tf.nn.softmax(attention, axis=-1)
        
        out = tf.matmul(attention, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, seq_len, self.num_heads * self.head_dim))
        
        return self.out_dense(out)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate)
        ])
        
    def call(self, x, training=False):
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x)
        x = x + attn_out
        
        # MLP with residual connection
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x, training=training)
        x = x + mlp_out
        
        return x

class PatchExtractor(layers.Layer):
    """Extract patches from images"""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        img_size = images.shape[1]  # Assuming square images
        num_patches = (img_size // self.patch_size) ** 2
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])
        return patches

class PatchReconstructor(layers.Layer):
    """Reconstruct image from patches"""
    def __init__(self, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        
    def call(self, patches):
        batch_size = tf.shape(patches)[0]
        h = w = self.img_size // self.patch_size
        patches = tf.reshape(patches, [batch_size, h, w, self.patch_size, self.patch_size, 3])
        patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
        images = tf.reshape(patches, [batch_size, self.img_size, self.img_size, 3])
        return images

class ViTModel(BaseTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
    def build_model(self):
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # Convert to patches and embed using custom layers
        patches = PatchExtractor(self.patch_size)(inputs)
        
        # Linear projection
        patch_embed = layers.Dense(self.embed_dim)(patches)
        num_patches = (self.img_size // self.patch_size) ** 2
        
        # Add positional embeddings layer
        class PositionalEmbedding(layers.Layer):
            def __init__(self, num_patches, embed_dim):
                super().__init__()
                self.num_patches = num_patches
                self.embed_dim = embed_dim
                
            def build(self, input_shape):
                self.pos_embed = self.add_weight(
                    shape=(1, self.num_patches, self.embed_dim),
                    initializer="random_normal",
                    name="pos_embed"
                )
                
            def call(self, x):
                return x + self.pos_embed
        
        x = PositionalEmbedding(num_patches, self.embed_dim)(patch_embed)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                self.embed_dim, self.num_heads, self.mlp_ratio, self.dropout_rate
            )(x)
        
        # Output projection to patch size
        patch_dim = self.patch_size * self.patch_size * 3
        x = layers.LayerNormalization()(x)
        x = layers.Dense(patch_dim)(x)
        
        # Reshape to image using custom layer
        outputs = PatchReconstructor(self.patch_size, self.img_size)(x)
        outputs = layers.Activation('sigmoid')(outputs)
        
        return tf.keras.Model(inputs, outputs, name="ViT_Reconstruction")
