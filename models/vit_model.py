import tensorflow as tf
from tensorflow.keras import layers
from .base_model import BaseTransformerModel

class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        
        self.q_dense = layers.Dense(embed_dim)
        self.k_dense = layers.Dense(embed_dim)
        self.v_dense = layers.Dense(embed_dim)
        self.out_dense = layers.Dense(embed_dim)
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x, training=False):
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
        attention = self.dropout(attention, training=training)
        
        out = tf.matmul(attention, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, seq_len, self.num_heads * self.head_dim))
        
        return self.out_dense(out)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
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
        attn_out = self.attention(norm_x, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        x = x + attn_out
        
        # MLP with residual connection
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x, training=training)
        mlp_out = self.dropout2(mlp_out, training=training)
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
        img_size = images.shape[1]
        num_patches = (img_size // self.patch_size) ** 2
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])
        return patches

class PositionalEmbedding(layers.Layer):
    """Learnable positional embeddings"""
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            shape=(1, self.num_patches, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="pos_embed"
        )
        
    def call(self, x):
        return x + self.pos_embed


class DecoderBlock(layers.Layer):
    """Decoder block with upsampling and refinement"""
    def __init__(self, filters, kernel_size=3, dropout=0.1):
        super().__init__()
        self.upsample = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same')
        self.norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same')
        self.norm2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x, training=False):
        x = self.upsample(x)
        x = self.norm1(x, training=training)
        x = tf.nn.gelu(x)
        
        x = self.conv1(x)
        x = self.norm2(x, training=training)
        x = tf.nn.gelu(x)
        x = self.dropout(x, training=training)
        
        x = self.conv2(x)
        x = tf.nn.gelu(x)
        
        return x


class ViTModel(BaseTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.decoder_filters = config.get('decoder_filters', [256, 128, 64, 32])
        
    def build_model(self):
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))

        # Patch embeddings (encoder)
        # Extract patches
        patches = PatchExtractor(self.patch_size)(inputs)
        
        # Linear projection to embed_dim
        patch_embed = layers.Dense(self.embed_dim, name='patch_embedding')(patches)
        
        # Add positional embeddings
        num_patches = (self.img_size // self.patch_size) ** 2
        x = PositionalEmbedding(num_patches, self.embed_dim)(patch_embed)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Transformer block
        for i in range(self.num_layers):
            x = TransformerBlock(
                self.embed_dim, self.num_heads, self.mlp_ratio, self.dropout_rate
            )(x)
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Reshape to spatial to prepare for decoder
        # Calculate spatial dimensions after patching
        h = w = self.img_size // self.patch_size
        
        # Reshape from (batch, num_patches, embed_dim) to (batch, h, w, embed_dim)
        x = layers.Reshape((h, w, self.embed_dim))(x)
        
        # Decoder
        # Initial conv to adjust channels
        x = layers.Conv2D(self.decoder_filters[0], 3, padding='same', activation='gelu')(x)
        x = layers.BatchNormalization()(x)
        
        # Progressive upsampling with decoder blocks
        for i, filters in enumerate(self.decoder_filters):
            x = DecoderBlock(filters, kernel_size=3, dropout=self.dropout_rate)(x)
            print(f"After decoder block {i+1}: shape = (batch, {h * (2**(i+1))}, {h * (2**(i+1))}, {filters})")
        
        # Final refinement
        x = layers.Conv2D(32, 3, padding='same', activation='gelu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(16, 3, padding='same', activation='gelu')(x)
        
        # Output layer
        outputs = layers.Conv2D(3, 1, padding='same', activation='sigmoid')(x)
        
        return tf.keras.Model(inputs, outputs, name="ViT_Reconstruction")