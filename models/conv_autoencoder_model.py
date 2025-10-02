import tensorflow as tf
from tensorflow.keras import layers
from .base_model import BaseTransformerModel

class EncoderBlock(layers.Layer):
    """Single encoder block with conv, optional BN, activation, and pooling"""
    def __init__(self, filters, kernel_size, use_batch_norm=True, dropout_rate=0.1):
        super().__init__()
        self.conv = layers.Conv2D(filters, kernel_size, padding="same")
        self.batch_norm = layers.BatchNormalization() if use_batch_norm else None
        self.activation = layers.Activation("relu")
        self.pool = layers.MaxPool2D()
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, x, training=False):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        skip = x  # Save for skip connection
        x = self.pool(x)
        return x, skip

class DecoderBlock(layers.Layer):
    """Single decoder block with upsampling, conv, optional BN and activation"""
    def __init__(self, filters, kernel_size, use_batch_norm=True, dropout_rate=0.1, upsampling_type="transpose"):
        super().__init__()
        self.upsampling_type = upsampling_type
        if upsampling_type == "transpose":
            self.up = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding="same")
        else:
            self.up = layers.UpSampling2D()
            self.conv_up = layers.Conv2D(filters, kernel_size, padding="same")
            
        self.conv = layers.Conv2D(filters, kernel_size, padding="same")
        self.batch_norm = layers.BatchNormalization() if use_batch_norm else None
        self.activation = layers.Activation("relu")
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, x, skip=None, training=False):
        if self.upsampling_type == "transpose":
            x = self.up(x)
        else:
            x = self.up(x)
            x = self.conv_up(x)
            
        if skip is not None:
            x = layers.Concatenate()([x, skip])
            
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        return x

class ConvolutionalAutoencoderModel(BaseTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        # Model architecture parameters
        self.initial_filters = config.get('initial_filters', 32)
        self.filter_growth_rate = config.get('filter_growth_rate', 2)
        self.num_encoder_blocks = config.get('num_encoder_blocks', 3)
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.use_skip_connections = config.get('use_skip_connections', True)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.kernel_size = config.get('kernel_size', 3)
        self.upsampling_type = config.get('upsampling_type', 'transpose')
        self.bottleneck_filters = config.get('bottleneck_filters', 256)
        
    def build_model(self):
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # Encoder
        x = inputs
        skip_connections = []
        filters = self.initial_filters
        
        # Encoder blocks
        for _ in range(self.num_encoder_blocks):
            encoder_block = EncoderBlock(
                filters=filters,
                kernel_size=self.kernel_size,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate
            )
            x, skip = encoder_block(x)
            if self.use_skip_connections:
                skip_connections.append(skip)
            filters *= self.filter_growth_rate
            
        # Bottleneck
        x = layers.Conv2D(self.bottleneck_filters, self.kernel_size, padding="same", activation="relu")(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Decoder blocks
        for skip in reversed(skip_connections):
            filters //= self.filter_growth_rate
            decoder_block = DecoderBlock(
                filters=filters,
                kernel_size=self.kernel_size,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
                upsampling_type=self.upsampling_type
            )
            x = decoder_block(x, skip)
            
        # Final reconstruction
        outputs = layers.Conv2D(3, 1, activation="sigmoid")(x)
        
        model = tf.keras.Model(inputs, outputs, name="conv_autoencoder")
        
        # Print model architecture details
        print("\nConvolutional Autoencoder Architecture:")
        print(f"Initial filters: {self.initial_filters}")
        print(f"Number of encoder blocks: {self.num_encoder_blocks}")
        print(f"Using batch normalization: {self.use_batch_norm}")
        print(f"Using skip connections: {self.use_skip_connections}")
        print(f"Upsampling type: {self.upsampling_type}")
        
        return model