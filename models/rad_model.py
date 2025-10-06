import math
import tensorflow as tf
from tensorflow.keras import layers
from .base_model import BaseTransformerModel

def sinusoidal_timestep_embedding(timesteps, dim):
	"""
	Create positional/timestep embeddings.
	Enables network to distinguish between noise levels

	Args:
		timesteps: shape [batch] int tensor of timesteps
		dim: embedding dimension

	Returns:
		Tensor of shape [batch, dim]
	"""
	half = dim // 2
	freqs = tf.exp(
		-math.log(10000) * tf.cast(tf.range(0, half), tf.float32) / float(half)
	)
	args = tf.cast(tf.expand_dims(timesteps, -1), tf.float32) * tf.expand_dims(freqs, 0)
	emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
	if dim % 2:
		emb = tf.pad(emb, [[0, 0], [0, 1]])
	return emb


class ConvBlock(layers.Layer):
	"""
	Standard residual-style convolution block for feature extraction
	"""
	def __init__(self, out_ch, kernel_size=3, activation=layers.Activation('swish')):
		super().__init__()
		self.conv1 = layers.Conv2D(out_ch, kernel_size, padding='same')
		self.conv2 = layers.Conv2D(out_ch, kernel_size, padding='same')
		self.norm1 = layers.BatchNormalization()
		self.norm2 = layers.BatchNormalization()
		self.act = activation

	def call(self, x, training=False):
		x = self.conv1(x)
		x = self.norm1(x, training=training)
		x = self.act(x)
		x = self.conv2(x)
		x = self.norm2(x, training=training)
		x = self.act(x)
		return x


class Downsample(layers.Layer):
	"""
	Reduces spatial resolution by half
	Used for encoder
	"""
	def __init__(self, out_ch):
		super().__init__()
		self.pool = layers.AveragePooling2D(pool_size=2)
		self.conv = layers.Conv2D(out_ch, 1, padding='same')

	def call(self, x):
		x = self.pool(x)
		return self.conv(x)


class Upsample(layers.Layer):
	"""
	Doubles spatial resolution
	Reverse of Downsample
	Used for decoder
	"""
	def __init__(self, out_ch):
		super().__init__()
		self.conv = layers.Conv2D(out_ch, 1, padding='same')

	def call(self, x):
		x = tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='nearest')
		return self.conv(x)


class RegionAwareUNet(tf.keras.Model):
	"""
	Combines U-Net structure, timestep conditioning, and region awareness.
	Compact UNet that conditions on a binary region mask and timestep.

	Inputs expected: concatenation of image and mask along channels: [B,H,W,C+1]
	where image is typically 3 channels and mask is 1 channel.
	"""

	def __init__(self, base_channels=64, channel_mults=(1, 2, 4), time_emb_dim=128):
		super().__init__()
		self.time_mlp = tf.keras.Sequential([
			layers.Dense(time_emb_dim, activation='swish'),
			layers.Dense(time_emb_dim),
		])

		self.in_conv = layers.Conv2D(base_channels, 3, padding='same')

		# initialize encoder 

		# initialize decoder

    # call function

        # timestep embedding
		
        # encoder
		
        # decoder
		
# RADModel class

    # build_model function


	



