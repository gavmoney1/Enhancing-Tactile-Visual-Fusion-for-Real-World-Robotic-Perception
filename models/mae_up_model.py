
import tensorflow as tf
from tensorflow.keras import layers
from .base_model import BaseTransformerModel
import math


class PatchEmbed(layers.Layer):
    def __init__(self, patch_size, embed_dim, name=None):
        super().__init__(name=name)
        self.patch_size = patch_size
        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="patch_proj"
        )

    def call(self, x):
        # x: [B, H, W, 3]
        x = self.proj(x)  # [B, H/ps, W/ps, embed_dim]
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]
        x = tf.reshape(x, [B, H * W, C])  # [B, N, C]
        return x


class PatchReassemble(layers.Layer):
    def __init__(self, img_size, patch_size, name=None):
        super().__init__(name=name)
        self.img_size = img_size
        self.patch_size = patch_size

    def call(self, patch_tokens):
        # patch_tokens: [B, N, ps*ps*3]
        B = tf.shape(patch_tokens)[0]
        ps = self.patch_size
        img = self.img_size
        grid = img // ps  # patches per side

        x = tf.reshape(
            patch_tokens,
            [B, grid, grid, ps, ps, 3]
        )  # [B, grid, grid, ps, ps, 3]

        # reorder to [B, grid*ps, grid*ps, 3]
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, grid * ps, grid * ps, 3])
        return x


class MLP(layers.Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0, name=None):
        super().__init__(name=name)
        self.fc1 = layers.Dense(hidden_dim, activation='gelu')
        self.drop1 = layers.Dropout(dropout)
        self.fc2 = layers.Dense(dim)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x


class TransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, name=None):
        super().__init__(name=name)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout,
            name="mha"
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def call(self, x, training=False):
        # x: [B, N, dim]
        x = x + self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            training=training
        )
        x = x + self.mlp(self.norm2(x), training=training)
        return x


class MAEUpModel(tf.keras.Model, BaseTransformerModel):
    def __init__(self, config):
        tf.keras.Model.__init__(self)
        BaseTransformerModel.__init__(self, config)

        self.patch_size   = config.get('patch_size', 16)
        self.img_size     = config['data']['img_size']
        self.mask_ratio   = config['training']['mask_ratio']  # e.g. 0.9

        # encoder cfg
        self.enc_embed_dim   = config['encoder']['embed_dim']
        self.enc_depth       = config['encoder']['depth']
        self.enc_num_heads   = config['encoder']['num_heads']
        self.enc_mlp_ratio   = config['encoder']['mlp_ratio']
        self.enc_dropout     = config['encoder']['dropout_rate']

        # decoder cfg
        self.dec_embed_dim   = config['decoder']['embed_dim']
        self.dec_depth       = config['decoder']['depth']
        self.dec_num_heads   = config['decoder']['num_heads']
        self.dec_mlp_ratio   = config['decoder']['mlp_ratio']
        self.dec_dropout     = config['decoder']['dropout_rate']

        grid_size    = self.img_size // self.patch_size
        num_patches  = grid_size * grid_size

        # position embeddings
        self.pos_embed_enc = layers.Embedding(
            input_dim=num_patches,
            output_dim=self.enc_embed_dim,
            name="pos_embed_enc"
        )
        self.pos_embed_dec = layers.Embedding(
            input_dim=num_patches,
            output_dim=self.dec_embed_dim,
            name="pos_embed_dec"
        )

        # patchify + unpatchify
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.enc_embed_dim,
            name="patch_embed"
        )
        self.reassemble = PatchReassemble(
            img_size=self.img_size,
            patch_size=self.patch_size,
            name="reassemble"
        )

        # mask token for decoder
        self.mask_token = self.add_weight(
            name="mask_token",
            shape=(1, 1, self.dec_embed_dim),
            initializer="random_normal",
            trainable=True,
        )

        # encoder transformer
        self.encoder_blocks = [
            TransformerBlock(
                dim=self.enc_embed_dim,
                num_heads=self.enc_num_heads,
                mlp_ratio=self.enc_mlp_ratio,
                dropout=self.enc_dropout,
                name=f"enc_block_{i}"
            )
            for i in range(self.enc_depth)
        ]
        self.enc_norm = layers.LayerNormalization(epsilon=1e-6, name="enc_norm")

        # proj to decoder dim
        self.enc_to_dec = layers.Dense(self.dec_embed_dim, name="enc_to_dec")

        # decoder transformer
        self.decoder_blocks = [
            TransformerBlock(
                dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=self.dec_mlp_ratio,
                dropout=self.dec_dropout,
                name=f"dec_block_{i}"
            )
            for i in range(self.dec_depth)
        ]
        self.dec_norm = layers.LayerNormalization(epsilon=1e-6, name="dec_norm")

        # final per-patch pixel head
        self.patch_pred = layers.Dense(
            units=self.patch_size * self.patch_size * 3,
            name="patch_pixel_head"
        )

    def call(self, inputs, training=False):
        """
        inputs: [B, H, W, 3] in [0,1]
        returns: [B, H, W, 3] reconstructed
        """

        B = tf.shape(inputs)[0]

        # === (A) Patchify ===
        # [B, N, enc_dim]
        patch_tokens = self.patch_embed(inputs)
        N = tf.shape(patch_tokens)[1]

        # === (B) Pos embed for encoder tokens ===
        patch_ids = tf.range(N, dtype=tf.int32)              # [N]
        pos_enc   = self.pos_embed_enc(patch_ids)[tf.newaxis, :, :]  # [1, N, enc_dim]
        tokens_with_pos = patch_tokens + pos_enc             # [B, N, enc_dim]

        # === (C) Compute visible patch indices (center square) in pure TF
        grid_size = self.img_size // self.patch_size  # scalar python int
        # how many patches per side to KEEP
        keep_frac = math.sqrt(1.0 - float(self.mask_ratio))  # python float
        keep_side = max(1, int(round(keep_frac * grid_size)))  # python int

        start = (grid_size - keep_side) // 2
        end   = start + keep_side

        # Build a [grid_size, grid_size] grid of indices [0..N-1]
        full_grid = tf.reshape(tf.range(grid_size * grid_size, dtype=tf.int32),
                               [grid_size, grid_size])               # [G,G]
        center_grid = full_grid[start:end, start:end]                 # [keep_side,keep_side]
        visible_idx = tf.reshape(center_grid, [-1])                   # [V]

        # one-hot visibility over N
        one_hot_vis = tf.one_hot(visible_idx, depth=N)                # [V, N]
        vis_mask = tf.reduce_max(one_hot_vis, axis=0)                 # [N]
        vis_mask = tf.reshape(vis_mask, [1, N, 1])                    # [1,N,1]
        vis_mask = tf.cast(vis_mask, tf.float32)                      # float

        # === (D) Gather visible tokens for encoder
        vis_tokens = tf.gather(tokens_with_pos, visible_idx, axis=1)  # [B, V, enc_dim]

        # === (E) Encode visible subset
        x = vis_tokens
        for blk in self.encoder_blocks:
            x = blk(x, training=training)
        x = self.enc_norm(x)                           # [B, V, enc_dim]

        # project to decoder dim
        x = self.enc_to_dec(x)                         # [B, V, dec_dim]

        # === (F) Expand encoder output back to full length N using einsum
        # one_hot_vis: [V, N]
        # x:           [B, V, dec_dim]
        # -> full_vis_tokens: [B, N, dec_dim]
        full_vis_tokens = tf.einsum('vn,bvd->bnd', one_hot_vis, x)

        # mask tokens for everywhere
        mask_tokens_full = tf.broadcast_to(
            self.mask_token,                # [1,1,dec_dim]
            [B, N, self.dec_embed_dim]      # [B,N,dec_dim]
        )

        vis_mask_b = tf.broadcast_to(vis_mask, [B, N, 1])  # [B,N,1]

        # choose encoded token where visible, mask_token otherwise
        dec_tokens_list = (
            full_vis_tokens * vis_mask_b +
            mask_tokens_full * (1.0 - vis_mask_b)
        )  # [B, N, dec_dim]

        # === (G) Add decoder positional embeddings ===
        pos_dec = self.pos_embed_dec(patch_ids)[tf.newaxis, :, :]  # [1,N,dec_dim]
        dec_tokens = dec_tokens_list + pos_dec                     # [B,N,dec_dim]

        # === (H) Decoder transformer on ALL tokens ===
        y = dec_tokens
        for blk in self.decoder_blocks:
            y = blk(y, training=training)
        y = self.dec_norm(y)  # [B,N,dec_dim]

        # === (I) Predict pixels per patch ===
        patch_pixels = self.patch_pred(y)  # [B,N,ps*ps*3]

        # === (J) Reassemble patches -> image ===
        img_out = self.reassemble(patch_pixels)  # [B,H,W,3]

        # === (K) Clip to [0,1] range
        img_out = tf.clip_by_value(img_out, 0.0, 1.0)

        return img_out

    def build_model(self):
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3), name="input_image")
        outputs = self(inputs, training=False)  # <-- call the model directly
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="mae_up")