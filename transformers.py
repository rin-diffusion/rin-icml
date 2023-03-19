"""Transformer."""

import math

import tensorflow as tf


def suffix_id(i):
  """Return suffix id for layer/variable name."""
  return '' if i == 0 else '_%d' % i


def get_shape(x):
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def get_angles(pos, i, dim):
  angle_rates = 1 / tf.pow(10000., tf.cast(2 * (i//2), tf.float32) / dim)
  return tf.cast(pos, tf.float32) * tf.cast(angle_rates, tf.float32)


def positional_encoding(coords, dim):
  """coords in (bsz, size), return (bsz, size, dim)."""
  angle_rads = get_angles(tf.expand_dims(coords, -1),
                          tf.range(dim)[tf.newaxis, tf.newaxis, :],
                          dim)

  # apply sin to even indices in the array; 2i
  angle_rads1 = tf.sin(angle_rads[:, :, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads2 = tf.cos(angle_rads[:, :, 1::2])

  pos_encoding = tf.concat([angle_rads1, angle_rads2], -1)

  return tf.cast(pos_encoding, dtype=tf.float32)


def get_1d_position_codes(seqlen, out_dim, normalization_max=6.2831852):
  """Get 2d positional embedding with sin/cos codes.

  Args:
    seqlen: a `int` specifying the length of the sequence.
    out_dim: a `int` specifying the output dimension of the encoding.
    normalization_max: normalize coordinates between [0, normalization_max].
      If None, raw coordinates from 0 to seqlen will be used.

  Returns:
    positional code of shape (1, seqlen, out_dim)
  """
  coords = tf.cast(tf.range(seqlen), tf.float32)
  if normalization_max is not None:
    coords = coords / (seqlen - 1) * normalization_max
  coords = positional_encoding(coords, out_dim)
  return coords


def get_2d_position_codes(height, width, out_dim, normalization_max=6.2831852):
  """Get 2d positional embedding with sin/cos codes.

  Args:
    height: a `int` specifying the height of the 2d image / feature map.
    width: a `int` specifying the width of the 2d image / feature map.
    out_dim: a `int` specifying the output dimension of the encoding.
      Must be divisible by 2.
    normalization_max: normalize coordinates between [0, normalization_max].
      If None, raw coordinates from 0 to height/width will be used.

  Returns:
    positional code of shape (1, height, width, out_dim)
  """
  y_coords = tf.cast(tf.range(height), tf.float32)
  if normalization_max is not None:
    y_coords = (
        y_coords / tf.cast(height - 1, dtype=tf.float32) * normalization_max)
  y_coords = positional_encoding(y_coords, out_dim//2)
  y_coords = tf.expand_dims(y_coords, 2)
  y_coords = tf.concat([y_coords, tf.zeros_like(y_coords)], -1)

  x_coords = tf.cast(tf.range(width), tf.float32)
  if normalization_max is not None:
    x_coords = (
        x_coords / tf.cast(width - 1, dtype=tf.float32) * normalization_max)
  x_coords = positional_encoding(x_coords, out_dim//2)
  x_coords = tf.expand_dims(x_coords, 1)
  x_coords = tf.concat([tf.zeros_like(x_coords), x_coords], -1)

  return y_coords + x_coords


def get_variable_initializer(name=None):
  if name is None:
    return tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)


def add_seq_pos_emb(self, pos_encoding, max_seq_len, dim,
                    name_prefix=None, initializer=None):
  """Add seq_pos_emb variable/tensor to model instance referenced by `self`."""
  if name_prefix is None:
    name_prefix = self.name
  if initializer is None:
    initializer = get_variable_initializer()
  if pos_encoding == 'learned':
    self.seq_pos_emb = self.add_weight(
        shape=(max_seq_len, dim), initializer=initializer,
        name='%s/seq_pos_embedding' % name_prefix)
  elif pos_encoding == 'sin_cos':
    sin_cos = get_1d_position_codes(
        max_seq_len, dim, normalization_max=6.2831852)
    self.seq_pos_emb = tf.reshape(sin_cos, [max_seq_len, dim])
  else:
    raise ValueError('Unknown pos encoding %s' % pos_encoding)


def add_vis_pos_emb(self,
                    pos_encoding,
                    n_rows,
                    n_cols,
                    dim,
                    name_prefix=None,
                    initializer=None,
                    return_only=False,
                    normalization_max=6.2831852):
  """Add vis_pos_emb variable/tensor to model instance referenced by `self`."""
  if name_prefix is None:
    name_prefix = self.name
  if initializer is None:
    initializer = get_variable_initializer()
  if pos_encoding == 'learned':
    vis_pos_emb = self.add_weight(
        shape=(n_rows * n_cols, dim), initializer=initializer,
        name='%s/vis_pos_embedding' % name_prefix)
  elif pos_encoding == 'sin_cos':
    if n_rows == 1 or n_cols == 1:
      sin_cos = get_1d_position_codes(
          n_rows * n_cols, dim, normalization_max=normalization_max)
    else:
      sin_cos = get_2d_position_codes(
          n_rows, n_cols, dim, normalization_max=normalization_max)
    vis_pos_emb = tf.reshape(sin_cos, [n_rows * n_cols, dim])
  else:
    raise ValueError('Unknown pos encoding %s' % pos_encoding)
  if not return_only:
    self.vis_pos_emb = vis_pos_emb
  return vis_pos_emb


def unfold(images, patch_size, patch_stride=None):
  if patch_stride is None:
    patch_stride = patch_size
  patches = tf.image.extract_patches(
      images,
      sizes=[1, patch_size, patch_size, 1],
      strides=[1, patch_stride, patch_stride, 1],
      rates=[1, 1, 1, 1],
      padding='VALID')
  return patches


class DropPath(tf.keras.layers.Layer):
  """For stochastic depth."""

  def __init__(self, drop_rate=0., **kwargs):
    """Initializes a drop path layer."""
    super(DropPath, self).__init__(**kwargs)
    self._drop_rate = drop_rate
    if self._drop_rate < 0 or self._drop_rate >= 1.0:
      raise ValueError('drop_rate {} is outside [0, 1)'.format(self._drop_rate))

  def call(self, x, training=False):
    """Performs a forward pass.

    Args:
      x: An input tensor of type tf.Tensor with shape [batch, height,
        width, channels].
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      The output tensor.
    """
    if self._drop_rate == 0. or not training:
      return x

    keep_rate = 1. - self._drop_rate
    xshape = tf.shape(x)
    drop_mask_shape = [xshape[0]] + [1] * (len(xshape) - 1)
    drop_mask = keep_rate + tf.random.uniform(drop_mask_shape, dtype=x.dtype)
    drop_mask = tf.math.divide(tf.floor(drop_mask), keep_rate)

    return x * drop_mask


class FeedForwardLayer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               dim_att,
               dim_mlp,
               drop_units=0.1,
               use_ln=False,
               ln_scale_shift=False,
               **kwargs):
    super(FeedForwardLayer, self).__init__(**kwargs)
    self.dense1 = tf.keras.layers.Dense(
        dim_mlp, activation=tf.nn.gelu, name='dense1')
    self.dropout = tf.keras.layers.Dropout(drop_units)
    self.dense2 = tf.keras.layers.Dense(dim_att, name='dense2')
    if use_ln:
      self.ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6,
          center=ln_scale_shift,
          scale=ln_scale_shift,
          name='mlp_ln')
    else:
      self.ln = lambda x: x

  def call(self, x, training):
    return self.dense2(self.dropout(self.ln(self.dense1(x)), training=training))


class MLP(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               num_layers,
               dim,
               mlp_ratio,
               drop_path=0.1,
               drop_units=0.,
               use_ffn_ln=False,
               ln_scale_shift=True,
               **kwargs):
    super(MLP, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.mlp_layers = [
        FeedForwardLayer(dim, dim * mlp_ratio, drop_units,
                         use_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
                         name='ffn' + suffix_id(i))
        for i in range(num_layers)
    ]
    self.layernorms = [
        tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            center=ln_scale_shift,
            scale=ln_scale_shift,
            name='ffn/ln' + suffix_id(i))
        for i in range(num_layers)
    ]
    self.dropp = DropPath(drop_path)

  def call(self, x, training, ret_list=False):
    x_list = [x]
    for i in range(self.num_layers):
      x_residual = self.mlp_layers[i](self.layernorms[i](x), training)
      x = x + self.dropp(x_residual, training)
      x_list.append(x)
    return (x, x_list) if ret_list else x


class TransformerEncoderLayer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               dim,
               mlp_ratio,
               num_heads,
               drop_path=0.1,
               drop_units=0.1,
               drop_att=0.,
               use_ffn_ln=False,
               ln_scale_shift=True,
               **kwargs):
    super(TransformerEncoderLayer, self).__init__(**kwargs)
    self.mha_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
        center=ln_scale_shift,
        scale=ln_scale_shift,
        name='mha/ln')
    self.mha = tf.keras.layers.MultiHeadAttention(
        num_heads, dim // num_heads, dropout=drop_att, name='mha')
    self.mlp = MLP(1, dim, mlp_ratio, drop_path, drop_units,
                   use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
                   name='mlp')
    self.dropp = DropPath(drop_path)

  def call(self, x, mask, training):
    # x shape (bsz, seq_len, dim_att), mask shape (bsz, seq_len, seq_len).
    x_ln = self.mha_ln(x)
    x_residual = self.mha(x_ln, x_ln, x_ln, mask, training=training)
    x = x + self.dropp(x_residual, training)
    x = self.mlp(x, training)
    return x


class TransformerEncoder(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               num_layers,
               dim,
               mlp_ratio,
               num_heads,
               drop_path=0.1,
               drop_units=0.1,
               drop_att=0.,
               use_ffn_ln=False,
               ln_scale_shift=True,
               **kwargs):
    super(TransformerEncoder, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.enc_layers = [
        TransformerEncoderLayer(  # pylint: disable=g-complex-comprehension
            dim, mlp_ratio, num_heads, drop_path, drop_units, drop_att,
            use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
            name='transformer_encoder' + suffix_id(i))
        for i in range(num_layers)
    ]

  def call(self, x, mask, training, ret_list=False):
    x_list = [x]
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, mask, training)
      x_list.append(x)
    return (x, x_list) if ret_list else x


class TransformerDecoderLayer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               dim,
               mlp_ratio,
               num_heads,
               drop_path=0.1,
               drop_units=0.1,
               drop_att=0.,
               dim_x_att=None,
               self_attention=True,
               cross_attention=True,
               use_mlp=True,
               use_enc_ln=False,
               use_ffn_ln=False,
               ln_scale_shift=True,
               **kwargs):
    super(TransformerDecoderLayer, self).__init__(**kwargs)
    self.self_attention = self_attention
    self.cross_attention = cross_attention
    self.use_mlp = use_mlp
    if self_attention:
      self.self_ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6,
          center=ln_scale_shift,
          scale=ln_scale_shift,
          name='self_mha/ln')
      self.self_mha = tf.keras.layers.MultiHeadAttention(
          num_heads, dim // num_heads, dropout=drop_att, name='self_mha')
    if cross_attention:
      self.cross_ln = tf.keras.layers.LayerNormalization(
          epsilon=1e-6,
          center=ln_scale_shift,
          scale=ln_scale_shift,
          name='cross_mha/ln')
      if use_enc_ln:
        self.enc_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            center=ln_scale_shift,
            scale=ln_scale_shift,
            name='cross_mha/enc_ln')
      else:
        self.enc_ln = lambda x: x
      dim_x_att = dim if dim_x_att is None else dim_x_att
      self.cross_mha = tf.keras.layers.MultiHeadAttention(
          num_heads, dim_x_att // num_heads, dropout=drop_att, name='cross_mha')
    if use_mlp:
      self.mlp = MLP(1, dim, mlp_ratio, drop_path, drop_units,
                     use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
                     name='mlp')
    self.dropp = DropPath(drop_path)

  def call(self, x, enc, cache, mask_self, mask_cross, training):
    """x in (bsz, seq, d), enc in (bsz, seq', d)."""
    x_for_cache = []
    if self.self_attention:
      x_for_cache = x_ln = kv_ln = self.self_ln(x)
      if cache is not None:  # Augment kv_ln with cache in (bsz, c_size, d).
        q_size, k_size = tf.shape(x)[1], tf.shape(cache)[1]
        mask_self = tf.concat([tf.ones([1, 1, q_size, k_size]), mask_self], -1)
        kv_ln = tf.concat([cache, x_ln], axis=1)
      x_res = self.self_mha(x_ln, kv_ln, kv_ln, mask_self, training=training)
      x = x + self.dropp(x_res, training)
    if self.cross_attention:
      x_ln = self.cross_ln(x)
      enc = self.enc_ln(enc)
      x_res = self.cross_mha(x_ln, enc, enc, mask_cross, training=training)
      x = x + self.dropp(x_res, training)
    if self.use_mlp:
      x = self.mlp(x, training)
    return x, x_for_cache


class TransformerDecoder(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               num_layers,
               dim,
               mlp_ratio,
               num_heads,
               drop_path=0.1,
               drop_units=0.1,
               drop_att=0.,
               dim_x_att=None,
               self_attention=True,
               cross_attention=True,
               use_mlp=True,
               use_enc_ln=False,
               use_ffn_ln=False,
               ln_scale_shift=True,
               **kwargs):
    super(TransformerDecoder, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.dec_layers = [
        TransformerDecoderLayer(  # pylint: disable=g-complex-comprehension
            dim,
            mlp_ratio,
            num_heads,
            drop_path,
            drop_units,
            drop_att,
            dim_x_att=dim_x_att,
            self_attention=self_attention,
            cross_attention=cross_attention,
            use_mlp=use_mlp,
            use_enc_ln=use_enc_ln,
            use_ffn_ln=use_ffn_ln,
            ln_scale_shift=ln_scale_shift,
            name='transformer_decoder_layer' + suffix_id(i))
        for i in range(num_layers)
    ]

  def call(self, x, enc, caches, mask_self, mask_cross, training):
    """x in (bsz, seq, d), enc in (bsz, seq', d)."""
    presents = []
    for i in range(self.num_layers):
      cache = None if caches is None else caches[i]
      x, x_for_cache = self.dec_layers[i](
          x, enc, cache, mask_self, mask_cross, training)
      presents.append(x_for_cache)

    return x, tf.stack(presents)

