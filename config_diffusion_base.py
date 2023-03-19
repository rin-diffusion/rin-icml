"""Base image/video diffusion config."""

# pylint: disable=invalid-name,line-too-long

from configs.config_base import architecture_config_map
from configs.config_base import D


def get_config(config_str=None):
  """config_str can be none or something that specifies meta-hyperparam."""
  if config_str:
    data_name, arch_variant = config_str.split(',')
  else:
    data_name = 'cifar10'
    arch_variant = 'transunet'
    arch_variant = 'tape'

  architecture_config_map.update({
      'tape': D(
          arch_name='tape',
          num_layers='4,4,4,4',
          latent_slots=32,
          latent_dim=256,
          latent_mlp_ratio=4,
          latent_num_heads=4,
          tape_dim=128,
          tape_mlp_ratio=2,
          rw_num_heads=1,
          conv_kernel_size=0,
          conv_drop_units=0.,
          drop_path=0.,
          drop_units=0.,
          drop_att=0.,
          drop_sc=0.,
          time_on_latent=False,
          cond_on_latent=False,
          cond_tape_writable=False,
          latent_pos_encoding='learned',
          tape_pos_encoding='learned',
      ),
  })

  dataset_config_map = {
      'mnist': D(
          name='object_recognition',
          tfds_name='mnist',
          train_split='train',
          eval_split='test',
          num_classes=10,
          image_size=28,
          batch_duplicates=1,
          cache_dataset=True,
          cropping='none',
          flipping='none',
          ),
      'cifar10': D(
          name='object_recognition',
          tfds_name='cifar10',
          train_split='train',
          eval_split='test',
          num_classes=10,
          image_size=32,
          batch_duplicates=1,
          cache_dataset=True,
          cropping='none',
          flipping='left_right',
          ),
      'imagenet2012': D(
          name='object_recognition',
          tfds_name='imagenet2012',
          train_split='train',
          eval_split='validation',
          num_classes=1000,
          image_size=64,
          batch_duplicates=1,
          cache_dataset=True,
          cropping='center',
          flipping='left_right',
          ),
      'kinetics600': D(
          name='kinetics600',
          tfds_name='kinetics600', # for eval config
          train_split='train',
          eval_split='test',
          num_classes=600,
          image_size=64,
          batch_duplicates=1,
          cache_dataset=False,
          cropping='none',
          flipping='none',
          seq_len=16,
          ),
  }

  task = D(
      name='image_generation',
      weight=1.,
  )
  task_d_list = [task]
  dataset_list = [dataset_config_map[data_name]]

  config = D(
      dataset=dataset_list[0],
      datasets=dataset_list,

      task=task_d_list[0],
      tasks=task_d_list,

      model=D(
          name='image_diffusion_model',
          train_schedule='cosine',
          infer_schedule='cosine',
          pred_type='eps',
          loss_type='eps',
          infer_iterations=100,
          td=0.,
          x0_clip='auto',
          b_scale=1.0,
          normalize_noisy_input=False,
          time_scaling=1000,
          pretrained_ckpt='',
          sampler_name='ddpm',
          conditional='class',
          self_cond='latent',
          flip_rate=0.,
          self_cond_rate=0.5,
          self_cond_by_masking=False,
          cond_dropout=0.,
          guidance=0.,

          # architecture extra
          use_cls_token=False,
          pos_encoding='sin_cos_plus_learned',
          patch_size=8,
          drop_path=0.,
          drop_units=0.,
          drop_att=0.,
      ),

      optimization=D(
          optimizer='lamb',
          learning_rate=1e-3,
          warmup_epochs=0,
          warmup_steps=5000,
          tail_steps=0,
          weight_decay=0.,
          global_clipnorm=1.0,
          momentum=0.9,
          beta1=0.9,
          beta2=0.999,
          eps=1e-8,
          learning_rate_schedule='none',
          learning_rate_scaling='none',
          end_lr_factor=0.0,
          ema_decay=0.9999,
          ema_name_exact_match=True,
          exclude_from_weight_decay='bias,beta,gamma',
      ),

      train=D(
          batch_size=512,
          epochs=100,
          steps=0,
          checkpoint_epochs=40,
          checkpoint_steps=0,
          keep_checkpoint_max=20,
          label_smoothing=0.,
      ),

      eval=D(
          tag='eval',
          checkpoint_dir='',
          batch_size=64,
          steps=100,  # this is an approximation.
      ),
  )

  # Update model with architecture variant.
  for key, value in architecture_config_map[arch_variant].items():
    config.model[key] = value

  return config
