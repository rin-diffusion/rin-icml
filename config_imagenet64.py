"""Config for RIN ImageNet 64x64 experiments."""

# pylint: disable=invalid-name,line-too-long

import config_diffusion_base as config_base

DATA_NAME = 'imagenet2012'
ARCH_VARIANT = 'tape'
IMAGE_SIZE = 64 * 1


def get_config(config_str=None):
  """Returns config."""
  del config_str
  config = config_base.get_config(f'{DATA_NAME},{ARCH_VARIANT}')
  config.dataset.image_size = IMAGE_SIZE
  config.model.name = 'image_diffusion_model'
  config.model.b_scale = 1.0
  config.model.pred_type = 'eps'
  config.model.conditional = 'class'
  config.model.time_on_latent = True
  config.model.cond_on_latent = True
  config.model.cond_tape_writable = False
  config.optimization.ema_decay = 0.9999
  config.eval.batch_size = 80
  config.eval.steps = 625
  return config


def get_sweep(h):
  """Get the hyperparamater sweep."""
  return h.chainit([
      h.product([
          h.sweep('config.train.steps', [500_000]),
          h.sweep('config.train.checkpoint_steps', [10_000]),
          h.sweep('config.train.keep_checkpoint_max', [20]),
          h.sweep('config.train.batch_size', [1024]),
          h.sweep('config.optimization.optimizer', ['lamb']),
          h.sweep('config.optimization.learning_rate', [1e-3]),
          h.sweep('config.optimization.learning_rate_schedule', ['cosine']),
          h.sweep('config.optimization.end_lr_factor', [0.]),
          h.sweep('config.optimization.warmup_steps', [10000]),
          h.sweep('config.optimization.weight_decay', [1e-2]),
          h.sweep('config.optimization.beta2', [0.999]),
          h.sweep('config.model.train_schedule', ['cosine']),
          h.sweep('config.model.pred_type', ['eps']),
          h.sweep('config.model.self_cond', ['latent']),
          h.sweep('config.model.self_cond_by_masking', [True]),
          h.sweep('config.model.self_cond_rate', [0.9]),
          h.sweep('config.model.total_time_steps', [1000]),

          h.sweep('config.model.patch_size', [2]),  # 4
          h.sweep('config.model.latent_pos_encoding', ['learned']),
          h.sweep('config.model.tape_pos_encoding', ['learned']),
          h.sweep('config.model.num_layers', ['4,4,4,4']),  # '6,6,6,6'
          h.sweep('config.model.latent_slots', [128]),
          h.sweep('config.model.latent_dim', [768]),
          h.sweep('config.model.latent_mlp_ratio', [4]),
          h.sweep('config.model.latent_num_heads', [16]),
          h.sweep('config.model.tape_dim', [512]),
          h.sweep('config.model.tape_mlp_ratio', [4]),
          h.sweep('config.model.rw_num_heads', [16]),
          h.sweep('config.model.drop_units', [0.]),
          h.sweep('config.model.drop_path', [0.]),
      ]),
  ])


def get_eval_args_and_tags(config, args, unused_config_flag):
  """Return eval args and tags."""
  args_and_tags = []
  for eval_split in [config.dataset.train_split]:
    for sampler_name in ['ddpm']:
      for infer_schedule in ['cosine']:
        for infer_iterations in [100, 250, 1000]:
          eval_args = args.copy()
          sampler_name_s = sampler_name.replace('@', '')
          infer_schedule_s = infer_schedule.replace('@', '').replace(',', 'c')
          eval_tag = f'ev_{eval_split}_{sampler_name_s}_{infer_schedule_s}_i{infer_iterations}'
          eval_args.update({
              'config.eval.tag': eval_tag,
              'config.dataset.eval_split': eval_split,
              'config.model.sampler_name': sampler_name,
              'config.model.infer_schedule': infer_schedule,
              'config.model.infer_iterations': infer_iterations,
          })
          args_and_tags.append((eval_args, eval_tag, None))
  return args_and_tags
