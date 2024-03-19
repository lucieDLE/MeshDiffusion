"""Config file for reproducing the results of DDPM on bedrooms."""

from configs.default_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = False
  training.reduce_mean = True
  training.batch_size = 2
  training.lip_scale = None
  training.train_dir = './'

  training.snapshot_freq_for_preemption = 1000

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'ancestral_sampling'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.dataset = 'ShapeNet'
  data.centered = True
  data.image_size = 64
  data.num_channels = 4
  data.train_csv = "train.csv" 
  data.val_csv = "val.csv" 
  data.num_workers = 4
  data.aug = True
  data.labels=True
  data.mount_point = 'data'
  data.meta_path = ''


  # model
  model = config.model
  model.name = 'ddpm_res64_cond'
  model.num_classes = 4
  model.scale_by_sigma = False
  model.num_scales = 1000
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 4, 4)
  model.num_res_blocks_first = 2
  model.num_res_blocks = 3
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.dropout = 0.1

  # optim
  optim = config.optim
  optim.lr = 1e-6

  config.eval.batch_size = 32

  config.eval.num_samples = 64
  config.eval.eval_dir = ''
  config.eval.gen_class = 0

  config.eval.ckpt_path = training.train_dir 
  config.eval.partial_dmtet_path = config.eval.eval_dir
  config.eval.tet_path = config.eval.eval_dir

  config.seed = 42

  return config