import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .models import layers
from . import losses
from .models import utils as mutils
from .models.ema import ExponentialMovingAverage
from .utils import restore_checkpoint
from . import sde_lib

def eval_gen(config, idx=0):
  np_generated = np.load('/home/lumargot/MeshDiffusion/all_grids.npy') ## n_class* n_sample, 4, res, res, res
  df_original = pd.read_csv('/home/lumargot/MeshDiffusion/condyles_4classes_grid_train.csv') ## torch.load(filename)
  outfile = '/home/lumargot/MeshDiffusion/eval_neigbors.csv'

  ## Model initialization
  print("Initializing model")
  score_model = mutils.create_model(config, use_parallel=False)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)


  timesteps = torch.linspace(sde.T, 1e-3, sde.N, device=config.device)
  smallest_t = timesteps[-1]

  print("Restoring checkpoint")
  state = restore_checkpoint(config.eval.ckpt_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  score_model.eval().requires_grad_(False)

  l_origin_features = []

  l_file = []
  l_in_class = []
  l_out_class = []
  l_origin_idx = []

  print("computing features of original samples")
  for idx, row in df_original.iterrows():
    origin_sample = torch.load(row['path']).to(config.device)
    batch_timestep = torch.ones(1, device=config.device) * smallest_t

    ## compute feature
    labels = torch.Tensor([row['class']]).long().to(config.device)
    origin_feature = compute_features(score_model, config, origin_sample,
                                       labels, batch_timestep)
    
    l_origin_features.append(origin_feature.to('cpu'))
  print("Done")
  np_features = np.stack(l_origin_features)
  np_features = np_features.reshape((173, 512*4*4*4))

  # flatt = np_features.reshape((np_features.shape[0], np_features.shape[1]*4*4*4))



  print("Initializing Nearest Neighbor")
  #Nearest Neighbor Algorithm
  k_neighbors = 1
  knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
  knn_model.fit(np_features)

  j=0
  class_idx=-1

  print("computing generated features and nearest neighbor")
  for gen_sample in np_generated:

    gen_sample = torch.from_numpy(gen_sample)
    if j %128 == 0:
      class_idx +=1

    ## compute feature
    batch_timestep = torch.ones(1, device=config.device) * smallest_t
    gen_feature = compute_features(score_model, config, gen_sample.to(config.device), 
                                   torch.Tensor([class_idx]).long().to(config.device), batch_timestep)
    j +=1

    gen_feature = gen_feature.to('cpu')
    gen_feature = gen_feature.reshape((gen_feature.shape[0], 512*4*4*4))
    ## then compute nearest neighbors
    distances, indices = knn_model.kneighbors(gen_feature)
    pred_class = df_original.iloc[indices[0]]['class'].to_numpy()


    l_file.append(j)
    l_in_class.append(class_idx)
    l_out_class.append(pred_class)
    l_origin_idx.append(indices[0])

  print("Done")


  df_out = pd.DataFrame({'gen_sample':l_file,
                        'gen_class':l_in_class,
                        'pred_class':l_out_class,
                        'origin_sample_idx':l_origin_idx})
  
  df_out.to_csv(outfile)

import pdb
def compute_features(model, config, x, labels, timesteps):

  labels = labels.reshape(1,)

  # act = layers.get_act(config)
  # pdb.set_trace()
  modules =  model.all_modules
  m_idx = 0

  # timestep/scale embedding
  temb = layers.get_timestep_embedding(timesteps, model.nf)
  # temb += class_emb
  temb = modules[m_idx](temb)
  m_idx += 1
  temb = modules[m_idx](model.act(temb))
  m_idx += 1
  # pdb.set_trace()
  
  tmp = nn.functional.one_hot(labels, num_classes=model.num_classes).float()
  embeddings=model.map_label(tmp)

  temb = temb + embeddings

  h = x

  # Downsampling block    
  hs = [modules[m_idx](h) + model.pos_layer(model.coords) + model.mask_layer(model.mask)]
  m_idx += 1
  for i_level in range(model.num_resolutions):
    # Residual blocks for this resolution
    for i_block in range(model.num_res_blocks):
      h = modules[m_idx](hs[-1], temb)
      m_idx += 1
      if h.shape[-1] in model.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      hs.append(h)
    if i_level != model.num_resolutions - 1:
      hs.append(modules[m_idx](hs[-1]))
      m_idx += 1

  h = hs[-1]
  h = modules[m_idx](h, temb)
  m_idx += 1
  h = modules[m_idx](h)
  m_idx += 1
  h = modules[m_idx](h, temb)
  m_idx += 1

  return h 