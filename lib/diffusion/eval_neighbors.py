import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os 
from .models import layers
from . import losses
from .models import utils as mutils
from .models.ema import ExponentialMovingAverage
from .utils import restore_checkpoint
from . import sde_lib
import pdb

def compute_features(model, config, x, labels, timesteps):
  labels = labels.reshape(1,)
 
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

def aggregate_npy(fold):
  all_labels = []
  all_mat = []
  all_meshes = []

  for sub_dir in os.listdir(fold):
    label=sub_dir
    class_dir = os.path.join(fold, sub_dir)
    if os.path.isdir(class_dir):
      for filename in os.listdir(class_dir):

        filepath = os.path.join(class_dir, filename)
        basename, ext = os.path.splitext(filepath)
        if ext == '.npy':
          mat = np.load(filepath) ## BS, channels, res, res, res 
          for array in mat:
            all_mat.append(array)

        if os.path.isdir(filepath): ## mesh dir
          for mesh in os.listdir(filepath):
            vtkfile=os.path.join(filepath,mesh)
            basename, vtk_ext = os.path.splitext(vtkfile)
            if vtk_ext == '.vtk':
              all_meshes.append(vtkfile)
              all_labels.append(label)

  all_mat = np.stack(all_mat)
  df_labels = pd.DataFrame({'surf':all_meshes, 'class': all_labels})
  return all_mat, df_labels

def replace_path(df, mount_point):

  l_files, l_label = [],[]
  for idx, row in df.iterrows():
    surf = row['surf']
    label = row['class']

    path, name = os.path.split(surf)
    id, ext = os.path.splitext(name)

    file=os.path.join(mount_point, id + '.vtk')

    l_files.append(file)
    l_label.append(label)
  
  return pd.DataFrame(data={'surf':l_files, 'class':l_label})

def eval_gen(config, idx=0):

  input_folder = config.eval.eval_dir
  mount_point = config.data.mount_point
  outfile=os.path.join(input_folder, 'condyles_4classes_cleaned_train.csv')
  num_samples = 96

  df_original = pd.read_csv(config.data.meta_path)

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

  l_origin_features,  l_file, l_in_class, l_out_class, l_origin_idx, l_dist = [], [], [], [], [], []

  print('preparing data')
  np_generated, df_labels = aggregate_npy(input_folder)

  print("computing features of original samples")
  for idx, row in df_original.iterrows():
    origin_sample = torch.load(row['surf']).to(config.device)

    origin_sample=origin_sample.unsqueeze(0)
    
    batch_timestep = torch.ones(1, device=config.device) * smallest_t

    ## compute feature
    labels = torch.Tensor([row['class']]).long().to(config.device)
    origin_feature = compute_features(score_model, config, origin_sample,
                                       labels, batch_timestep)
    
    l_origin_features.append(origin_feature.to('cpu'))
  print("Done")
  
  np_features = np.stack(l_origin_features)
  np_features = np_features.reshape((len(df_original), origin_feature.numel()))
 
  print("Initializing Nearest Neighbor")
  #Nearest Neighbor Algorithm
  k_neighbors = 3
  knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
  knn_model.fit(np_features)

  dist, idx =  knn_model.kneighbors(np_features)
  j=0

  l_idx_keep = []

  print("computing generated features and nearest neighbor")
  for gen_sample in np_generated:
    gen_sample = torch.from_numpy(gen_sample)


    gen_sample = gen_sample.unsqueeze(0)
    class_idx = int(df_labels.iloc[j]['class'])
    surf = df_labels.iloc[j]['surf']

    ## compute feature
    batch_timestep = torch.ones(1, device=config.device) * smallest_t
    gen_feature = compute_features(score_model, config, gen_sample.to(config.device), 
                                   torch.Tensor([class_idx]).long().to(config.device), batch_timestep)
    j +=1

    gen_feature = gen_feature.to('cpu')
    gen_feature = gen_feature.reshape((gen_feature.shape[0], gen_feature.numel()))

    ## then compute nearest neighbors
    distances, indices = knn_model.kneighbors(gen_feature)
    pred_k, dist_k, idx_k = [], [], []
    
    for i in range(k_neighbors):
      dist, idx = distances[:,i].item(), indices[:,i].item()
      pred = df_original.iloc[idx]['class']

      pred_k.append(pred)
      dist_k.append(dist)
      idx_k.append(idx)

    if class_idx == pred_k[0]:
       l_idx_keep.append(surf)

    l_file.append(j)
    l_in_class.append(class_idx)
    l_out_class.append(pred_k)
    l_origin_idx.append(idx_k)
    l_dist.append(dist_k)

    pred_k, dist_k, idx_k = [], [], []

  df_labels = df_labels[df_labels['surf'].isin(l_idx_keep)]


  pdb.set_trace()
  df_original = replace_path(df_original, mount_point)



  num_samples=2000
  for i in range(class_idx+1):
    df_i = df_labels.loc[df_labels['class']==str(i)]
    num_samples = min(len(df_i), num_samples)


  df_labels_samples = pd.DataFrame(columns=['surf', 'class'])
  for i in range(class_idx+1):

    df_i = df_labels.loc[df_labels['class']==str(i)]
    min_samples = min(len(df_i), num_samples)
    df_i = df_i.sample(min_samples)
    df_labels_samples = pd.concat([df_labels_samples, df_i])

  df_all = pd.concat([df_labels_samples, df_original])
  df_all.to_csv(outfile)