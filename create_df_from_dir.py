import pandas as pd 
import os
import pdb
import numpy as np
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    '02858304': 'boat', '02834778': 'bicycle',
}


# input_folder='/CMF/data/lumargot/ShapeNet/obj/'
# out_path = '/CMF/data/lumargot/ShapeNet/obj_path.csv'
# l_class=[]
# l_tag=[]
# l_surf=[]

# class_id = 0
# for sub_dir in os.listdir(input_folder):
#     print(sub_dir)
#     if sub_dir != '.git':
#         sub_dir_path = os.path.join(input_folder, sub_dir)
#         if os.path.isdir(sub_dir_path):
#             for sub_sub in os.listdir(sub_dir_path):

#                 fold_path = os.path.join(sub_dir_path,sub_sub)
#                 fold_path = os.path.join(fold_path, 'models')

#                 for filename in os.listdir(fold_path):
#                     file_path = os.path.join(fold_path, filename)
#                     basename, ext = os.path.splitext(filename)

#                     if ext == '.obj':

#                         l_tag.append(synsetid_to_cate[sub_dir])
#                         l_class.append(class_id)
#                         l_surf.append(file_path)
        
#             class_id +=1
# pdb.set_trace()
# df = pd.DataFrame({'class':l_class,'surf':l_surf,'tag': l_tag})
# df.to_csv(out_path)



input_folder= '/CMF/data/lumargot/DCBIA/TMJOA_2018/meshes/generated/'
outpath='/home/lumargot/MeshDiffusion/all_grids.npy'
all_mat = []
for sub_dir in os.listdir(input_folder):
    print(sub_dir)
    class_dir = os.path.join(input_folder, sub_dir)
    if os.path.isdir(class_dir):
        for filename in os.listdir(class_dir):

            filepath = os.path.join(class_dir, filename)
            basename, ext = os.path.splitext(filepath)
            if ext == '.npy':
                mat = np.load(filepath) ## BS, channels, res, res, res 
                for array in mat:
                    all_mat.append(array)

pdb.set_trace()
all_mat = np.stack(all_mat)
np.save(outpath,all_mat)