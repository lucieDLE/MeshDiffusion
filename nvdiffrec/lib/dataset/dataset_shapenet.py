# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch

import os
import json
import pandas as pd

###############################################################################
# ShapeNet Dataset to get meshes
###############################################################################

class ShapeNetDataset(object):
    def __init__(self, filename):
        name, ext = os.path.splitext(filename)

        if ext == '.csv':
            df = pd.read_csv(filename)
            self.mesh_list = df['surf'].to_list()
                             
        elif ext == '.json':
            with open(filename, 'r') as f:
                self.mesh_list = json.load(f)
        else:
            print('extension not recognised')
        
        print(f"len all data: {len(self.mesh_list)}")

    def __len__(self):
        return len(self.mesh_list)


    def __getitem__(self, idx):
        return self.mesh_list[idx]
