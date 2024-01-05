import numpy as np
import pandas as pd 
import os 

import pdb

in_file = 'meta_condyles.csv'
class_file = '/CMF/data/lumargot/condyles_4classes_train_vtk.csv'

df_in = pd.read_csv(in_file)
df_class = pd.read_csv(class_file)
pdb.set_trace()
