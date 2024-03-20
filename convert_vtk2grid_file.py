import numpy as np
import pandas as pd 
import vtk
import argparse
import os 
import pdb

def main(args):

    df_vtk_original= pd.read_csv(args.in_csv)

    class_list = []
    grid_list = []
    
    for _, row in df_vtk_original.iterrows():
        vtk_path = row['surf']
        _, name = os.path.split(vtk_path)
        id_file, _ = os.path.splitext(name)

        grid_file = os.path.join(args.grid_dir, id_file + args.ext)
        class_list.append(row['class'])
        grid_list.append(grid_file)

    

    df_out = pd.DataFrame(data={'path':grid_list,
                                'class':class_list
                         })

    pdb.set_trace()
    df_out.to_csv(os.path.join(args.out, args.name +'.csv'))


    if args.split:
        df_test = df_out.sample(frac=0.2)
        df_train = df_out.drop(df_test.index).reset_index()

        df_train_train = df_train.sample(frac=0.8)
        df_train_test = df_train.drop(df_train_train.index)

        df_test.to_csv(os.path.join(args.out, args.name +'_test.csv'))
        df_train_test.to_csv(os.path.join(args.out, args.name +'_train_test.csv'))
        df_train_train.to_csv(os.path.join(args.out, args.name +'_train_train.csv'))

        print(len(df_out), len(df_test), len(df_train_test), len(df_train_train))
        print('done')


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--grid_dir', type=str, help='directory containing grid files', required=True)
    parser.add_argument('--in_csv', type=str, help='input CSV file containing meshes', required=True)
    parser.add_argument('--out', type=str, help='output directory to save csv files containing grid paths', default='./')

    parser.add_argument('--split', type=bool, help='split csv in train/val/test', default=None)

    parser.add_argument('--name', type=str, help='name of output csv', default='grid')
    parser.add_argument('--ext', type=str, help='extension of grid file', default='.pt')

    args = parser.parse_args()

    main(args)