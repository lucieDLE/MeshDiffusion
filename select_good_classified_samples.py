import pandas as pd 
import os 
import argparse

def main(args):

    df_new_train = pd.DataFrame()
    df_missclassified = pd.DataFrame()


    in_file = os.path.join(args.input_dir, 'condyles_4classes_aggregate_prediction.csv')

    df = pd.read_csv(in_file)
    df_true = df.loc[df['class'] == df['pred']]

    df_false = df.drop(df_true.index)

    df_new_train = pd.concat([df_new_train, df_true])
    df_missclassified = pd.concat([df_missclassified, df_false])

    df_new_train.to_csv(os.path.join(args.input_dir, 'condyles_4classes_cleaned.csv'))
    df_missclassified.to_csv(os.path.join(args.input_dir, 'condyles_4classes_misclassified.csv'))


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Evaluate Diffusion Model')
    parser.add_argument('--input_dir', type=str, help='path to test folder of k-fold classification', required=True)

    args = parser.parse_args()

    main(args)