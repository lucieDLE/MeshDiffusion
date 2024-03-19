import os
import sys
import glob 
import subprocess
import argparse
import pandas as pd
import numpy as np
import pdb

class bcolors:
    HEADER = '\033[95m'
    #blue
    PROC = '\033[94m'
    #CYAN
    INFO = '\033[96m'
    #green
    SUCCESS = '\033[92m'
    #yellow
    WARNING = '\033[93m'
    #red
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_last_checkpoint(checkpoint_dir):
    # Get the last checkpoint
    checkpoint_paths = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_paths = sorted(glob.iglob(checkpoint_paths), key=os.path.getctime, reverse=True) 
    if len(checkpoint_paths) > 0:
        return checkpoint_paths[0]
    return None

def get_val_loss(checkpoint_path):
    # Get the last checkpoint
    checkpoint_fn = os.path.basename(checkpoint_path)
    checkpoint_dict = eval('dict(%s)' % checkpoint_fn.replace('-',',').replace(' ','').replace('.ckpt', ''))
    return checkpoint_dict['val_loss']

def get_best_checkpoint(checkpoint_dir):
    # Get the last checkpoint
    checkpoint_paths = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_paths = sorted(glob.iglob(checkpoint_paths), key=get_val_loss) 
    if len(checkpoint_paths) > 0:
        return checkpoint_paths[0]
    return None

def get_argparse_dict(parser):
    # Get the default arguments from the parser
    default = {}
    for action in parser._actions:
        if action.dest != "help":
            default[action.dest] = action.default
    return default

def replace_extension(fname, new_extension):
    return os.path.splitext(fname)[0] + new_extension

def get_output_filename(fname, suffix):
    return replace_extension(fname, f'_{suffix}')

def save_data_folds(df_train,df_test,fname,i):
    train = fname.replace('.csv', '_fold' + str(i) + '_train.csv')
    df_train.to_csv(train, index=False)
    eval_fn = fname.replace('.csv', '_fold' + str(i) + '_test.csv')
    df_test.to_csv(eval_fn, index=False)
   
def save_data(df_train,df_test,fname,csv):
    if csv:
        train = fname.replace('.csv', '_train.csv')
        df_train.to_csv(train, index=False)
        eval_fn = fname.replace('.csv', '_test.csv')
        df_test.to_csv(eval_fn, index=False)
    else:
        train_fn = fname.replace('.parquet', '_train.parquet')
        df_train.to_parquet(train_fn, index=False)
        eval_fn = fname.replace('.parquet', '_test.parquet')
        df_test.to_parquet(eval_fn, index=False)

def split_data_folds_test_train(fname, k_folds, split):
    df = pd.read_csv(fname)

    group_ids = np.array(range(len(df.index)))
    samples = int(len(group_ids)*split)
    np.random.shuffle(group_ids)
    start_f = 0
    end_f = samples

    for i in range(k_folds):
        id_test = group_ids[start_f:end_f]
        df_train = df[~df.index.isin(id_test)]
        df_test = df.iloc[id_test]

        save_data_folds(df_train,df_test,fname,i)

        start_f += samples
        end_f += samples



def create_generated_csv(input_folder, out_file):

    all_labels = []
    l_all_files = []
    for sub_dir in os.listdir(input_folder):
        label=sub_dir
        class_dir = os.path.join(input_folder, sub_dir)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir): ## iterates on inside : meshes and *.npy

                filepath = os.path.join(class_dir, filename)

                if os.path.isdir(filepath):
                    for mesh in os.listdir(filepath):
                        mesh_path = os.path.join(filepath, mesh)
                        l_all_files.append(mesh_path)
                        all_labels.append(label)

    df_labels = pd.DataFrame({'surf': l_all_files, 'labels': all_labels})
    df_labels.to_csv(out_file)
 

###################################################################################################################################################################################################
##################################################################################    MAIN    #####################################################################################################
###################################################################################################################################################################################################


def main(args, arg_groups):
    # Main function
    create_folds = False

    DIFFUSION_CONFIG = args.diff_cfg
    MOUNT_POINT = args.mount_point
    NUM_SAMPLES = args.num_samples

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if not os.path.exists(args.data_out):
        os.makedirs(args.data_out)
    
    # Kill the program if the number of folds is 0
    if args.folds == 0:
        sys.exit("The value of nn is 0. You must specify a value greater than 0.")

    # Kill the program if the split is negative
    if args.valid_split < 0:
        sys.exit("The value of split is negative. You must specify a value greater than 0.")

################################################################################## SPLIT PART #####################################################################################################

    df_train =pd.read_csv(args.csv)
    num_class= len(np.unique(df_train['class']))

    for f in range(args.folds):

        ext = os.path.splitext(args.csv)[1]
        csv_train = get_output_filename(args.csv, f'fold{f}_train_train.csv')
        csv_test = get_output_filename(args.csv, f'fold{f}_test.csv')

        if not os.path.exists(csv_train) or not os.path.exists(csv_test):
            create_folds = True
            break

    if create_folds:

        split = 0.2

        # Creation of test and train dataset for each fold
        # csv_train = get_output_filename(args.csv, 'train.csv')
        split_data_folds_test_train(args.csv, args.folds, split)

        print(f"{bcolors.SUCCESS}End of creating the {args.folds} folds {bcolors.ENDC}")

#################################################################################### TRAIN PART #####################################################################################################

    for f in range(args.folds):
        #Train the model for each fold
        print(bcolors.INFO, "Start training for fold {f}".format(f=f), bcolors.ENDC)
        csv_train = args.csv.replace(ext, '_fold{f}_train.csv').format(f=f)

        training_dir = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))
        output_dir = os.path.join(args.data_out, 'train', 'fold{f}'.format(f=f))

        if not os.path.exists(training_dir):
            os.makedirs(training_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # last_checkpoint = get_last_checkpoint(saxi_train_args['out'])
        last_checkpoint = None
        
        if last_checkpoint is None:
            command = [sys.executable, os.path.join(os.path.dirname(__file__), 'main_diffusion.py')]
            command.append(f'--mode=train')
            command.append(f'--config={DIFFUSION_CONFIG}')
            command.append(f'--config.data.meta_path={csv_train}')
            command.append(f'--config.eval.eval_dir={output_dir}')
            command.append(f'--config.training.train_dir={training_dir}')

            subprocess.run(command)

        print(bcolors.SUCCESS, "End training for fold {f}".format(f=f), bcolors.ENDC)

#################################################################################### GENERATION PART #####################################################################################################

    for f in range(args.folds):

        print(bcolors.INFO, "Start generating {n} for fold {f}".format(n= NUM_SAMPLES, f=f), bcolors.ENDC)


        csv_train = args.csv.replace(ext, '_fold{f}_train.csv').format(f=f)

        training_dir = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))
        output_dir = os.path.join(args.data_out, 'train', 'fold{f}'.format(f=f))

        ckpt_path = os.path.join(training_dir, 'checkpoints/checkpoint_100000.pth')

        for class_id in range(num_class):

            print(bcolors.INFO, "Class {f} ...".format(f=class_id), bcolors.ENDC)
            class_dir = os.path.join(output_dir, str(class_id))

            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            """
            python main_diffusion.py --config=$DIFFUSION_CONFIG --mode=uncond_gen \
            --config.eval.eval_dir=$OUTPUT_PATH  --config.eval.ckpt_path=$CKPT_PATH
            """

            command = [sys.executable, os.path.join(os.path.dirname(__file__), 'main_diffusion.py')]
            command.append(f'--mode=uncond_gen')
            command.append(f'--config={DIFFUSION_CONFIG}')
            # command.append(f'--config.data.meta_path={csv_train}')
            command.append(f'--config.eval.eval_dir={class_dir}')
            command.append(f'--config.eval.ckpt_path={ckpt_path}')
            command.append(f'--config.eval.gen_class={int(class_id)}')
            command.append(f'--config.eval.batch_size={args.batch_size}')
            command.append(f'--config.eval.num_samples={NUM_SAMPLES}')

            subprocess.run(command)
            # print(command)
        print(bcolors.SUCCESS, "End Generating samples for fold {f}".format(f=f), bcolors.ENDC)


#################################################################################### RECONSTRUCTION PART #####################################################################################################

    # change directory
    print(bcolors.INFO, "Changing working directory to reconstruct samples".format(f=f), bcolors.ENDC)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    nvdiff = os.path.join(script_directory, 'nvdiffrec')        
    DMTET_CONFIG = os.path.join(nvdiff, 'configs/res64.json')


    for f in range(args.folds):

        print(bcolors.INFO, "Start Reconstructing samples for fold {f}".format(f=f), bcolors.ENDC)

        output_dir = os.path.join(args.data_out, 'train', 'fold{f}'.format(f=f))

        for class_id in range(num_class):

            print(bcolors.INFO, "Class {f} ...".format(f=class_id), bcolors.ENDC)
            class_dir = os.path.join(output_dir, str(class_id))

            npy_files = os.listdir(class_dir)
            npy_files = [file for file in npy_files if file.endswith('.npy')]

            meshdir = class_dir
    
            for sample in npy_files:

                """  
                cd nvdiffrec
                python eval.py --config $DMTET_CONFIG --out-dir $OUT_DIR --sample-path $SAMPLE_PATH \
                --deform-scale $DEFORM_SCALE [--angle-ind $ANGLE_INDEX] [--num-smoothing-steps $NUM_SMOOTHING_STEPS]
                """
                
                sample_path = os.path.join(class_dir, sample)
                command = [sys.executable, os.path.join(nvdiff, 'eval.py')]

                command.append(f"--config={DMTET_CONFIG}")
                command.append(f"--out-dir={meshdir}")
                command.append(f"--sample-path={sample_path}")
                command.append(f"--deform-scale={int(3)}")

                subprocess.run(command)
        print(bcolors.SUCCESS, "End Reconstruction for fold {f}".format(f=f), bcolors.ENDC)

            

#################################################################################### Cleaning PART #####################################################################################################
    # exit()
    for f in range(args.folds):
        
        print(bcolors.INFO, "Start Cleaning process for fold {f}".format(f=f), bcolors.ENDC)

        csv_test = args.csv.replace('.csv', '_fold{f}_test.csv').format(f=f)
        csv_train = args.csv.replace('.csv', '_fold{f}_train.csv').format(f=f)
        input_dir = os.path.join(args.data_out, 'train', 'fold{f}'.format(f=f))


        training_dir = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))
        ckpt_path = os.path.join(training_dir, 'checkpoints/checkpoint_10000.pth')


        csv_sample=os.path.join(input_dir, 'condyles_4classes_cleaned_train.csv')


        """
        python main_diffusion.py --mode eval_gen --config $DIFFUSION_CONFIG --config.eval.ckpt_path $CKPT_PATH  --config.eval.eval_dir $OUTPUT_PATH--config.data.meta_path $SAMPLE_PATH --config.data.mount_point $MOUNT_POINT
        """

        command = [sys.executable, os.path.join(os.path.dirname(__file__), 'main_diffusion.py')]
        command.append(f'--mode=eval_gen')
        command.append(f'--config={DIFFUSION_CONFIG}')
        command.append(f'--config.data.meta_path={csv_test}')
        command.append(f'--config.eval.eval_dir={input_dir}')
        command.append(f'--config.eval.ckpt_path={ckpt_path}')
        command.append(f'--config.data.mount_point={MOUNT_POINT}')

        subprocess.run(command)

        print(bcolors.SUCCESS, "End cleaning for fold {f}".format(f=f), bcolors.ENDC)


        # ## TO DO : Evaluate each model 
        # """
        # python test-meshDiff-metrics.py --csv_sample condyles_generated_vtk.csv --csv_sample test_csv_vtk.csv --out_csv results.csv
        # """
    
        # print(bcolors.INFO, "Start evaluation for fold {f}".format(f=f), bcolors.ENDC)
        # out_csv = os.path.join(output_dir, 'fold{f}_results.csv'.format(f=f))

        # command = [sys.executable, os.path.join(nvdiff, 'test-meshDiff-metrics.py')]

        # command.append(f"--csv_sample={csv_sample}")       # the generated data as .vtk
        # command.append(f"--csv_original-dir={csv_test}")   # containing the .pt files 
        # command.append(f"--mount_point={MOUNT_POINT}")     # where the vtk files are 
        # command.append(f"--out_csv={out_csv}")

        # print(bcolors.SUCCESS, "End of evaluation for fold {f}".format(f=f), bcolors.ENDC)


################################################################################# EVALUATION PART #####################################################################################################

    ## TO DO
        
################################################################################## AGGREGATE PART #####################################################################################################

    ## TO DO
        
############################################################## TEST + EVALUATION OF THE BEST MODEL ######################################################################################################################

    ## TO DO

#####################################################################################################################################################################################################

def cml():
    # Command line interface
    parser = argparse.ArgumentParser(description='Automatically train and evaluate a N fold cross-validation model for Shape Analysis Explainability and Interpretability')

    # Arguments used for split the data into the different folds
    split_group = parser.add_argument_group('Split')
    split_group.add_argument('--csv', help='CSV with columns surf,class', type=str, required=True)
    split_group.add_argument('--folds', help='Number of folds', type=int, default=5)
    split_group.add_argument('--valid_split', help='Split float [0-1]', type=float, default=0.2)
    split_group.add_argument('--group_by', help='GroupBy criteria in the CSV. For example, SubjectID in case the same subjects has multiple timepoints/data points and the subject must belong to the same data split', type=str, default=None)

    # Arguments used for training
    train_group = parser.add_argument_group('Train')
    train_group.add_argument('--batch_size', help='batch size', type=int, default=16)
    train_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    train_group.add_argument('--diff_cfg', help='path to diffusion', type=str, default="configs/res_64.py")

    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--out', help='directory to save models', type=str, default="./")
    out_group.add_argument('--data_out', help='directory to save data', type=str, default="./")
    out_group.add_argument('--num_samples', help='number of samples to generate', type=int, default=128)

    args = parser.parse_args()

    arg_groups = {}
    for group in parser._action_groups:
        arg_groups[group.title] = {a.dest:getattr(args,a.dest,None) for a in group._group_actions}

    main(args, arg_groups)


if __name__ == '__main__':
    cml()