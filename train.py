import torch
import argparse
import os

from stereomag.mpi import MPI

parser = argparse.ArgumentParser(description="Stereo magnification with learned depths")
parser.add_argument("--checkpoint_dir", default="checkpoints", help="Location to save the models.", type=str)
parser.add_argument('--experiment_name', default='', help='Name for the experiment to run.', type=str)
parser.add_argument(
    '--cameras_glob',
    default='../mpi/camera_metadata/train/????????????????.txt',
    help='Glob string for training set camera files.',
    type=str
)
parser.add_argument('which_color_pred', default='bg', help='Color output format: [alpha_only,single,bg,fgbg,all].', type=str)
parser.add_argument()
parser.add_argument()
parser.add_argument()
parser.add_argument()
parser.add_argument()
parser.add_argument()
parser.add_argument()
parser.add_argument()
parser.add_argument()
parser.add_argument()
parser.add_argument()

def train():
    args = parser.parse_args()
    model_save_path = os.path.join(args.checkpoint_dir, args.experiment_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # data_loader = SequenceDataloader
    # train_batch = data_loader.sample_batch()

    model =

if __name__ == "__main":
    train()