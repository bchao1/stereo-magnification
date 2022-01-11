python3 train.py \
    --checkpoint_dir=checkpoints \
    --cameras_glob=../../mnt/MPI/camera_metadata/train/????????????????.txt \
    --image_dir=../../mnt/MPI/images/train \
    --experiment_name=origin_stereo \
    --continue_train=True \
    --save_latest_freq=100