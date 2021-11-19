import os
import glob

root = "/mnt/data/bchao/lf_datasets/hci_data/"
lf_folders = os.listdir(root)

files = glob.glob(os.path.join(root, "origami", "*Cam*"))
files.sort()
for i, file in enumerate(files):
    os.system(f"cp {file} ../sfm_images/{i}.jpeg")