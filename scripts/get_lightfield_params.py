import os 
import glob
import configparser
import imageio
from utils import *
# Process camera intrinsic and extrinsics for New HCI light field data

root = "/mnt/data/bchao/lf_datasets/hci_data/"
lf_folders = os.listdir(root)

for folder in lf_folders:
    try:
        depth_map = read_pfm_file(os.path.join(root, folder, "gt_depth_lowres.pfm"))
        file = os.path.join(root, folder, "parameters.cfg")
        config = configparser.ConfigParser()
        config.read(file)
        
        intrinsics = config["intrinsics"]
        extrinsics = config["extrinsics"]
        meta = config["meta"]

        baseline_m = float(extrinsics["baseline_mm"]) * 1e-3
        focal_length_m = float(intrinsics["focal_length_mm"]) * 1e-3
        pixel_size_m = float(intrinsics["sensor_size_mm"]) / float(intrinsics["image_resolution_x_px"]) * 1e-3
        fx = float(intrinsics["focal_length_mm"]) / float(intrinsics["sensor_size_mm"])   

        disp_min = meta["disp_min"]
        min_disp_pix = float(meta["disp_min"])
        max_disp_pix = float(meta["disp_max"])

        #min_disp_pix = 1
        #max_disp_pix += (1 - min_disp_pix)

        min_depth_m = baseline_m * focal_length_m / (min_disp_pix * pixel_size_m)
        max_depth_m = baseline_m * focal_length_m / (max_disp_pix * pixel_size_m)
        print(f"{folder} | Depth range: ({depth_map.min()}, {depth_map.max()}) | Baseline {baseline_m} | Min disp {disp_min} | Fx {fx}")
        print(f"{folder} | Depth range: ({min_depth_m}, {max_depth_m}) | Baseline {baseline_m} | Min disp {disp_min} | Fx {fx}")
    except:
        pass

#greek | Depth range: (5.817390441894531, 8.418572425842285) | Baseline 0.08 | Min disp -3.5 | fx: 2.857
