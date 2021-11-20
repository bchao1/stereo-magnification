import os 
import glob
import configparser
import imageio
from utils import *
# Process camera intrinsic and extrinsics for New HCI light field data

def disp_to_depth(disp, beta, focus_distance, sensor_size, baseline, focal_length, resolution):
    return beta / (disp * focus_distance * sensor_size + baseline * focal_length * resolution)

def depth_to_disp(depth, beta, focus_distance, sensor_size, baseline, focal_length, resolution):
    return (beta / depth - baseline * focal_length * resolution) / (focus_distance * sensor_size)

root = "/mnt/data/bchao/lf_datasets/hci_data/"
lf_folders = os.listdir(root)

for folder in lf_folders:
    #depth_map = read_pfm_file(os.path.join(root, folder, "gt_depth_lowres.pfm"))
    file = os.path.join(root, folder, "parameters.cfg")
    config = configparser.ConfigParser()
    config.read(file)
    
    intrinsics = config["intrinsics"]
    extrinsics = config["extrinsics"]
    meta = config["meta"]

    pixel_resolution = float(intrinsics["image_resolution_x_px"])
    baseline_m = float(extrinsics["baseline_mm"]) * 1e-3
    focal_length_m = float(intrinsics["focal_length_mm"]) * 1e-3
    pixel_size_m = float(intrinsics["sensor_size_mm"]) / pixel_resolution * 1e-3
    print(pixel_size_m)
    focus_distance_m = float(extrinsics["focus_distance_m"])
    sensor_size_m = float(intrinsics["sensor_size_mm"]) * 1e-3
    fx = focal_length_m / sensor_size_m
    disp_min_pix = float(meta["disp_min"])
    disp_max_pix = float(meta["disp_max"])
    beta = baseline_m * focal_length_m * focus_distance_m * pixel_resolution
    principal_shift_m = focal_length_m * baseline_m / focus_distance_m
    principal_shift_pix = principal_shift_m / pixel_size_m

    depth_max = disp_to_depth(disp_min_pix, beta, focus_distance_m, sensor_size_m, baseline_m, focal_length_m, pixel_resolution)
    depth_min = disp_to_depth(disp_max_pix, beta, focus_distance_m, sensor_size_m, baseline_m, focal_length_m, pixel_resolution)
    
    disp_max = depth_to_disp(depth_min, beta, focus_distance_m, sensor_size_m, baseline_m, focal_length_m, pixel_resolution)
    disp_min = depth_to_disp(depth_max, beta, focus_distance_m, sensor_size_m, baseline_m, focal_length_m, pixel_resolution)
    print(f"{folder} | Depth: ({depth_min}, {depth_max}) | fx {fx} | x_shift {principal_shift_pix} | baseline {baseline_m}")
    print(disp_min, disp_max)
    print("focus", focus_distance_m)

#greek | Depth range: (5.817390441894531, 8.418572425842285) | Baseline 0.08 | Min disp -3.5 | fx: 2.857
