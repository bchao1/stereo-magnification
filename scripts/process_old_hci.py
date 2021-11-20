import numpy as np
import h5py
from PIL import Image

filename = "/mnt/data/bchao/lf_datasets/old_hci/blender/buddha/lf.h5"
with h5py.File(filename, "r") as file:
    # List all groups
    attrs = file.attrs.keys()
    metadata = {}
    for attr in file.attrs.keys():
        metadata[attr] = file.attrs[attr]
    #print(metadata.keys())
    baseline = metadata["dH"]
    f = metadata["focalLength"] # in pixel
    shift = metadata["shift"]
    print(f"Baseline {baseline} | F {f} | Shift {shift}")
    # Get the data
    lf = file["LF"][()]
    depth = file["GT_DEPTH"][()]
    print(depth.min(), depth.max())
    left = lf[4, 0]
    right = lf[4, -1]
    Image.fromarray(left).save("../examples/buddha/left.png")
    Image.fromarray(right).save("../examples/buddha/right.png")
