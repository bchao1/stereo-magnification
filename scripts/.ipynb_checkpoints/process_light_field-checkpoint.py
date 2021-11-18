import os
import numpy as np
import torch
from PIL import Image
import glob

# load torch light field data 

def torch_to_numpy_img(data):
    data = data.cpu().numpy()
    data = (data * 255).astype(np.uint8)
    if data.shape[-1] != 3:
        data = np.transpose(data, (1, 2, 0))
    return data 

def reshape_lf(lf):
    N_views = lf.shape[0]
    ang_res = np.sqrt(N_views).astype(np.int)
    return lf.reshape(ang_res, ang_res, *lf.shape[1:])

def load_mpi_renders(path, mode):
    imgs = glob.glob(f'{path}/*{mode}*')
    imgs.sort()
    print(imgs)
    images = []
    for img in imgs:
        images.append(Image.open(img))
    return images

def torch_batch_to_PIL_list(tensor):
    images = []
    for img in tensor:
        img = torch_to_numpy_img(img)
        images.append(Image.fromarray(img))
    return images

def save_gif(PIL_list, path):
    PIL_list[0].save(path, save_all=True, append_images=PIL_list[1:], duration=100, loop=0)

alpha_imgs = load_mpi_renders("../examples/origami_v2/results", "alpha")
save_gif(alpha_imgs, "../examples/origami_v2/alpha.gif")

rgb_imgs = load_mpi_renders("../examples/origami_v2/results", "rgb")
save_gif(rgb_imgs, "../examples/origami_v2/rgb.gif")
exit()

root = "/mount/data/light_field/hci"
lf_folders = os.listdir(root)
lf_folders.sort()

lf = torch.load(os.path.join(root, "origami", "lf.pt"))
lf = reshape_lf(lf)

left = lf[lf.shape[0] // 2, 0]
right = lf[lf.shape[0] // 2, -1]
left = torch_to_numpy_img(left)
right = torch_to_numpy_img(right)

Image.fromarray(left).save("../examples/origami_v2/left.png")
Image.fromarray(right).save("../examples/origami_v2/right.png")
exit()





