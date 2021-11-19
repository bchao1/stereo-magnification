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

def PIL_list_to_numpy(PIL_list):
    data = []
    for img in PIL_list:
        data.append(np.array(img))
    return data

def psnr_single(x, target):
    x = x.astype(np.float64)
    target = target.astype(np.float64)
    mse = ((x - target)**2).mean()
    return 20 * np.log10(255 / np.sqrt(mse))

def psnr_list(x_list, target_list):
    psnr_ = []
    for x, target in zip(x_list, target_list):
        psnr_.append(psnr_single(x, target))
    return np.array(psnr_).mean()
    
def save_gif(PIL_list, path):
    PIL_list[0].save(path, save_all=True, append_images=PIL_list[1:], duration=100, loop=0)
    
def save_stereo(lf, path):
    left = lf[lf.shape[0] // 2, 0]
    right = lf[lf.shape[0] // 2, -1]
    left = torch_to_numpy_img(left)
    right = torch_to_numpy_img(right)

    Image.fromarray(left).save(f"{path}/left.png")
    Image.fromarray(right).save(f"{path}/right.png")

renders = load_mpi_renders("../examples/origami_v2/results", "render")
save_gif(renders, "../examples/origami_v2/lf.gif")
exit()
#alpha_imgs = load_mpi_renders("../examples/origami_v2/results", "alpha")
#save_gif(alpha_imgs, "../examples/origami_v2/alpha.gif")

#rgb_imgs = load_mpi_renders("../examples/origami_v2/results", "rgb")
#save_gif(rgb_imgs, "../examples/origami_v2/rgb.gif")
#exit()

root = "/mount/data/light_field/hci"
lf_folders = os.listdir(root)
lf_folders.sort()

lf = torch.load(os.path.join(root, "origami", "lf.pt"))
lf = reshape_lf(lf)

center_row = lf[lf.shape[0] // 2]

gt_np = PIL_list_to_numpy(torch_batch_to_PIL_list(center_row))
syn_np = PIL_list_to_numpy(load_mpi_renders("../examples/origami_v1/results", "render"))

PSNR = psnr_list(syn_np, gt_np)

print(PSNR)




