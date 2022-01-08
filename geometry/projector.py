import torch
from . import homography

def meshgrid_abs(batch, height, width, is_homogeneous=True):
    xs = torch.linspace(0.0, torch.FloatTensor(width - 1), width)
    ys = torch.linspace(0.0, torch.FloatTensor(height - 1), height)
    xs, ys = torch.meshgrid(xs, ys)

    if is_homogeneous:
        ones = torch.ones_like(xs)
        coords = torch.stack([xs, ys, ones], dim=0)
    else:
        coords = torch.stack([xs, ys], dim=0)
    coords = torch.tile(coords.unsqueeze(0), (batch, 1, 1, 1))
    return coords

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous = True):
    batch, height, width = depth.shape
    depth = torch.reshape(depth, (batch, 1, -1))
    pixel_coords = torch.reshape(pixel_coords, (batch, 3, -1))
    cam_coords = torch.matmul(torch.inverse(intrinsics), pixel_coords) * depth
    if is_homogeneous:
        ones = torch.ones((batch, 1, height * width))
        cam_coords =torch.cat([cam_coords, ones], dim=1)
    cam_coords = torch.reshape(cam_coords, (batch, -1, height, width))
    return cam_coords

def cam2pixel(cam_coords, proj):
    batch, _, height, width = cam_coords.shape
    cam_coords = torch.reshape(cam_coords, (batch, 4, -1))
    unnormalized_pixel_coords = torch.matmul(proj, cam_coords)
    xy_u = unnormalized_pixel_coords[:, 0:2, :]
    z_u = unnormalized_pixel_coords[:, 2:3, :]
    pixel_coords = xy_u / (z_u + 1e-10)
    pixel_coords = torch.reshape(pixel_coords, (batch, 2, height, width))
    return torch.permute(pixel_coords, (0, 2, 3, 1))

def projective_inverse_warp(img, depth, pose, intrinsics, ret_flows=False):
    batch, _, height, width = img.shape

    pixel_coords = meshgrid_abs(batch, height, width)

    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)

    filler = torch.Tensor([0.0, 0.0, 0.0, 1.0]).reshape((1, 1, 4))
    filler = torch.tile(filler, (batch, 1, 1))
    intrinsics = torch.cat([intrinsics, torch.zeros((batch, 3, 1))], dim=2)
    intrinsics = torch.cat([intrinsics, filler], dim=1)

    proj_tgt_cam_to_src_pixel = torch.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)

    output_img = torch.grid_sampler(img, src_pixel_coords)
    if ret_flows:
        return output_img, src_pixel_coords - cam_coords
    else:
        return output_img
    

def plane_sweep(img, depth_planes, pose, intrinsics):
    batch, _, height, width = img.shape
    plane_sweep_volume = []

    for depth in depth_planes:
        cur_depth = torch.full(size=(batch, height, width), fill_value=depth)
        warped_img = projective_inverse_warp(img, cur_depth, pose, intrinsics)
        plane_sweep_volume.append(warped_img)
    plane_sweep_volume = torch.cat(plane_sweep_volume, dim=3)
    return plane_sweep_volume