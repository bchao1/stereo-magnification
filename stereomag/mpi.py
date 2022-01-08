import os
import time
import torch
import geometry.projector as pj

class MPI(object):
    def __init__():
        pass

    def infer_mpi(self,
        raw_src_images,
        raw_ref_image,
        ref_pose,
        src_poses,
        intrinsics,
        which_color_pred,
        num_mpi_planes,
        psv_planes,
        extra_outputs=""
    ):
        """Construct the MPI inference graph.
        Args:
            raw_src_images: stack of source images [batch, height, width, 3*#source]
            raw_ref_image: reference image [batch, height, width, 3]
            ref_pose: reference frame pose (world to camera) [batch, 4, 4]
            src_poses: source frame poses (world to camera) [batch, #source, 4, 4]
            intrinsics: camera intrinsics [batch, 3, 3]
            which_color_pred: method for predicting the color at each MPI plane (see README)
            num_mpi_planes: number of MPI planes to predict
            psv_planes: list of depth of plane sweep volume (PSV) planes
            extra_outputs: extra variables to output in addition to RGBA layers
        Returns:
            outputs: a collection of output tensors.
        """
        batch_size, _, img_height, img_width = raw_src_images.shape
        src_images = self.preprocess_image(raw_src_images)
        ref_image = self.preprocess_image(raw_ref_image)

        net_input = self.format_network_input(ref_image, src_images[:, :, :, 3:], ref_pose, src_poses[:, 1:], psv_planes, intrinsics)

        if which_color_pred == "bg":
            mpi_pred, mpi_portion = 

    def format_network_input(self,
        ref_image,
        psv_src_images,
        ref_pose,
        psv_src_poses,
        planes,
        intrinsics
    ):
        num_psv_source = psv_src_poses.shape[0]
        net_input = []
        net_input.append(ref_image)
        for i in range(num_psv_source):
            cur_pose = torch.matmul(psv_src_poses[:, i], torch.inverse(ref_pose))
            cur_image = psv_src_images[:, :, :, i * 3:(i + 1) * 3]
            cur_psv = pj.plane_sweep(cur_image, planes, cur_pose, intrinsics)
            net_input.append(cur_psv)
        net_input = torch.cat(net_input, dim=3)
        return net_input

    def preprocess_image(self, image: torch.TensorType):
        image = torch.FloatTensor(image)
        return image * 2 - 1
