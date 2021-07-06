from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os

def save_path_formatter(args, parser):
    job_id = os.getenv('SLURM_JOB_ID')
    job_name = os.getenv('SLURM_JOB_NAME')
    commit = Path(os.popen("git log --pretty=format:'%h' -n 1").read())

    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)

    save_path = Path(','.join([job_name,job_id]))
    return data_folder_name/commit/save_path



def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def log_output_tensorboard(writer, prefix, index, suffix, n_iter, depth, disp, warped, diff, mask, disparities, uncertainties):
    disp_to_show = tensor2array(disp[0], max_value=None, colormap='magma')
    depth_to_show = tensor2array(depth[0], max_value=None)
    writer.add_image('{} Dispnet Averaged {}/{}'.format(prefix, suffix, index), disp_to_show, n_iter)
    writer.add_image('{} Depth Averaged {}/{}'.format(prefix, suffix, index), depth_to_show, n_iter)
    # log warped images along with explainability mask

    for j, (disp, uncertainty) in enumerate(zip(disparities, uncertainties)): 
        disp_to_show = tensor2array(disp[0], max_value=None, colormap='magma')
        writer.add_image('{} Dispnet Output Nr {} {}/{}'.format(prefix, j,suffix, index), disp_to_show, n_iter)
        unvertainty_to_show = tensor2array(uncertainty[0], max_value=None, colormap='magma')
        writer.add_image('{} Uncertainty Output Nr {} {}/{}'.format(prefix, j,suffix, index), unvertainty_to_show, n_iter)

    if (warped is None) or (diff is None):
        return
    for j, (warped_j, diff_j) in enumerate(zip(warped, diff)):
        whole_suffix = '{} {}/{}'.format(suffix, j, index)
        warped_to_show = tensor2array(warped_j)
        diff_to_show = tensor2array(0.5*diff_j)
        writer.add_image('{} Warped Outputs {}'.format(prefix, whole_suffix), warped_to_show, n_iter)
        writer.add_image('{} Diff Outputs {}'.format(prefix, whole_suffix), diff_to_show, n_iter)
        if mask is not None:
            mask_to_show = tensor2array(mask[0, j], max_value=1, colormap='bone')
            writer.add_image('{} Exp mask Outputs {}'.format(prefix, whole_suffix), mask_to_show, n_iter)


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor[tensor < np.inf].max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        norm_array[norm_array == np.inf] = np.nan
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)[:3]

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    return array


def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix, filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix, filename), save_path/'{}_model_best.pth.tar'.format(prefix))
