import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

def depth_transform(depth, intrinc, RT):
    b, c, h, w = depth.shape
    x = torch.arange(0, w, device=depth.device).float()
    y = torch.arange(0, h, device=depth.device).float()
    yy, xx = torch.meshgrid(y, x)
    xx = (xx.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    yy = (yy.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    ones_tensor = torch.ones_like(xx, device=depth.device)
    xyz = torch.cat((xx, yy, ones_tensor), dim=3).reshape((b, h * w, 3))
    dd = depth.reshape((b, h * w, 1))

    center = torch.cat((intrinc[:, 0, 2].unsqueeze(1), intrinc[:, 1, 2].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    focal = torch.cat((intrinc[:, 0, 0].unsqueeze(1), intrinc[:, 1, 1].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    xyz[:, :, :2] = (xyz[:, :, :2] - center) / focal
    xyz = (xyz * dd).transpose(1, 2)

    R12 = RT[:, :3, :3]
    t12 = RT[:, :3, 3]
    xyz = torch.matmul(R12, xyz) + t12.reshape((-1, 3, 1))
    #     xy = xyz[:,:2]/xyz[:,2].reshape((b, 1, -1))

    dd_transform = xyz[:, 2].reshape((b, 1, h, w))
    return dd_transform  # , xyz

def bilinear_sample_function(xyz, color, padding_mode, mode="bilinear"):
    b, c, h, w = color.shape[:4]

    xx = xyz[:, :, 0].unsqueeze(-1)
    yy = xyz[:, :, 1].unsqueeze(-1)

    xx_norm = xx / (w - 1) * 2 - 1
    yy_norm = yy / (h - 1) * 2 - 1
    xx_mask = ((xx_norm > 1) + (xx_norm < -1)).detach()
    yy_mask = ((yy_norm > 1) + (yy_norm < -1)).detach()

    if padding_mode == 'zeros':
        xx_norm[xx_mask] = 2
        yy_norm[yy_mask] = 2
    # mask = ((xx_norm > 1) + (xx_norm < -1) + (yy_norm < -1) + (yy_norm > 1)).detach().squeeze()
    # mask = mask.unsqueeze(1).expand(b, 3, h * w)

    pixel_coords = torch.stack([xx_norm, yy_norm], dim=2).reshape((b, h, w, 2))  # [B, H*W, 2]
    color_pred = F.grid_sample(color, pixel_coords, mode=mode, padding_mode=padding_mode)

    return color_pred

def warp(depth_tgt, color_src, intrinc_src, intrinc_tgt, extrinc_src, extrinc_tgt):
    '''
    :param depth_tgt: B*1*H*W
    :param color_src: B*3*H*W
    :param intrinc_src: B*3*3
    :param intrinc_tgt: B*3*3
    :param extrinc_src: B*4*4
    :param extrinc_tgt: B*4*4
    :return: color_tgt
        mask_tgt
    '''
    depth_tgt = depth_tgt.permute(0, 2, 3, 1).contiguous()
    b, h, w = depth_tgt.shape[:3]

    x = torch.arange(0, w, device=depth_tgt.device).float()
    y = torch.arange(0, h, device=depth_tgt.device).float()
    yy, xx = torch.meshgrid(y, x)
    xx = (xx.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    yy = (yy.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    ones_tensor = torch.ones_like(xx, device=depth_tgt.device)
    xyz = torch.cat((xx, yy, ones_tensor), dim=3).reshape((b, h * w, 3))
    dd = depth_tgt.reshape((b, h * w, 1))

    center = torch.cat((intrinc_tgt[:, 0, 2].unsqueeze(1), intrinc_tgt[:, 1, 2].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    focal = torch.cat((intrinc_tgt[:, 0, 0].unsqueeze(1), intrinc_tgt[:, 1, 1].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    xyz[:, :, :2] = (xyz[:, :, :2] - center) / focal
    xyz = (xyz * dd).transpose(1, 2)

    #     R_src_inv = (extrinc_src[:, :3, :3]).transpose(1, 2).reshape((b, 3, 3))
    #     R_tgt = extrinc_tgt[:, :3, :3].reshape((b, 3, 3))
    #     t_src = extrinc_src[:, :3, 3].reshape((b, 3, 1))
    #     t_tgt = extrinc_tgt[:, :3, 3].reshape((b, 3, 1))
    #     R12 = torch.matmul(R_src_inv, R_tgt)
    #     t12 = torch.matmul(R_src_inv, t_tgt) - torch.matmul(R_src_inv, t_src)
    # xyz_copy = copy.deepcopy(xyz)
    # Rm = extrinc_tgt[:,:3,:3]
    # tm = extrinc_tgt[:,:3, 3]
    # xyz_wrd = torch.matmul(Rm, xyz_copy) + tm.reshape((-1, 3, 1))

    RT = torch.matmul(extrinc_src.inverse(), extrinc_tgt)
    R12 = RT[:, :3, :3]
    t12 = RT[:, :3, 3]
    xyz = torch.matmul(R12, xyz) + t12.reshape((-1, 3, 1))

    eps = 2.220446049250313e-16

    xyz = torch.matmul(intrinc_src, xyz)
    xyz_wrd = copy.deepcopy(xyz)
    xyz = (xyz / (xyz[:, 2, :].unsqueeze(1) + eps )).transpose(1, 2)
    # print(xyz)

    color_tgt = bilinear_sample_function(xyz, color_src, 'border')

    return color_tgt

def prepare_warp_and_occ(depth_tgt, color_src, intrinc_src, intrinc_tgt, extrinc_src, extrinc_tgt,
                        occ_thres=0.002, return_xyz_img_global=False):
    '''
    :param depth_tgt: B*1*H*W
    :param color_src: B*C*H*W # anything rgb, depth, normal.. here will be depth_src
    :param intrinc_src: B*3*3
    :param intrinc_tgt: B*3*3
    :param extrinc_src: B*4*4 # cv
    :param extrinc_tgt: B*4*4
    :return: 1. xyz, xyzs in src view, for convenience
             2. occ region
    '''
    depth_tgt = depth_tgt.permute(0, 2, 3, 1).contiguous()
    b, h, w = depth_tgt.shape[:3]
    zero_mask_tgt = depth_tgt == 0 # B, H, W, 1 bool

    x = torch.arange(0, w, device=depth_tgt.device).float()
    y = torch.arange(0, h, device=depth_tgt.device).float()
    yy, xx = torch.meshgrid(y, x)
    xx = (xx.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    yy = (yy.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    ones_tensor = torch.ones_like(xx, device=depth_tgt.device)
    xyz = torch.cat((xx, yy, ones_tensor), dim=3).reshape((b, h * w, 3))
    dd = depth_tgt.reshape((b, h * w, 1))

    center = torch.cat((intrinc_tgt[:, 0, 2].unsqueeze(1), intrinc_tgt[:, 1, 2].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    focal = torch.cat((intrinc_tgt[:, 0, 0].unsqueeze(1), intrinc_tgt[:, 1, 1].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    xyz[:, :, :2] = (xyz[:, :, :2] - center) / focal
    xyz = (xyz * dd).transpose(1, 2)

    if return_xyz_img_global:
        RT_g = extrinc_tgt
        R12_g = RT_g[:, :3, :3]
        t12_g = RT_g[:, :3, 3]
        xyz_g = torch.matmul(R12_g, xyz) + t12_g.reshape((-1, 3, 1))
        xyz_g = xyz_g.reshape(-1, 3, h, w)

    RT = torch.matmul(extrinc_src.inverse(), extrinc_tgt)
    R12 = RT[:, :3, :3]
    t12 = RT[:, :3, 3]
    xyz = torch.matmul(R12, xyz) + t12.reshape((-1, 3, 1))

    # here the xyz are in src transform
    # print(xyz.shape, b, h, w)
    z_depth = xyz[:,2:3,:].transpose(1, 2).reshape(b, h, w, 1)

    eps = 2.220446049250313e-16

    xyz = torch.matmul(intrinc_src, xyz)
    # xyz_wrd = copy.deepcopy(xyz)
    xyz = (xyz / (xyz[:, 2, :].unsqueeze(1) + eps )).transpose(1, 2)
    color_tgt = bilinear_sample_function(xyz, color_src, 'border')

    warped_depth = color_tgt.permute(0,2,3,1) # b, c, h, w
    occ = torch.abs(z_depth - warped_depth) > occ_thres

    if return_xyz_img_global:
        return xyz, occ, z_depth, warped_depth, xyz_g
    return xyz, occ, z_depth, warped_depth


def warp_single_np(depth_tgt, color_src, intrinc_src, intrinc_tgt, extrinc_src, extrinc_tgt):
    '''
    :param depth_tgt: H*W -> B*1*H*W
    :param color_src: H*W*3 -> B*3*H*W
    :param intrinc_src: 3*3 -> B*3*3
    :param intrinc_tgt: B*3*3
    :param extrinc_src: 4*4 -> B*4*4, c2w
    :param extrinc_tgt: 4*4 -> B*4*4, c2w
    :return: color_tgt
        mask_tgt
    '''
    depth_tgt = torch.tensor(depth_tgt).unsqueeze(0).unsqueeze(0)
    color_src = torch.tensor(color_src).permute(2,0,1).unsqueeze(0)
    intrinc_src = torch.tensor(intrinc_src).unsqueeze(0)
    intrinc_tgt = torch.tensor(intrinc_tgt).unsqueeze(0)
    extrinc_src = torch.tensor(extrinc_src).unsqueeze(0)
    extrinc_tgt = torch.tensor(extrinc_tgt).unsqueeze(0)
    color_tgt, xyz_wrd, pixel_coords = warp(depth_tgt.float(), color_src.float(), intrinc_src.float(), intrinc_tgt.float(),
                            extrinc_src.float(), extrinc_tgt.float())
    # 1, 3, H, W -> H, W, 3
    return color_tgt.squeeze(0).permute(1,2,0).numpy()

def blend_image(self, images, depths, intrincs, extrincs, cosmaps=None):

    color_srcL = images[:, 1]
    color_srcR = images[:, 2]
    depth_srcL = depths[:, 1]
    depth_srcR = depths[:, 2]
    # print(color_srcL.shape)
    # print(depth_srcL.shape)
    colordepth_srcL = torch.cat([color_srcL, depth_srcL],dim=1)
    colordepth_srcR = torch.cat([color_srcR, depth_srcR],dim=1)

    intrinc_srcL = intrincs[:, 1]
    intrinc_srcR = intrincs[:, 2]
    extrinc_srcL = extrincs[:, 1]
    extrinc_srcR = extrincs[:, 2]

    if not cosmaps is None:
        cosmap_srcL = cosmaps[:, 0]
        cosmap_srcR = cosmaps[:, 1]

    self.image_gt = images[:, 0]
    depth_tgt   = depths[:, 0]
    intrinc_tgt = intrincs[:, 0]
    extrinc_tgt = extrincs[:, 0]

    # print(color_srcL.shape)

    # print(colordepth_srcL.shape)
    # print(intrinc_srcL.shape)
    # print(extrinc_srcL.shape)
    # print(depth_tgt.shape)

    colordepth_wrapL = warp(depth_tgt.float(), colordepth_srcL.float(), intrinc_srcL.float(), intrinc_tgt.float(),
                            extrinc_srcL.float(), extrinc_tgt.float())
    colordepth_wrapR = warp(depth_tgt.float(), colordepth_srcR.float(), intrinc_srcR.float(), intrinc_tgt.float(),
                            extrinc_srcR.float(), extrinc_tgt.float())

    if not cosmaps is None:
        self.cosmap_wrapL = warp(depth_tgt.float(), cosmap_srcL.float(), intrinc_srcL.float(), intrinc_tgt.float(),
                            extrinc_srcL.float(), extrinc_tgt.float())[:, :1]
        self.cosmap_wrapR = warp(depth_tgt.float(), cosmap_srcR.float(), intrinc_srcR.float(), intrinc_tgt.float(),
                            extrinc_srcR.float(), extrinc_tgt.float())[:, :1]

    self.depth_diffL = torch.abs(depth_tgt - colordepth_wrapL[:, 3:]) / 3.0
    self.depth_diffR = torch.abs(depth_tgt - colordepth_wrapR[:, 3:]) / 3.0

    # print(colordepth_wrapL[:,:3].shape)
    # print(depth_diffL.shape)
    # self.colordiff_wrapL = torch.cat([colordepth_wrapL[:, :3], self.depth_diffL], dim=1)
    # self.colordiff_wrapR = 