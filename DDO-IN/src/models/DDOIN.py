import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output, display
from matplotlib import pyplot as plt
import os
from utils.data_utils import FFT, IFFT
from utils.losses import GradientEntropyLoss


network_config = {
    "otype": "CutLassMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 256,
    "n_hidden_layers": 1,
}

mot_network_config = {
    "otype": "FullyFusedMLP",
    "activation": "Tanh",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 1,
}

encoding_config = {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "fine_resolution": 320,
    "per_level_scale": 2,
    "interpolation": "Linear",
}


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


def create_mask(p, image_size):

    if p == 0:
        return np.ones(image_size, dtype=np.float32)
    patch_size = (p, p)
    mask = np.zeros(image_size, dtype=np.float32)
    for x in range(0, image_size[0], patch_size[0]):
        for y in range(0, image_size[1], patch_size[1]):
            patch_mask = np.zeros(patch_size, dtype=np.float32)
            if (x // patch_size[0] + y // patch_size[1]) % 2 == 1:
                patch_mask = np.ones(patch_size, dtype=np.float32)
            mask[x:x + patch_size[0], y:y + patch_size[1]] = patch_mask
    return mask


def make_grids(sizes, device="cpu"):
    dims = len(sizes)
    lisnapces = [torch.linspace(-1, 1, s, device=device) for s in sizes]
    mehses = torch.meshgrid(*lisnapces, indexing="ij")
    coords = torch.stack(mehses, dim=-1).view(-1, dims)
    return coords


class IMMoCo(nn.Module):
    def __init__(self, masks):
        super().__init__()

        self.image_inr = tcnn.NetworkWithInputEncoding(
            2, 2, encoding_config, network_config
        )
        self.motion_inr = tcnn.NetworkWithInputEncoding(
            3, 2, encoding_config, mot_network_config
        )

        self.masks = masks
        self.num_movements, self.x, self.num_lines = masks.shape

        self.device = masks.device

        self.identy_grid = F.affine_grid(
            torch.eye(2, 3, device=self.masks.device).unsqueeze(0),
            torch.Size((1, 1, self.x, self.num_lines)),
            align_corners=True,
        )

        self.input_grid = make_grids(
            (self.num_movements, self.x, self.num_lines), device=self.device
        )

    def forward(self):
        image_prior = (
            self.image_inr(self.identy_grid.view(-1, 2))
            .float()
            .view(self.x, self.num_lines, 2)
        )
        image_prior = image_prior[..., 0] + 1j * image_prior[..., 1]

        images = image_prior.squeeze().unsqueeze(0).repeat(self.num_movements, 1, 1)

        grids = self.motion_inr(self.input_grid).float().tanh().view(
            self.num_movements, self.x, self.num_lines, 2
        ) + self.identy_grid.view(1, self.x, self.num_lines, 2)

        motion_images = torch.view_as_complex(
            F.grid_sample(
                torch.view_as_real(images).permute(0, 3, 1, 2),
                grids,
                mode="bilinear",
                align_corners=False,
                padding_mode="zeros",
            )
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        kspace_out = (FFT(image_prior).squeeze() * (1 - self.masks.sum(0)).float()) + (
                FFT(motion_images) * self.masks.float()
        ).sum(0)
 
     
        return kspace_out, image_prior

import torch
import torch.nn.functional as F

def adaptive_weighting_loss_complex(output, target, mask,iters,device):
    """
The weighted loss calculation of complex images makes the model focus on the artifact region and consider both real and imaginary parts.

Parameters:
output (torch.Tensor): Model output image (plural)
target (torch.Tensor): true image (plural)
mask (torch.Tensor): binary mask, representing the artifact region (1 represents the artifact, 0 represents the normal region)

Back:
torch.Tensor: Weighted loss
    """

    output_real, output_imag = torch.view_as_real(output).unbind(-1)
    target_real, target_imag = torch.view_as_real(target).unbind(-1)
    output_real_img, output_imag_img = torch.view_as_real(compose_image_freq).unbind(-1)
    target_real_img, target_imag_img = torch.view_as_real(ori_image_reconstructed).unbind(-1)

    real_loss = (output_real - target_real) ** 2
    imag_loss = (output_imag - target_imag) ** 2
    real_loss_img = (output_real_img - target_real_img) ** 2
    imag_loss_img = (output_imag_img - target_imag_img) ** 2
    # print("real_loss",real_loss)
    # print(iters)

    # loss_img = real_loss_img + imag_loss_img
    # print("loss_img",loss_img)
    
    loss = real_loss + imag_loss
    # print("loss_img",loss_img)
    weights=0.5-((iters+1)/400)
    # weights_combine = 1-((iters+1)/800)

    weighted_loss = weights*(loss * mask) + (1 - weights)*(loss * (1- mask))
    # print("weighted_loss_before",weighted_loss)

    
    # weighted_loss = weighted_loss * weights_combine + loss_img * (1 - weights_combine)
    
    # print("weighted_loss",weighted_loss)
    weighted_loss = weighted_loss.to(device)

    return torch.mean(weighted_loss).to(device)

def save_images(low_freq, high_freq_keep, high_freq_change, kspace_foward_model, gt_k_space,mask_js,x_re_freq, output_dir="./saved_images"):

    os.makedirs(output_dir, exist_ok=True)
    low_freq_copy = low_freq.clone().cpu().detach().numpy()
    high_freq_keep_copy = high_freq_keep.clone().cpu().detach().numpy()
    high_freq_change_copy = high_freq_change.clone().cpu().detach().numpy()
    kspace_foward_model_copy = kspace_foward_model.clone().cpu().detach().numpy()
    gt_k_space_copy = gt_k_space.clone().cpu().detach().numpy()
    mask_js_copy = mask_js.clone().cpu().detach().numpy()
    x_re_freq_copy = x_re_freq.clone().cpu().detach().numpy()


    x_re_freq_copy = np.abs(x_re_freq_copy) 
    plt.imshow(np.log(x_re_freq_copy + 0.01), cmap='gray') 
    plt.title("fre_combine")
    plt.axis('off')
    plt.savefig(f"{output_dir}/fre_combine.png")
    plt.close()
    
    mask_js_copy = np.abs(mask_js_copy)  
    plt.imshow(np.log(mask_js_copy + 1e-10), cmap='gray') 
    plt.title("mask")
    plt.axis('off')
    plt.savefig(f"{output_dir}/mask.png")
    plt.close()
    
    
    low_freq_copy = np.abs(low_freq_copy) 
    plt.imshow(np.log(low_freq_copy+ 1e-10), cmap='gray') 
    plt.title("Low Frequency Keep")
    plt.axis('off')
    plt.savefig(f"{output_dir}/low_freq_keep.png")
    plt.close()

    high_freq_keep_copy = np.abs(high_freq_keep_copy)  
    plt.imshow(np.log(high_freq_keep_copy+ 1e-10), cmap='gray') 
    plt.title("High Frequency Keep")
    plt.axis('off')
    plt.savefig(f"{output_dir}/high_freq_keep.png")
    plt.close()

 
    high_freq_change_copy = np.abs(high_freq_change_copy)  
    plt.imshow(np.log(high_freq_change_copy + 0.01), cmap='gray') 
    plt.title("High Frequency Change")
    plt.axis('off')
    plt.savefig(f"{output_dir}/high_freq_change.png")
    plt.close()


    kspace_foward_model_copy = np.abs(kspace_foward_model_copy)  
    plt.imshow(np.log(kspace_foward_model_copy + 1e-10), cmap='gray') 
    plt.title("K-space Forward Model")
    plt.axis('off')
    plt.savefig(f"{output_dir}/kspace_forward_model.png")
    plt.close()

   
    gt_k_space_copy = np.abs(gt_k_space_copy) 
    plt.imshow(np.log(gt_k_space_copy + 1e-10), cmap='gray') 
    plt.title("ground True")
    plt.axis('off')
    plt.savefig(f"{output_dir}/gt_model.png")
    plt.close()





def DDOIN_motion_correction(
        kspace_corr, masks, iters=200, learning_rate=1e-2, lambda_ge=1e-2, debug=False
):
    """_summary_

    Args:
        kspace_corr (_type_): _description_
        masks (_type_): _description_
        iters (int, optional): _description_. Defaults to 200.
        learning_rate (_type_, optional): _description_. Defaults to 1e-2.
        lambda_ge (_type_, optional): _description_. Defaults to 1e-2.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # with ClearCache():

    IMOCO = IMMoCo(masks)

    # data should be between 1 and 16000
    scale = kspace_corr.abs().max()

    kspace_motion_norm = kspace_corr.div(scale).mul(16000)

    kspace_input = kspace_motion_norm.clone().detach().cuda()

    if debug:
        print(f"Scale: {scale:.4f}")
        print(
            f"Kspace input: {kspace_input.abs().min().item():.4f}, {kspace_input.abs().max().item():.4f}"
        )

    optimizer = torch.optim.Adam(
        [
            {"params": IMOCO.motion_inr.parameters(), "lr": learning_rate},
            {"params": IMOCO.image_inr.parameters(), "lr": learning_rate},
        ]
    )

    if debug:
        stats = []
        summar_steps = 20
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[2].set_title("Loss")
        axs[2].set_xlabel("Iterations")
        axs[2].set_xlim(0, iters)

    for j in range(iters):

        optimizer.zero_grad()


        kspace_foward_model, image_prior = IMOCO()

            
        freq = torch.fft.fftfreq(kspace_foward_model.shape[0], d=1 / np.pi)
        freq = torch.fft.fftshift(freq) 
        

        
   
        keep = torch.abs(freq) < np.pi / 10
        change = torch.abs(freq) >= np.pi / 10


        device = kspace_input.device


        low_freq = torch.zeros_like(kspace_input, dtype=torch.complex64, device=kspace_input.device)
        high_freq_keep = torch.zeros_like(kspace_input, dtype=torch.complex64, device=kspace_input.device)
        high_freq_change = torch.zeros_like(kspace_input, dtype=torch.complex64, device=kspace_input.device)
        # output_dir="./saved_images"
        # os.makedirs(output_dir, exist_ok=True)
        # if(j==199):
        # #     low_freq_copy = low_freq.clone()
        # #     high_freq_keep_copy = high_freq_keep.clone()
        # #     high_freq_change_copy = high_freq_change.clone()    
        # #     kspace_input_copy = kspace_input.clone()  
            
            
        # #     plt.imshow(torch.abs(low_freq_copy).cpu().detach().numpy(), cmap='gray')
        # #     plt.title("Low Frequency Keep")
        # #     plt.axis('off')
        # #     plt.savefig(f"{output_dir}/low_freq_keep_mask.png")
        # #     plt.close()

        # #     # 
        # #     plt.imshow(torch.abs(high_freq_keep_copy).cpu().detach().numpy(), cmap='gray')
        # #     plt.title("High Frequency Keep")
        # #     plt.axis('off')
        # #     plt.savefig(f"{output_dir}/high_freq_keep_mask.png")
        # #     plt.close()

        # # # 
        # #     plt.imshow(torch.abs(high_freq_change_copy).cpu().detach().numpy(), cmap='gray')
        # #     plt.title("High Frequency Change")
        # #     plt.axis('off')
        # #     plt.savefig(f"{output_dir}/high_freq_change_mask.png")
        # #     plt.close() 


        # #  
        # #     min_val = torch.min(torch.abs(kspace_input_copy))
        # #     max_val = torch.max(torch.abs(kspace_input_copy))


        # #     kspace_input_copy_normalized = (torch.abs(kspace_input_copy) - min_val) / (max_val - min_val)


        # #     print("kspace_input_copy normalized", kspace_input_copy_normalized)


        # #     plt.imshow(kspace_input_copy_normalized.cpu().detach().numpy(), cmap='gray')
        # #     plt.title("High Frequency Change (Normalized)")
        # #     plt.axis('off')
        # #     plt.savefig(f"{output_dir}/high_freq_change_mask_normalized.png")
        # #     plt.close()


            




        #     plt.subplot(1, 1, 1)
        #     plt.imshow(torch.abs(image_prior).cpu().detach().numpy(), cmap='gray')
        #     plt.title("Spatial Domain Image")
        #     plt.axis('off')




        #     output_dir = "./saved_images" 
        #     plt.savefig(f"{output_dir}/comparison_spatial_frequency_log.png")
        #     plt.close()
 

        gt_k_space = kspace_input


        low_freq[keep] = kspace_foward_model[keep]
        high_freq_keep[change] = gt_k_space[change]
        high_freq_change[change] = kspace_foward_model[change]


        jt_mask = create_mask(8, kspace_input.shape)
        jt_mask = torch.tensor(jt_mask, dtype=torch.float32, device=kspace_input.device)
        jt_mask = jt_mask.to(device)

        ori_image_reconstructed = IFFT(kspace_input)
        image_reconstructed = IFFT(kspace_foward_model)
        # print("ori_image_reconstructed",ori_image_reconstructed)
        # print("image_reconstructed",image_reconstructed)

        if j % 2 == 1: 
            compose_image = ori_image_reconstructed * jt_mask + image_reconstructed * (1 - jt_mask)
        else:  
            compose_image = ori_image_reconstructed * (1 - jt_mask) + image_reconstructed * jt_mask

        compose_image_freq = FFT(compose_image)

   
        mask_js=masks.sum(0).float()
        x_re_freq = low_freq + high_freq_keep * mask_js + high_freq_change * (1 - mask_js)
        # x_re_freq = low_freq + high_freq_change * jt_mask + high_freq_change * (1 - jt_mask)
        # print("low_freq",low_freq.shape)
        # print("masks",masks.shape)

        x_re_freq = x_re_freq.to(kspace_input.device).requires_grad_()
        # if(j==100):
        #     # print("gt_k_space",gt_k_space)
        #     save_images(low_freq, high_freq_keep, high_freq_change, kspace_foward_model, kspace_input,mask_js,x_re_freq)
        
        # print(f'old: {kspace_foward_model}')
        # print(f'new: {x_re_freq}')
        num = 0.5 + (j / 400)
        final_space_img = x_re_freq * (1 - num) + compose_image_freq * num

        # pixel_loss = F.l1_loss(compose_image, ori_image_reconstructed)

        # print("final_space_img",final_space_img.shape)
        # print("kspace_input",kspace_input.shape)
        
        loss_inr = adaptive_weighting_loss_complex(x_re_freq, kspace_input, masks.sum(0).float(),j,device) + GradientEntropyLoss()(image_prior).mul(lambda_ge) 

        # loss_inr = adaptive_weighting_loss_complex(x_re_freq, kspace_input, masks.sum(0).float(),j,device,compose_image,ori_image_reconstructed) + GradientEntropyLoss()(image_prior).mul(lambda_ge) 
        
        # loss_inr = F.mse_loss(
        #     torch.view_as_real(kspace_foward_model), torch.view_as_real(kspace_input)
        # ) + GradientEntropyLoss()(image_prior).mul(lambda_ge) 
        loss_inr.backward()
        optimizer.step()

        if debug:
            stats.append(loss_inr.item())

        if j % (iters // 10) and j > (iters // 2):
            lambda_ge *= 0.5
        if debug:
            if j % summar_steps == 0 or j == 199:
                print(f"iter: {j}, DC_Loss: {loss_inr:.4f}")

                axs[0].imshow(image_prior.abs().detach().cpu().squeeze(), cmap="gray")
                axs[0].set_title("IM-MoCo Image")
                axs[0].set_axis_off()

                axs[1].imshow(
                    IFFT(kspace_foward_model.detach().cpu()).abs(), cmap="gray"
                )
                axs[1].set_title("Motion Forward")
                axs[1].set_axis_off()

                axs[2].plot(stats)

                plt.close()  ##could prevent memory leak?
                display(fig)
                clear_output(wait=True)

    # clear all gpu memory
    torch.cuda.empty_cache()
    del kspace_input, kspace_motion_norm, IMOCO, optimizer, loss_inr

    return image_prior, kspace_foward_model
