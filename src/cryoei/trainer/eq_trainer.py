"""
Class to train the model using the standard equivariant loss function with  with loss on rotated crops.
"""


import torch
import numpy as np
from math import sqrt
from trainer.base_trainer import BaseTrainer
from utils.utils import batch_rot_4vol, batch_rot_wedge_full_4vols,crop_vol,fourier_loss, fourier_loss_batch,get_measurement,get_measurement_multi_wedge
from utils.utils import symmetrize_3D

class EquivariantTrainer(BaseTrainer):
    """
    Trainer for equivariant models using the standard equivariant loss function with loss on rotated crops.
    """

    def setup(self):
        """
        Setup method for the EquivariantTrainer.
        Initializes the wedge and window for the model.
        """
        self.crop_size = int(self.configs.input_crop_size)
        self.crop_size_eq = self.configs.input_crop_size
        self.window_input = self.initialize_window(self.crop_size)


        # make windows 1 

            
        if self.configs.wedge_double_size:
            self.wedge_size = self.crop_size*2
        else:
            self.wedge_size = self.crop_size

        self.wedge_full = self.initialize_wedge(self.wedge_size)


        self.wedge_input = self.wedge_full[:-1,:-1,:-1]  # remove last row, column and slice to make it odd sized

        self.wedge_input = self.get_real_binary_filter(self.wedge_input)


        




        self.window = self.initialize_window(self.crop_size_eq)
        self.window_n2n = self.initialize_window(self.crop_size_eq)

        self.wedge_ref = self.initialize_wedge(self.crop_size_eq, 
                                               wedge_support=self.configs.ref_wedge_support)[:-1,:-1,:-1]
        self.wedge_ref = self.get_real_binary_filter(self.wedge_ref)
        
        self.gaussian_window = self.gaussian_3d_window(self.crop_size_eq, 
                                                       sigma=self.configs.sigma_gaussian_window).to(self.device)


        #self.window = None

        if self.configs.no_window:
            print('No window applied')
            self.window = None
            self.window_input = None

        if self.configs.no_n2n_window:
            print('No N2N window applied')
            self.window_n2n = None

        theta = torch.zeros(1,3,4)
        theta[:,:,:3] = torch.eye(3)
        self.grid = torch.nn.functional.affine_grid(theta, (1,1,self.crop_size,
                                                            self.crop_size,
                                                            self.crop_size)).to(self.device)
        
    # def merge_crops(self,inp_small,inp_large):

    #     """
    #     Merges the small input crop into the large input crop using the merge window.
    #     """

    #     inp_large = inp_large.clone()

    #     if len(inp_small.shape) == 4:
    #         window_used = self.window_n2n[None]
    #     elif len(inp_small.shape) == 3:
    #         window_used = self.window_n2n
    #     if self.configs.merge_use_window:
    #         insert_crop = crop_vol(inp_large,self.crop_size_eq)*(1-window_used) + inp_small*window_used
    #     else:
    #         insert_crop =  inp_small 

    #     if len(inp_small.shape) == 4:
    #         inp_large[:,self.crop_size//2-self.crop_size_eq//2:self.crop_size//2+self.crop_size_eq//2,
    #                 self.crop_size//2-self.crop_size_eq//2:self.crop_size//2+self.crop_size_eq//2,
    #                 self.crop_size//2-self.crop_size_eq//2:self.crop_size//2+self.crop_size_eq//2] =  insert_crop
    #     elif len(inp_small.shape) == 3:
    #         inp_large[self.crop_size//2-self.crop_size_eq//2:self.crop_size//2+self.crop_size_eq//2,
    #                 self.crop_size//2-self.crop_size_eq//2:self.crop_size//2+self.crop_size_eq//2,
    #                 self.crop_size//2-self.crop_size_eq//2:self.crop_size//2+self.crop_size_eq//2] =  insert_crop
    #     else:
    #         raise ValueError("Input shape not supported for merging crops.")
    #     return inp_large
    
    def get_estimates(self, inp_1, inp_2):
        """
        Computes the estimates for the input crops inp_1 and inp_2.
        """
        if self.configs.use_mixed_precision:
            with self.autocast():
                est_1 = self.model(inp_1[:,None])[:,0]
                est_2 = self.model(inp_2[:,None])[:,0]
            est_1 = est_1.float()
            est_2 = est_2.float()
        else:
            est_1 = self.model(inp_1[:,None])[:,0]
            est_2 = self.model(inp_2[:,None])[:,0]

        return est_1, est_2
    


    def directional_tv_loss_iso(self, inp):
        """
        Computes the isotropic total variation loss for the input.
        """
        if len(inp.shape) == 3:
            inp = inp[:,None]
        if self.configs.tv_dim ==1:
            dval = torch.abs(inp[:,:,:,:-1] - inp[:,:,:,1:])**2
        elif self.configs.tv_dim == 2:
            dx = torch.abs(inp[:,:,:-1,:] - inp[:,:,1:,:])**2 
            dy = torch.abs(inp[:,:,:,:-1] - inp[:,:,:,1:])**2
            dval = dx[:,:,:,:-1] + dy[:,:,:-1]
        elif self.configs.tv_dim == 3:
            dx = torch.abs(inp[:,:,:-1,:] - inp[:,:,1:,:])**2 
            dy = torch.abs(inp[:,:,:,:-1] - inp[:,:,:,1:])**2
            dz = torch.abs(inp[:,:-1,:,:] - inp[:,1:,:,:])**2
            dval = dx[:,:-1,:,:-1] + dy[:,:-1,:-1] + dz[:,:,:-1,:-1]
        else:
            raise ValueError("Invalid tv_dim value. Must be 1, 2, or 3.")
        return torch.mean(torch.sqrt(dval + 1e-8))

    def gaussian_3d_window(self, size, sigma=1.0):
        """
        Create a 3D Gaussian window.
        
        Args:
            size (int): Size of the window (size x size x size).
            sigma (float): Standard deviation of the Gaussian.
        
        Returns:
            torch.Tensor: 3D Gaussian window.
        """
        x = torch.linspace(-1, 1, size+1)[:size]  # Ensure size is correct
        y = torch.linspace(-1, 1, size+1)[:size]
        z = torch.linspace(-1, 1, size+1)[:size]
        x, y, z = torch.meshgrid(x, y, z, indexing='ij')
        gauss = torch.exp(-((x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))) 

        gauss /= gauss.max()  # Normalize the window
        return gauss  # Add a channel dimension

    def compute_loss(self,inp_1,inp_2):


        # Flip the eq_use_direct flag if the current epoch is in the eq_use_direct_flips list

        if self.configs.use_flips_tr:
            if self.current_epoch in self.configs.eq_use_direct_flips:
                self.configs.eq_use_direct = not self.configs.eq_use_direct

        est_1, est_2 = self.get_estimates(inp_1, inp_2)


        if self.configs.use_gaussian_window:
            est_1 = get_measurement(est_1, self.gaussian_window)
            est_2 = get_measurement(est_2, self.gaussian_window)
            inp_1 = get_measurement(inp_1, self.gaussian_window)
            inp_2 = get_measurement(inp_2, self.gaussian_window)


        obs_loss = fourier_loss(inp_2,est_1,
                                self.wedge_input, 
                                self.criteria,
                                use_fourier=self.configs.use_fourier,
                                view_as_real=self.configs.view_as_real,
                                window=self.window_input) + fourier_loss(inp_1,
                                                        est_2, 
                                                        self.wedge_input, 
                                                        self.criteria,
                                                        use_fourier=self.configs.use_fourier,
                                                        view_as_real=self.configs.view_as_real,
                                                        window=self.window_input)


        if self.configs.detach_estimates:
            est_1 = est_1.detach()
            est_2 = est_2.detach()
            
        est_1_rot, est_2_rot, inp_1_rot, inp_2_rot, wedge_rot = batch_rot_4vol(est_1,est_2, inp_1, inp_2, k_sets=self.k_sets,wedge=self.wedge_full)

        wedge_rot = wedge_rot[:,:-1,:-1,:-1]  # remove last row, column and slice to make it odd sized

        if self.configs.eq_real_correction:
            wedge_rot = self.get_real_binary_filters_batch(wedge_rot)




        if self.configs.use_rotated_obs_loss:
            obs_rot_loss = fourier_loss_batch(inp_2_rot, est_1_rot,
                            wedge_rot, 
                            self.criteria,
                            use_fourier=self.configs.use_fourier,
                            view_as_real=self.configs.view_as_real,
                            window=self.window) + fourier_loss_batch(inp_1_rot,
                                                    est_2_rot,
                                                    wedge_rot,
                                                    self.criteria,
                                                    use_fourier=self.configs.use_fourier,
                                                    view_as_real=self.configs.view_as_real,
                                                    window=self.window)
            
            obs_loss = obs_loss + obs_rot_loss*self.configs.obs_rot_scale
        else:
            obs_rot_loss = 0



        est_1_ref = est_1_rot.clone()
        est_2_ref = est_2_rot.clone()

        if self.configs.detach_reference:
            est_1_ref = est_1_ref.detach()
            est_2_ref = est_2_ref.detach()

        est_1_ref = get_measurement(est_1_ref, self.wedge_ref)
        est_2_ref = get_measurement(est_2_ref, self.wedge_ref)


        if self.current_epoch < self.configs.miss_wedge_delay:
            est_1_rot_inp = est_1_rot.clone()
            est_2_rot_inp = est_2_rot.clone()
        else:
            # if we are past the miss wedge delay, we use the reference wedge for the rotated
            est_1_rot_inp = get_measurement(est_1_rot, self.wedge_input)
            est_2_rot_inp = get_measurement(est_2_rot, self.wedge_input)


        est_1_rot_est, est_2_rot_est = self.get_estimates(est_1_rot_inp, est_2_rot_inp)


        if self.configs.use_gaussian_window:
            est_1_rot_est = get_measurement(est_1_rot_est, self.gaussian_window)
            est_2_rot_est = get_measurement(est_2_rot_est, self.gaussian_window)
            est_1_ref = get_measurement(est_1_ref, self.gaussian_window)
            est_2_ref = get_measurement(est_2_ref, self.gaussian_window)


        if self.configs.n2n_eq:
            if self.configs.eq_use_direct:
                equi_loss_est = (self.criteria(est_2_ref, est_1_rot_est) + self.criteria(est_1_ref, est_2_rot_est))* self.configs.scale
            else:
                equi_loss_est = (fourier_loss_batch(est_2_ref,est_1_rot_est,
                                wedge_rot, 
                                self.criteria,
                                use_fourier=self.configs.use_fourier,
                                view_as_real=self.configs.view_as_real,
                                window=self.window) + fourier_loss_batch(est_1_ref,
                                                        est_2_rot_est, 
                                                        wedge_rot, 
                                                        self.criteria,
                                                        use_fourier=self.configs.use_fourier,
                                                        view_as_real=self.configs.view_as_real,
                                                        window=self.window))*self.configs.scale
        else:
            equi_loss_est = 0      


        if self.configs.use_rotated_inp_reference:
            equi_loss_inp = (fourier_loss_batch(inp_2_rot,est_1_rot_est,
                            wedge_rot, 
                            self.criteria,
                            use_fourier=self.configs.use_fourier,
                            view_as_real=self.configs.view_as_real,
                            window=self.window) + fourier_loss_batch(inp_1_rot,
                                                    est_2_rot_est, 
                                                    wedge_rot, 
                                                    self.criteria,
                                                    use_fourier=self.configs.use_fourier,
                                                    view_as_real=self.configs.view_as_real,
                                                    window=self.window))*self.configs.scale_inp              
        else:
            equi_loss_inp = 0

        loss = obs_loss + equi_loss_est + equi_loss_inp

        if self.configs.use_directional_tv_loss:
            tv_loss = self.directional_tv_loss_iso(est_1) + self.directional_tv_loss_iso(est_2)
            loss += tv_loss * self.configs.tv_scale

        equi_loss = equi_loss_est + equi_loss_inp

        with torch.no_grad():
            self.loss_set.append(loss.item())
            self.obs_loss_set.append(obs_loss.item())
            self.equi_loss_set.append(equi_loss.item())
            diff_loss = torch.mean(torch.abs(est_1 - est_2))
            self.diff_loss_set.append(diff_loss.item())
        return loss
    