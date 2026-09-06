"""
Class to train the model using the standard equivariant loss function with  with loss on rotated crops.
"""


import torch
from .base_trainer import BaseTrainer
from icecream.utils.utils import batch_rot_4vol, batch_rot_wedge_full_4vols,crop_vol,fourier_loss, fourier_loss_batch,get_measurement,get_measurement_multi_wedge

class EquivariantTrainer(BaseTrainer):
    """
    Trainer for equivariant models using the standard equivariant loss function with loss on rotated crops.
    """

    def setup(self):
        """
        Setup method for the EquivariantTrainer.
        Initializes the wedge and window for the model.
        """
        self.crop_size = int(self.configs.crop_size)
        self.crop_size_eq = self.configs.crop_size
        self.window_input = self.initialize_window(self.crop_size)
        if self.configs.wedge_double_size:
            self.wedge_size = self.crop_size * 2
        else:
            self.wedge_size = self.crop_size

        self.wedge_full_set = []
        self.wedge_input_set = []
        self.wedge_ref_set = []
        for i in range(len(self.angle_max_set)):
            if self.load_device:
                wedge_full = self.initialize_wedge(self.angle_max_set[i], 
                                                   self.angle_min_set[i], 
                                                   self.wedge_size).to(self.device)
            else:
                wedge_full = self.initialize_wedge(self.angle_max_set[i], self.angle_min_set[i], self.wedge_size).cpu()
            self.wedge_full_set.append(wedge_full)
            wedge_input = self.get_real_binary_filter(wedge_full[:-1, :-1, :-1])
            self.wedge_input_set.append(wedge_input)
            wedge_ref = self.initialize_wedge(self.angle_max_set[i], 
                                              self.angle_min_set[i], 
                                              self.crop_size_eq,
                                              wedge_support=self.configs.ref_wedge_support)[:-1, :-1, :-1]
            
            wedge_ref = self.get_real_binary_filter(wedge_ref)

            if self.load_device:
                wedge_ref = wedge_ref.to(self.device)
            else:
                wedge_ref = wedge_ref.cpu()
            wedge_ref = self.get_real_binary_filter(wedge_ref)
            self.wedge_ref_set.append(wedge_ref)
        # Device-resident copies for compute_loss, made once. The lists above stay where they
        # are because the dataset (MultiVolume) reads them.
        self.wedge_input_dev = [w.to(self.device) for w in self.wedge_input_set]
        self.wedge_ref_dev = [w.to(self.device) for w in self.wedge_ref_set]
        self.wedge_full_dev = [w.to(self.device) for w in self.wedge_full_set]
        self.window = self.initialize_window(self.crop_size_eq)
        if self.configs.no_window:
            print('No window applied')
            self.window = None
            self.window_input = None



    def compute_loss(self, inp_1, inp_2, idx):
        est_1, est_2 = self.get_estimates(inp_1, inp_2)
        wedge_input = self.wedge_input_dev[idx]
        wedge_ref = self.wedge_ref_dev[idx]
        wedge_full = self.wedge_full_dev[idx]
        # eq_single_direction: keep one Noise2Noise direction per step, doubled so the expected loss is unchanged.
        # The s=1 direction is the mirror of s=0, so swapping the halves lets the code below only compute s=0.
        one_dir = self.configs.eq_single_direction
        scale = 2.0 if one_dir else 1.0
        if one_dir and torch.rand(1).item() < 0.5:
            inp_1, inp_2, est_1, est_2 = inp_2, inp_1, est_2, est_1

        obs_loss = fourier_loss(inp_2, est_1, wedge_input, self.criteria,
                                use_fourier=self.configs.use_fourier,
                                view_as_real=self.configs.view_as_real,
                                window=self.window_input)
        if one_dir:
            obs_loss = scale * obs_loss
        else:
            obs_loss = obs_loss + fourier_loss(inp_1, est_2, wedge_input, self.criteria,
                                               use_fourier=self.configs.use_fourier,
                                               view_as_real=self.configs.view_as_real,
                                               window=self.window_input)
        est_1_rot, est_2_rot, inp_1_rot, inp_2_rot, wedge_rot = batch_rot_4vol(est_1,est_2, inp_1, inp_2, k_sets=self.k_sets,wedge=wedge_full)
        wedge_rot = wedge_rot[:,:-1,:-1,:-1]  # remove last row, column and slice to make it odd sized
        # make the rotated wedge positive semidefinite
        wedge_rot = self.get_real_binary_filters_batch(wedge_rot)
        est_1_ref = est_1_rot.clone()
        est_2_ref = est_2_rot.clone()
        est_1_ref = get_measurement(est_1_ref, wedge_ref)
        est_2_ref = get_measurement(est_2_ref, wedge_ref)
        # eq_stop_grad_inner: the first-pass output is a constant input to the second pass.
        if self.configs.eq_stop_grad_inner:
            est_1_rot, est_2_rot = est_1_rot.detach(), est_2_rot.detach()
        # Apply wedge to rotated estimates and run the second pass (once when one_dir)
        est_1_rot_inp = get_measurement(est_1_rot, wedge_input)
        if one_dir:
            est_1_rot_est, = self.get_estimates(est_1_rot_inp)
            pairs = [(est_2_ref, est_1_rot_est)]
        else:
            est_2_rot_inp = get_measurement(est_2_rot, wedge_input)
            est_1_rot_est, est_2_rot_est = self.get_estimates(est_1_rot_inp, est_2_rot_inp)
            pairs = [(est_2_ref, est_1_rot_est), (est_1_ref, est_2_rot_est)]
        if self.configs.eq_use_direct:
            equi_loss_est = sum(self.criteria(ref, est) for ref, est in pairs) * scale * self.configs.eq_weight
        else:
            equi_loss_est = sum(fourier_loss_batch(ref, est, wedge_rot, self.criteria,
                                                   use_fourier=self.configs.use_fourier,
                                                   view_as_real=self.configs.view_as_real,
                                                   window=self.window) for ref, est in pairs) * scale * self.configs.eq_weight
        loss = obs_loss + equi_loss_est
        equi_loss = equi_loss_est

        #print(loss.item(), obs_loss.item(), equi_loss.item())

        # Keep the scalars on the device; compute_average_loss reads them back in one sync.
        with torch.no_grad():
            self.loss_set.append(loss.detach())
            self.obs_loss_set.append(obs_loss.detach())
            self.equi_loss_set.append(equi_loss.detach())
            self.diff_loss_set.append(torch.mean(torch.abs(est_1 - est_2)))
        return loss

