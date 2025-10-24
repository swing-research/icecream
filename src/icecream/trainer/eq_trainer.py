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
            wedge_full = self.initialize_wedge(self.angle_max_set[i], self.angle_min_set[i], self.wedge_size).cpu()
            self.wedge_full_set.append(wedge_full)
            wedge_input = self.get_real_binary_filter(wedge_full[:-1, :-1, :-1])
            self.wedge_input_set.append(wedge_input)
            wedge_ref = self.initialize_wedge(self.angle_max_set[i], self.angle_min_set[i], self.crop_size_eq).cpu()[
                        :-1, :-1, :-1]
            wedge_ref = self.get_real_binary_filter(wedge_ref)
            self.wedge_ref_set.append(wedge_ref)
        self.window = self.initialize_window(self.crop_size_eq)
        if self.configs.no_window:
            print('No window applied')
            self.window = None
            self.window_input = None



    def compute_loss(self, inp_1, inp_2, idx):
        est_1, est_2 = self.get_estimates(inp_1, inp_2)
        wedge_input = self.wedge_input_set[idx].to(self.device)
        wedge_ref = self.wedge_ref_set[idx].to(self.device)
        wedge_full = self.wedge_full_set[idx].to(self.device)

        obs_loss = fourier_loss(inp_2,est_1,
                                wedge_input,
                                self.criteria,
                                use_fourier=self.configs.use_fourier,
                                view_as_real=self.configs.view_as_real,
                                window=self.window_input) + fourier_loss(inp_1,
                                                        est_2, 
                                                        wedge_input,
                                                        self.criteria,
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
        # Apply wedge to rotated estimates
        est_1_rot_inp = get_measurement(est_1_rot, wedge_input)
        est_2_rot_inp = get_measurement(est_2_rot, wedge_input)
        est_1_rot_est, est_2_rot_est = self.get_estimates(est_1_rot_inp, est_2_rot_inp)
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
        loss = obs_loss + equi_loss_est
        equi_loss = equi_loss_est

        with torch.no_grad():
            self.loss_set.append(loss.item())
            self.obs_loss_set.append(obs_loss.item())
            self.equi_loss_set.append(equi_loss.item())
            diff_loss = torch.mean(torch.abs(est_1 - est_2))
            self.diff_loss_set.append(diff_loss.item())
        return loss

