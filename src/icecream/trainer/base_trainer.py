"""
Base trainer class for training models using nois2noise type loss
"""

import os
import sys
import csv
import json
import torch
import mrcfile
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from icecream.utils.mask_util import make_mask
from icecream.dataset.multi_volumes import MultiVolume
from icecream.utils.inference_util import inference
from icecream.utils.utils import get_wedge_3d_new, symmetrize_3D, get_measurement, fourier_loss

class BaseTrainer:
    def __init__(self,
                 configs,
                 model,
                 angle_max_set=[60],
                 angle_min_set=[-60],
                 angles_set = None,
                 save_path='./',
                 ):

        raw_device = configs.device
        if isinstance(raw_device, int):
            device = "cpu" if raw_device == -1 else f"cuda:{raw_device}"
        else:
            device = str(raw_device)
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.save_path = save_path

        self.loss_set = []
        self.diff_loss_set = []
        self.loss_avg_set = []
        self.diff_loss_avg_set = []
        self.configs = configs

        self.load_device = configs.load_device

        self.criteria = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(configs.learning_rate))

        self.obs_loss_set = []
        self.obs_loss_avg_set = []
        self.equi_loss_set = []
        self.equi_loss_avg_set = []

        self.angle_max_set = angle_max_set
        self.angle_min_set = angle_min_set

        self.setup()

    def setup(self):
        self.crop_size = int(self.configs.crop_size)
        self.crop_size_eq = int(self.configs.crop_size)
        self.window = self.initialize_window(self.crop_size)
        # self.wedge_input = self.initialize_wedge(self.crop_size)
        self.wedge_input_set = []
        self.wedge_eq_set = []
        for i in range(len(self.angle_max_set)):
            wedge_input = self.initialize_wedge(self.angle_max_set[i],self.angle_min_set[i],self.crop_size).cpu()
            wedge_eq = self.initialize_wedge(self.angle_max_set[i],self.angle_min_set[i],self.crop_size_eq).cpu()
            self.wedge_input_set.append(wedge_input)
            self.wedge_eq_set.append(wedge_eq)

    def get_real_binary_filter(self, binary_filter):
        binary_filter_sym = symmetrize_3D(binary_filter)
        binary_filter_t = (binary_filter + binary_filter_sym) / 2
        binary_filter_t[binary_filter_t > 0.1] = 1

        return binary_filter_t

    def get_real_binary_filters_batch(self, binary_filters):
        """
        Applies the get_real_binary_filter to each filter in the batch.
        """
        binary_filters_sym = torch.zeros_like(binary_filters)
        for i in range(binary_filters.shape[0]):
            binary_filters_sym[i] = self.get_real_binary_filter(binary_filters[i])
        return binary_filters_sym

    def normalize_volume(self, vol):
        return (vol - vol.mean()) / (vol.std() + 1e-8)

    def initialize_wedge(self, angle_max, angle_min, crop_size):
        """
        Initialize the wedge for the model.
        This method should be overridden by subclasses if needed.
        """
        assert angle_min < angle_max, "angle_min should be less than angle_max"
        wedge,ball = get_wedge_3d_new(crop_size,
                                      max_angle = angle_max,
                                      min_angle = angle_min,
                                      rotation = 0,
                                      low_support=self.configs.wedge_low_support)
        wedge_t = torch.tensor(wedge, dtype=torch.float32, device=self.device)
        wedge_t_sym = symmetrize_3D(wedge_t)
        wedge_t = (wedge_t_sym + wedge_t)/2
        wedge_t[wedge_t>0.1]  = 1
        return  wedge_t

    def initialize_window(self, crop_size):
        w = np.zeros((crop_size, crop_size, crop_size))
        w[crop_size // 4:-crop_size // 4, crop_size // 4:-crop_size // 4, crop_size // 4:-crop_size // 4] = 1
        w_t = torch.tensor(w, dtype=torch.float32, device=self.device)
        return w_t

    def load_volume(self, vol_path):
        """
        Load a single volume from the given path to the cpu.
        Args:
            vol_path (str): Path to the volume file.
        Returns:
            torch.Tensor: Loaded volume as a tensor.
        """
        vol = mrcfile.open(vol_path).data
        vol = np.moveaxis(vol, 0, 2).astype(np.float32)
        vol_t = torch.tensor(vol, dtype=torch.float32, device='cpu')
        return self.normalize_volume(vol_t)

    def load_data(self, vol_paths_1, vol_paths_2, vol_mask_path=None, use_mask=False, mask_frac=0.3, mask_tomo_side=5, mask_tomo_density_perc=50., mask_tomo_std_perc=50.):
        """
        Load data from the given volume paths.
        Args:
            vol_paths_1 (list): List of paths to the first set of volumes.
            vol_paths_2 (list): List of paths to the second set of volumes.
            vol_mask_path (str, optional): Path to the volume mask. Defaults to None.
        """
        print(vol_paths_1)
        print(vol_paths_2)
        self.vol_paths_1 = vol_paths_1
        self.vol_paths_2 = vol_paths_2
        self.vol_mask_path = vol_mask_path

        self.n_volumes = len(self.vol_paths_1)

        if len(vol_paths_1) != len(vol_paths_2):
            raise ValueError("The number of volume paths for vol_paths_1 and vol_paths_2 must be the same.")

        # Load and store all the volume in a list on the CPU. Probably sub-optimal but enough at the moment.
        vol_1_set = []
        vol_2_set = []
        vol_mask_set = []
        for i in range(len(vol_paths_1)):
            vol_1_t = self.load_volume(vol_paths_1[i])
            vol_2_t = self.load_volume(vol_paths_2[i])
            print(f"Loaded volumes: \n {vol_paths_1[i]}\n and\n {vol_paths_2[i]}")

            print(f"They have shape (x,y,z): {list(vol_1_t.shape)} and {list(vol_2_t.shape)}.")
            if vol_mask_path is not None:
                vol_mask = mrcfile.open(vol_mask_path[i]).data
                vol_mask = np.moveaxis(vol_mask, 0, 2).astype(np.float32)
                vol_mask_t = torch.tensor(vol_mask, dtype=torch.float32, device='cpu')
            else:
                if use_mask:
                    vol_avg = ((vol_1_t + vol_2_t) / 2).numpy()
                    vol_mask = make_mask(vol_avg, mask_boundary=None, side=mask_tomo_side, density_percentage=mask_tomo_density_perc, std_percentage=mask_tomo_std_perc)
                    vol_mask_t = torch.tensor(vol_mask, dtype=torch.float32, device='cpu')
                else:
                    vol_mask_t = None
                    mask_frac = 0.0
            if self.load_device:
                vol_1_set.append(vol_1_t.to(self.device))
                vol_2_set.append(vol_2_t.to(self.device))
            else:
                vol_1_set.append(vol_1_t.cpu())
                vol_2_set.append(vol_2_t.cpu())
            if vol_mask_t is not None:
                vol_mask_set.append(vol_mask_t.cpu())

        if len(vol_mask_set) == 1:
            vol_mask_set = None

        self.vol_data = MultiVolume(volume_1_set=vol_1_set,
                                    volume_2_set=vol_2_set,
                                    wedge_set=self.wedge_input_set,
                                    wedge_eq_set=None,
                                    mask_set=vol_mask_set,
                                    mask_frac=mask_frac,
                                    crop_size=self.crop_size,
                                    use_flips=self.configs.use_flips,
                                    n_crops=self.configs.batch_size,
                                    normalize_crops=self.configs.normalize_crops,
                                    device=self.device)
        if self.n_volumes == 1:
            # Faster to have a single worker if only a single volume
            self.configs.num_workers = 0

        if self.load_device: # then fit all on GPU and use one worker and don't pin the memory
            self.vol_loader = DataLoader(self.vol_data,
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=0,
                                         pin_memory=False)
        else:
            self.vol_loader = DataLoader(self.vol_data,
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=self.configs.num_workers,
                                         pin_memory=True)

        self.k_sets = self.vol_data.k_sets

    def get_estimates(self, inp_1, inp_2):
        """
        Computes the estimates for the input crops inp_1 and inp_2.
        """
        if self.configs.use_mixed_precision:
            with self.autocast:
                est_1 = self.model(inp_1[:, None])[:, 0]
                est_2 = self.model(inp_2[:, None])[:, 0]
            est_1 = est_1.float()
            est_2 = est_2.float()
        else:
            est_1 = self.model(inp_1[:, None])[:, 0]
            est_2 = self.model(inp_2[:, None])[:, 0]

        return est_1, est_2

    def train(self, iterations=None, configs=None):
        self.model.train()
        if iterations is None:
            iterations = self.configs.iterations
        if self.configs.use_mixed_precision:
            print("Using mixed precision training")
            scaler = torch.amp.GradScaler('cuda')
            self.autocast = torch.amp.autocast('cuda')
        else:
            print("Not using mixed precision training")
            scaler = None
            self.autocast = None
        # So that the total number of iterations given to the model is the same whatever the number of volumes.
        self.current_iteration = 0
        if hasattr(self.configs, 'compile'):
            if self.configs.compile:
                print("Compiling the model")
                self.model = torch.compile(self.model, mode='max-autotune', fullgraph=True)

        disable_bar = not sys.stderr.isatty()  # keep logs clean on non-TTY (e.g., SLURM)
        pbar = tqdm(total=iterations, desc="Training", dynamic_ncols=True, disable=disable_bar)
        ema = None
        alpha = 0.1  # EMA smoothing for display

        print("####################")
        print("  Started training the model.")
        print("####################")
        # Actual training loop
        for iteration in range(iterations):
            loss_val_set = []
            for data in self.vol_loader:
                inp_1 = data['input_1'][0].to(self.device)
                inp_2 = data['input_2'][0].to(self.device)
                idx = data['idx'][0].item()

                self.optimizer.zero_grad()
                if self.configs.use_inp_wedge:
                    wedge = data['wedge'][0].to(self.device)
                    inp_1 = get_measurement(inp_1, wedge)
                    inp_2 = get_measurement(inp_2, wedge)

                loss = self.compute_loss(inp_1, inp_2, idx)
                if self.configs.use_mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                loss_val_set.append(float(loss.detach().item()))

            loss_val = np.mean(loss_val_set)
            ema = loss_val if ema is None else (alpha * loss_val + (1 - alpha) * ema)
            pbar.set_postfix(iteration=iteration + 1, ema_loss=f"{ema:.4f}", loss=f"{loss_val:.4f}")
            pbar.update(1)
            if iteration > 0 and iteration % self.configs.compute_avg_loss_n_iterations == 0:
                self.compute_average_loss()

                iter_ = np.arange(0, iteration * self.n_volumes+1, self.configs.compute_avg_loss_n_iterations)[1:]
                iter_ = iter_[:len(self.loss_avg_set)]
                plt.semilogy(iter_, np.array(self.loss_avg_set), label='Average loss')
                plt.savefig(os.path.join(self.save_path, 'losse_avg.png'), dpi=300, bbox_inches='tight')
                plt.close()
                plt.semilogy(iter_, np.array(self.equi_loss_avg_set), label='Equivariant loss')
                plt.savefig(os.path.join(self.save_path, 'losse_equi.png'), dpi=300, bbox_inches='tight')
                plt.close()
                plt.semilogy(iter_, np.array(self.obs_loss_avg_set), label='Data-fidelity loss')
                plt.savefig(os.path.join(self.save_path, 'losse_obs.png'), dpi=300, bbox_inches='tight')
                plt.close()

                filename = os.path.join(self.save_path, 'losses.csv')
                with open(filename, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Iterations", "Avg loss", "Equivariant loss", "Data-fidelity loss"])  # header
                    for row in zip(iter_, self.loss_avg_set, self.equi_loss_avg_set, self.obs_loss_avg_set):
                        writer.writerow(row)

            if iteration > 0 and iteration % self.configs.save_n_iterations == 0:
                self.save_model(iteration)

            if iteration > 0 and self.configs.save_tomo_n_iterations > 0 and iteration % self.configs.save_tomo_n_iterations == 0:
                print("Predict tomograms with current model.")
                vol_est_list = self.predict_dir(**configs.predict_params)
                for i in range(len(vol_est_list)):
                    # Save the estimated volume
                    name_1 = self.vol_paths_1[i].split('/')[-1].split('.mrc')[0]
                    name_2 = self.vol_paths_2[i].split('/')[-1].split('.mrc')[0]
                    vol_est_name = 'latest_prediction_' + name_1 +'_'+ name_2 + '.mrc'
                    vol_save_path = os.path.join(self.save_path, vol_est_name)
                    out = mrcfile.new(vol_save_path, overwrite=True)
                    out.set_data(np.moveaxis(vol_est_list[i].astype(np.float32), 2, 0))
                    out.close()
        print("####################")
        print("  Finished training the model.")
        print("####################")

    def compute_loss(self, inp_1, inp_2, idx):
        """
        Compute the loss between two inputs.
        Args:
            inp_1 (torch.Tensor): First input tensor.
            inp_2 (torch.Tensor): Second input tensor.
        Returns:
            torch.Tensor: Computed loss.
        """
        est_1, est_2 = self.get_estimates(inp_1, inp_2)
        wedge_input = self.wedge_input_set[idx].to(self.device)

        loss = fourier_loss(target=inp_1,
                            estimate=est_2,
                            wedge=wedge_input,
                            criteria=self.criteria,
                            use_fourier=self.configs.use_fourier,
                            window=self.window) + fourier_loss(target=inp_2,
                                                               estimate=est_1,
                                                               wedge=wedge_input,
                                                               criteria=self.criteria,
                                                               use_fourier=self.configs.use_fourier,
                                                               window=self.window)
        with torch.no_grad():
            self.loss_set.append(loss.item())
            diff_loss = torch.mean(torch.abs(est_1 - est_2))
            self.diff_loss_set.append(diff_loss.item())
        return loss


    def compute_average_loss(self):
        """
        Compute the average loss over the loss set.
        """
        avg_loss = np.mean(self.loss_set)
        avg_diff_loss = np.mean(self.diff_loss_set)

        avg_obs_loss = np.mean(self.obs_loss_set)
        avg_equi_loss = np.mean(self.equi_loss_set)
        #print(f"Average Loss: {avg_loss}, Average Diff Loss: {avg_diff_loss}")
        self.loss_set = []
        self.diff_loss_set = []
        self.obs_loss_set = []
        self.equi_loss_set = []
        self.loss_avg_set.append(avg_loss)
        self.diff_loss_avg_set.append(avg_diff_loss)
        self.obs_loss_avg_set.append(avg_obs_loss)
        self.equi_loss_avg_set.append(avg_equi_loss)
        print(
            f"Average Loss: {avg_loss}, Average Diff Loss: {avg_diff_loss}, Average Obs Loss: {avg_obs_loss}, Average Equi Loss: {avg_equi_loss}")

    def save_model(self, iteration=None):
        """
        Save the model state and loss information.
        Args:
            iteration (int): Current iteration number.
        """
        if iteration is None:
            iteration = self.configs.iterations
        # make sure the save path exists
        model_save_path = os.path.join(self.save_path, 'model')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        # save the model state
        model_path = os.path.join(model_save_path, f'model_iteration_{iteration}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_set': self.loss_set,
            'diff_loss_set': self.diff_loss_set,
            'loss_avg_set': self.loss_avg_set,
            'diff_loss_avg_set': self.diff_loss_avg_set,
            'obs_loss_avg_set': self.obs_loss_avg_set,
            'equi_loss_avg_set': self.equi_loss_avg_set,
        }, model_path)

        # save the configs as json
        config_path = os.path.join(model_save_path, 'train_configs.json')
        with open(config_path, 'w') as f:
            json.dump(self.configs.__dict__, f, indent=4)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path, pretrained=False):
        """
        Load the model state from the given path.
        Args:
            model_path (str): Path to the model state file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
        if pretrained:
            try:
                print("Loading pretrained model")
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            except KeyError as e:
                self.model.load_state_dict(checkpoint, strict=False)
        else:
            print("Loading model from checkpoint")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss_set = checkpoint['loss_set']
            self.diff_loss_set = checkpoint['diff_loss_set']
            self.loss_avg_set = checkpoint['loss_avg_set']
            self.diff_loss_avg_set = checkpoint['diff_loss_avg_set']
            self.obs_loss_avg_set = checkpoint['obs_loss_avg_set']
            self.equi_loss_avg_set = checkpoint['equi_loss_avg_set']

        print(f"Model loaded from {model_path}")

    def predict_dir(self, stride=None,
                    crop_size=None,
                    batch_size=2,
                    pre_pad=True,
                    pre_pad_size=None,
                    avg_pool=False,
                    **kwargs):
        """
        Predict using the trained model.
        This method should be overridden by subclasses if needed.
        stride: int, optional, if None the input crop size is used as stride
        batch_size: int, optional, number of crops to process at once
        run_multi: bool, optional, to run the model multiple times on each crop
        wedge: torch.Tensor, optional, wedge to apply to the input
        pre_pad: bool, optional, whether to apply pre-padding to the input
        pre_pad_size: int, optional, size of the pre-padding to apply
        avg_pool: bool, optional, whether to apply average pooling to the input, to somewhat mimic the rotational effects presetn in training
        """
        self.model.eval()
        if stride is None:
            stride = self.configs.crop_size
        if pre_pad_size is None:
            pre_pad_size = self.configs.crop_size // 4
        if crop_size is None:
            crop_size = self.configs.crop_size
        else:
            self.window = self.initialize_window(crop_size)
            pre_pad_size = crop_size // 4
        if hasattr(self.configs, 'window_type'):
            self.window_type = self.configs.window_type
        else:
            self.window_type = None

        vol_est_list = []
        for i in range(len(self.vol_data.volume_1_set)):
            wedge_used = self.wedge_input_set[i]
            vol_est_1, _ = inference(model=self.model,
                                     vol_input=self.vol_data.volume_1_set[i],
                                     size=crop_size,
                                     stride=stride,
                                     batch_size=batch_size,
                                     window=self.window,
                                     wedge=wedge_used,
                                     pre_pad=pre_pad,
                                     pre_pad_size=pre_pad_size,
                                     device=self.device,
                                     upsampled_=self.configs.upsample_volume,
                                     avg_pool=avg_pool)

            vol_est_2, _ = inference(model=self.model,
                                     vol_input=self.vol_data.volume_2_set[i],
                                     size=crop_size,
                                     stride=stride,
                                     batch_size=batch_size,
                                     window=self.window,
                                     wedge=wedge_used,
                                     pre_pad=pre_pad,
                                     pre_pad_size=pre_pad_size,
                                     device=self.device,
                                     upsampled_=self.configs.upsample_volume,
                                     avg_pool=avg_pool)
            vol_est = (vol_est_1 + vol_est_2) / 2
            vol_est_list.append(vol_est)
        return vol_est_list
