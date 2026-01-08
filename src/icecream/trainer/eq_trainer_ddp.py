"""
Class to train the model using the standard equivariant loss function with  with loss on rotated crops.
"""
import os
import numpy as np
import mrcfile
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from .eq_trainer import EquivariantTrainer
from icecream.utils.utils import batch_rot_4vol, batch_rot_wedge_full_4vols,crop_vol,fourier_loss, fourier_loss_batch,get_measurement,get_measurement_multi_wedge
from torch.utils.data import DataLoader
from icecream.dataset.multi_volumes import MultiVolume
from tqdm import tqdm


class EquivariantTrainerDDP(EquivariantTrainer):
    """
    Trainer for equivariant models using the standard equivariant loss function with loss on rotated crops.
    """
    def __init__(self,
                 configs,
                 model,
                 world_size,
                 rank,
                 angle_max_set=[60],
                 angle_min_set=[-60],
                 angles_set = None,
                 save_path='./',
                 ):

        self.world_size = world_size
        self.rank = rank
        raw_device = configs.device[self.rank]
        if isinstance(raw_device, int):
            device = "cpu" if raw_device == -1 else f"cuda:{raw_device}"
        else:
            device = str(raw_device)
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.model= torch.nn.parallel.DistributedDataParallel(self.model, 
                                                              device_ids=[self.device])
        self.save_path = save_path

        print(f"Trainer initialized on rank {self.rank} with device {self.device}.")

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

    def load_data(self, vol_paths_1,
                   vol_paths_2,
                     vol_mask_path=None, 
                     vol_1_set=[],
                     vol_2_set=[],
                     vol_mask_set=[],
                     mask_frac=0.3,):
        """
        Load data from the given volume paths.
        Args:
            vol_paths_1 (list): List of paths to the first set of volumes.
            vol_paths_2 (list): List of paths to the second set of volumes.
            vol_mask_path (str, optional): Path to the volume mask. Defaults to None.
        """

        self.vol_paths_1 = vol_paths_1
        self.vol_paths_2 = vol_paths_2
        self.vol_mask_path = vol_mask_path

        self.n_volumes = len(self.vol_paths_1)

       
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

        # Hardcoded for now as we observe massive slowdown with other values
        self.configs.num_workers = 0

        sampler = torch.utils.data.DistributedSampler(
                self.vol_data,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )


        if self.load_device: # then fit all on GPU and use one worker and don't pin the memory
            self.vol_loader = DataLoader(self.vol_data,
                                         batch_size=1,
                                         sampler=sampler,
                                         num_workers=0,
                                         pin_memory=False)
        else:
            self.vol_loader = DataLoader(self.vol_data,
                                         batch_size=1,
                                         sampler=sampler,
                                         num_workers=self.configs.num_workers,
                                         pin_memory=True)

        self.k_sets = self.vol_data.k_sets


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

        # disable_bar = not sys.stderr.isatty()  # keep logs clean on non-TTY (e.g., SLURM)
        if self.rank == 0:
            pbar = tqdm(total=iterations, desc="Training", dynamic_ncols=True, disable=False)
        ema = None
        alpha = 0.1  # EMA smoothing for display

        if self.rank == 0:
            print("####################")
            print("  Started training the model.")
            print("####################")
        # Actual training loop
        iteration = -1
        loss_val = np.nan
        ema = np.nan
        print(f"Total iterations: {self.configs.iterations}")
        while iteration < self.configs.iterations:
        # for iteration in range(iterations):
            loss_val_set = []
            for data in self.vol_loader:
                iteration += 1
                inp_1 = data['input_1'][0].to(self.device)
                inp_2 = data['input_2'][0].to(self.device)
                #print(torch.linalg.norm(inp_1))
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

                #print(f"Rank {self.rank}, Iteration {iteration+1}, Loss: {loss_val_set[-1]:.4f}")
                if self.rank == 0:
                    if iteration > len(self.vol_loader) and iteration % self.configs.compute_avg_loss_n_iterations == 0:
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

                    if iteration > len(self.vol_loader) and iteration % self.configs.save_n_iterations == 0:
                        self.save_model(iteration)

                    if iteration > len(self.vol_loader) and self.configs.save_tomo_n_iterations > 0 and iteration % self.configs.save_tomo_n_iterations == 0:
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
                if self.rank == 0:
                    pbar.set_postfix(iteration=iteration + 1, ema_loss=f"{ema:.4f}", loss=f"{loss_val:.4f}")
                    pbar.update(1)

            loss_val = np.mean(loss_val_set)
            ema = loss_val if np.isnan(ema) else (alpha * loss_val + (1 - alpha) * ema)
        if self.rank == 0:
            print("####################")
            print("  Finished training the model.")
            print("####################")

