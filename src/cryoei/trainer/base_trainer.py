"""
Base trainer class for training models using nois2noise type loss
"""

import os
import torch
import numpy as np
import mrcfile
from torch.utils.data import DataLoader
from utils.utils import get_wedge_3d_new,symmetrize_3D,get_measurement,fourier_loss
from utils.mask_util import make_mask

from dataset.volumes import singleVolume
from utils.inference_util import inference,inference_2
from tqdm import tqdm
import json

class BaseTrainer:
    def __init__(self,
                 configs,
                 model,
                 angle_max= 60,
                 angle_min = -60,
                 angles = None,
                 save_path = './',
                 ):
        
        self.device = configs.device
        self.model = model.to(self.device)
        self.save_path = save_path
        
        self.loss_set = []
        self.diff_loss_set = []
        self.loss_avg_set = []
        self.diff_loss_avg_set = []
        self.configs = configs

        self.criteria = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(configs.learning_rate))


        self.obs_loss_set = []
        self.obs_loss_avg_set = []
        self.equi_loss_set = []
        self.equi_loss_avg_set = []


        if angles is not None:
            self.angle_max = np.max(angles)
            self.angle_min = np.min(angles)
        else:
            self.angle_max = angle_max
            self.angle_min = angle_min
        self.setup()
        


    def setup(self):
        self.crop_size = self.configs.input_crop_size

        self.wedge_input = self.initialize_wedge(self.crop_size)
        self.window = self.initialize_window(self.crop_size)


    def get_real_binary_filter(self, binary_filter):

        binary_filter_sym = symmetrize_3D(binary_filter)
        binary_filter_t = (binary_filter + binary_filter_sym)/2
        binary_filter_t[binary_filter_t>0.1]  = 1

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
        return (vol - vol.mean()) / vol.std()

    def initialize_wedge(self,crop_size, wedge_support = None):
        """
        Initialize the wedge for the model.
        This method should be overridden by subclasses if needed.
        """

        if wedge_support is None:
            wedge_support = self.configs.wedge_low_support
        wedge,ball = get_wedge_3d_new(crop_size,
                                      max_angle = self.angle_max,
                                      min_angle = self.angle_min,
                                      rotation = 0,
                                      low_support=wedge_support,
                                      use_spherical_support= self.configs.use_spherical_support)
        wedge_t = torch.tensor(wedge, dtype=torch.float32, device=self.device)
        wedge_t_sym = symmetrize_3D(wedge_t)
        wedge_t = (wedge_t_sym + wedge_t)/2
        wedge_t[wedge_t>0.1]  = 1

        return  wedge_t

    def initialize_window(self,crop_size):
        w = np.zeros((crop_size,crop_size,crop_size))
        w[crop_size//4:-crop_size//4,crop_size//4:-crop_size//4,crop_size//4:-crop_size//4] = 1
        w_t = torch.tensor(w, dtype=torch.float32, device=self.device)     
        return w_t   


    def load_data(self,vol_paths_1, vol_paths_2, vol_mask_path = None, use_mask = False, mask_frac = 0.3):

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

        if len(vol_paths_1) != len(vol_paths_2):
            raise ValueError("The number of volume paths for vol_paths_1 and vol_paths_2 must be the same.")
        if len(vol_paths_1) >1:
            # raise not implemented error
            raise NotImplementedError("Loading multiple volumes is not implemented yet. Please provide a single volume path for each set.")
        
        if len(vol_paths_1) == 1:
            vol_1 = mrcfile.open(vol_paths_1[0] ).data
            vol_1 = np.moveaxis(vol_1,0,2).astype(np.float32)
            vol_1_t =torch.tensor(vol_1, dtype=torch.float32, device='cpu')
            vol_1_t = self.normalize_volume(vol_1_t)
            

            vol_2 = mrcfile.open(vol_paths_2[0] ).data
            vol_2 = np.moveaxis(vol_2,0,2).astype(np.float32)
            vol_2_t =torch.tensor(vol_2, dtype=torch.float32, device='cpu')
            vol_2_t = self.normalize_volume(vol_2_t)

        if vol_mask_path is not None:
            vol_mask = mrcfile.open(vol_mask_path).data
            vol_mask = np.moveaxis(vol_mask,0,2).astype(np.float32)
            vol_mask_t = torch.tensor(vol_mask, dtype=torch.float32, device='cpu')
        else:
            if use_mask:
                vol_avg= ((vol_1_t + vol_2_t)/2).cpu().numpy()
                #TODO: add these parameters to the config
                vol_mask = make_mask(vol_avg,mask_boundary = None, side = 5, density_percentage=50., std_percentage=50)
                vol_mask_t =torch.tensor(vol_mask, dtype=torch.float32, device=self.device)

            else:
                vol_mask_t = None
                mask_frac = 0.0

            if hasattr(self.configs, 'window_type') is False:
                self.configs.window_type = 'boxcar'

            self.vol_data = singleVolume(volume_1 = vol_1_t,
                        volume_2=  vol_2_t, 
                        wedge = self.wedge_input, 
                        mask= vol_mask_t,
                        mask_frac= mask_frac,
                        crop_size= self.crop_size,
                        use_flips=self.configs.use_flips,
                        normalize_crops=self.configs.normalize_crops,
                        upsample_volume=self.configs.upsample_volume,
                        window_type=self.configs.window_type,
                        min_distance=self.configs.min_distance,
                        device=self.device
                        )
            
            self.k_sets = self.vol_data.k_sets
            


    def train(self,repeats =None):
        self.model.train()

        if repeats is None:
            repeats = self.configs.epochs
        # Temperary fix for the scheduler

        if hasattr(self.configs, 'use_scheduler'):
            if self.configs.use_scheduler:
                # print("Using cosine scheduler with warmup") 
                # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=repeats)
                # warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)
                print("Using multistep scheduler")
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                                    milestones=self.configs.scheduler_milestones, 
                                                                    )
                self.use_scheduler = True

            else:
                print("Scheduler not used")
                self.use_scheduler = False
        else:

            self.use_scheduler = False

        if self.configs.use_mixed_precision:
            print("Using mixed precision training")
            scaler = torch.cuda.amp.GradScaler()
            self.autocast = torch.cuda.amp.autocast
        else:
            print("Not using mixed precision training")
            scaler = None
            autocast = None            
        self.current_epoch = 0
        for epoch in tqdm(range(repeats), desc="Training"):
            self.optimizer.zero_grad()
            self.current_epoch = epoch
            data = self.vol_data.get_random_crop(self.configs.batch_size)
            inp_1= data['input_1'].to(self.device)
            inp_2 = data['input_2'].to(self.device)
            if self.configs.use_inp_wedge:
                inp_1 = get_measurement(inp_1,self.wedge_input)
                inp_2 = get_measurement(inp_2,self.wedge_input)


            loss = self.compute_loss(inp_1,inp_2)
            if self.configs.use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
    
            if self.use_scheduler:
                print("Stepping the scheduler")
                lr_scheduler.step()
                # with warmup_scheduler.dampening():
                #     lr_scheduler.step()
            tqdm.write(f"Epoch {epoch+1}/{repeats}, Loss: {loss.item():.4f}")
            if epoch>0 and epoch % self.configs.compute_avg_loss_n_epochs == 0:
                self.compute_average_loss()


            if epoch>0 and epoch % self.configs.save_n_epochs == 0:
                self.save_model(epoch)


        
    def compute_loss(self, inp_1, inp_2):
        """
        Compute the loss between two inputs.
        Args:
            inp_1 (torch.Tensor): First input tensor.
            inp_2 (torch.Tensor): Second input tensor.
        Returns:
            torch.Tensor: Computed loss.
        """
        est_1= self.model(inp_1[:,0])[:,0]
        est_2= self.model(inp_2[:,0])[:,0]


        loss = fourier_loss(target = inp_1, 
                            estimate = est_2,
                            criteria = self.criteria,
                              use_fourier = self.configs.use_fourier,
                                window = self.window) + fourier_loss(target = inp_2,
                            estimate = est_1,
                            criteria = self.criteria,
                              use_fourier = self.configs.use_fourier,
                                window = self.window)
       
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
        print(f"Average Loss: {avg_loss}, Average Diff Loss: {avg_diff_loss}, Average Obs Loss: {avg_obs_loss}, Average Equi Loss: {avg_equi_loss}")

    def save_model(self, epoch=None):
        """
        Save the model state and loss information.
        Args:
            epoch (int): Current epoch number.
        """

        if epoch is None:
            epoch = self.configs.epochs
        # make sure the save path exists
        model_save_path = os.path.join(self.save_path, 'model')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        # save the model state
        model_path = os.path.join(model_save_path, f'model_epoch_{epoch}.pt')
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
        
        checkpoint = torch.load(model_path, weights_only=False)
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



    def predict(self,stride = None,
                crop_size = None,
                batch_size =2, 
                run_multi = None, 
                wedge =None,
                update_missing_wedge = False,
                denoise_first = False,
                pre_pad = True,
                pre_pad_size = None,
                avg_pool = False):
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
            stride = self.configs.input_crop_size

        if pre_pad_size is None:
            pre_pad_size = self.configs.input_crop_size//4

        if crop_size is None:
            crop_size = self.configs.input_crop_size
        else:
            self.window = self.initialize_window(crop_size)
            pre_pad_size = crop_size//4

        if self.configs.no_window:
            self.window = None

        #vol_input = (self.vol_data.volume_1 + self.vol_data.volume_2)/2

        # vol_est, _ = inference(model = self.model, 
        #                             vol_input = vol_input,
        #                             size =self.configs.input_crop_size, 
        #                             stride =stride, 
        #                             batch_size=batch_size,
        #                             window = self.window,
        #                             run_multi=run_multi,
        #                             wedge = wedge,
        #                             pre_pad=pre_pad,
        #                             pre_pad_size = pre_pad_size,
        #                             avg_pool=avg_pool)

        # vol_input = (self.vol_data.volume_1 + self.vol_data.volume_2)/2

        # vol_est, _ = inference(model = self.model, 
        #                             vol_input = vol_input,
        #                             size =self.configs.input_crop_size, 
        #                             stride =stride, 
        #                             batch_size=batch_size,
        #                             window = self.window,
        #                             run_multi=None,
        #                             wedge = wedge,
        #                             pre_pad=pre_pad,
        #                             pre_pad_size = pre_pad_size,
        #                             avg_pool=avg_pool)
        # if run_multi is not None and run_multi > 0:
        #     for _ in range(run_multi):
        #         vol_est = torch.tensor(vol_est, dtype=torch.float32, device=self.device)
        #         vol_est, _ = inference(model = self.model, 
        #                                 vol_input = vol_est,
        #                                 size =self.configs.input_crop_size, 
        #                                 stride =stride, 
        #                                 batch_size=batch_size,
        #                                 window = self.window,
        #                                 run_multi=None,
        #                                 wedge = wedge,
        #                                 pre_pad=pre_pad,
        #                                 pre_pad_size = pre_pad_size,
        #                                 avg_pool=avg_pool)
        wedge_used = wedge
        update_missing_wedge_flag = False
        if update_missing_wedge:
            update_missing_wedge_flag = True
            if wedge is None:
                # if the self.wedge_eq is present use it
                if hasattr(self, 'wedge_eq'):
                    wedge_used = self.wedge_eq
                else:
                    wedge_used = self.wedge_input
                

        if denoise_first:
            update_missing_wedge_flag = False
            wedge_used = None

        if hasattr(self.configs, 'window_type'):
            self.window_type = self.configs.window_type
        else:
            self.window_type =  None

        vol_est_1, _ = inference(model = self.model, 
                                    vol_input = self.vol_data.volume_1,
                                    size =crop_size, 
                                    stride =stride, 
                                    batch_size=batch_size,
                                    window = self.window,
                                    window_type = self.window_type,
                                    run_multi=None,
                                    wedge = wedge_used,
                                    update_missing_wedge=update_missing_wedge_flag,
                                    pre_pad=pre_pad,
                                    pre_pad_size = pre_pad_size,
                                    device=self.device,
                                    upsampled_=self.configs.upsample_volume,
                                    avg_pool=avg_pool)

        vol_est_2, _ = inference(model = self.model, 
                                    vol_input = self.vol_data.volume_2,
                                    size =crop_size, 
                                    stride =stride, 
                                    batch_size=batch_size,
                                    window = self.window,
                                    window_type = self.window_type,
                                    run_multi=None,
                                    wedge = wedge_used,
                                    update_missing_wedge=update_missing_wedge_flag,
                                    pre_pad=pre_pad,
                                    device=self.device,
                                    pre_pad_size = pre_pad_size,
                                    upsampled_=self.configs.upsample_volume,
                                    avg_pool=avg_pool)   


        if denoise_first:
            update_missing_wedge_flag = True
            if wedge is None:
                # if the self.wedge_eq is present use it
                if hasattr(self, 'wedge_eq'):
                    wedge_used = self.wedge_eq
                else:
                    wedge_used = self.wedge_input    
        
        if run_multi is not None and run_multi > 0:
            for i in range(run_multi):

                # Dont update missing wedge for the last run
                if denoise_first is False:
                    if i == (run_multi - 1) and update_missing_wedge:
                        wedge_used = None 
                        update_missing_wedge_flag = False

                      

                vol_est_1 = torch.tensor(vol_est_1, dtype=torch.float32, device=self.device)
                vol_est_2 = torch.tensor(vol_est_2, dtype=torch.float32, device=self.device)

                vol_est_1, _ = inference(model = self.model, 
                                            vol_input = vol_est_1,
                                            size =crop_size, 
                                            stride =stride, 
                                            batch_size=batch_size,
                                            window = self.window,
                                            run_multi=None,
                                            wedge = wedge_used,
                                            update_missing_wedge=update_missing_wedge_flag,
                                            pre_pad=pre_pad,
                                            pre_pad_size = pre_pad_size,
                                            avg_pool=avg_pool)

                vol_est_2, _ = inference(model = self.model, 
                                            vol_input = vol_est_2,
                                            size =crop_size, 
                                            stride =stride, 
                                            batch_size=batch_size,
                                            window = self.window,
                                            run_multi=None,
                                            wedge = wedge_used,
                                            update_missing_wedge=update_missing_wedge_flag,
                                            pre_pad=pre_pad,
                                            pre_pad_size = pre_pad_size,
                                            avg_pool=avg_pool)  

        vol_est = (vol_est_1 + vol_est_2)/2
        
        return vol_est


    def predict_dir(self,stride = None,
                crop_size = None,
                batch_size =2, 
                run_multi = None, 
                wedge =None,
                update_missing_wedge = False,
                denoise_first = False,
                pre_pad = True,
                pre_pad_size = None,
                avg_pool = False):
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
            stride = self.configs.input_crop_size

        if pre_pad_size is None:
            pre_pad_size = self.configs.input_crop_size//4

        if crop_size is None:
            crop_size = self.configs.input_crop_size
        else:
            self.window = self.initialize_window(crop_size)
            pre_pad_size = crop_size//4


        # if self.configs.no_window:
        #     self.window = None

        wedge_used = self.wedge_input

        if hasattr(self.configs, 'window_type'):
            self.window_type = self.configs.window_type
        else:
            self.window_type =  None

        vol_est_1, _ = inference_2(model = self.model, 
                                    vol_input = self.vol_data.volume_1,
                                    size =crop_size, 
                                    stride =stride, 
                                    batch_size=batch_size,
                                    window = self.window,
                                    window_type = self.window_type,
                                    run_multi=None,
                                    wedge = wedge_used,
                                    update_missing_wedge=False,
                                    pre_pad=pre_pad,
                                    pre_pad_size = pre_pad_size,
                                    device=self.device,
                                    upsampled_=self.configs.upsample_volume,
                                    avg_pool=avg_pool)

        vol_est_2, _ = inference_2(model = self.model, 
                                    vol_input = self.vol_data.volume_2,
                                    size =crop_size, 
                                    stride =stride, 
                                    batch_size=batch_size,
                                    window = self.window,
                                    window_type = self.window_type,
                                    run_multi=None,
                                    wedge = wedge_used,
                                    update_missing_wedge=False,
                                    pre_pad=pre_pad,
                                    device=self.device,
                                    pre_pad_size = pre_pad_size,
                                    upsampled_=self.configs.upsample_volume,
                                    avg_pool=avg_pool)   


 
        vol_est = (vol_est_1 + vol_est_2)/2
        
        return vol_est
    

