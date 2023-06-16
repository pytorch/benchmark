# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from ...util.model import BenchmarkModel
from .build_sam import sam_model_registry
from .predictor import SamPredictor
from PIL import Image
import numpy as np
import cv2
from torchbenchmark.tasks import COMPUTER_VISION
from .transforms import ResizeLongestSide

import torch
import os
import monai
import tqdm
from torch.utils.data import Dataset, DataLoader


class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(os.path.join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float()


    
class Model(BenchmarkModel):
    task = COMPUTER_VISION.SEGMENTATION
    DEFAULT_EVAL_BSIZE = 32
    
    def __init__(self, test, device, jit=False, batch_size=1, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        
        # Checkpoint options are here https://github.com/facebookresearch/segment-anything#model-checkpoints
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')
        sam_checkpoint = os.path.join(data_folder, 'sam_vit_h_4b8939.pth')
        model_type = "vit_h"

        self.model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.model.to(device=device)   
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')

        image_path = os.path.join(data_folder, 'truck.jpg')
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)     
        self.sample_image  = torch.randn((3, 256, 256)).to(device)

   
    def get_module(self):
        example_input = [
            {
                'image': self.sample_image,
                'original_size': (256, 256),
            }
        ]

        multimask_output = False

        return self.model, (example_input, multimask_output)
            
    def train(self):
        # Code copied from https://github.com/bowang-lab/MedSAM/blob/main/finetune_and_inference_tutorial.py
        
        
        npz_tr_path = 'data/Npz_files/CT_Abd-Gallbladder/train'
        work_dir = './work_dir'
        task_name = 'CT_Abd-Gallbladder'
        # prepare SAM model
        model_type = 'vit_b'
        checkpoint = 'work_dir/SAM/sam_vit_b_01ec64.pth'
        device = 'cuda:0'
        model_save_path = os.path.join(work_dir, task_name)
        os.makedirs(model_save_path, exist_ok=True)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        num_epochs = 100
        losses = []
        best_loss = 1e10

        train_dataset = NpzDataset(npz_tr_path)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        for epoch in range(num_epochs):
            epoch_loss = 0
            # train
            for step, (image_embedding, gt2D, boxes) in enumerate(tqdm(train_dataloader)):
                # do not compute gradients for image encoder and prompt encoder
                with torch.no_grad():
                    # convert box to 1024x1024 grid
                    box_np = boxes.numpy()
                    sam_trans = ResizeLongestSide(self.model.image_encoder.img_size)
                    box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
                    box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                    if len(box_torch.shape) == 2:
                        box_torch = box_torch[:, None, :] # (B, 1, 4)
                    # get prompt embeddings 
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None,
                    )
                # predicted masks
                mask_predictions, _ = self.model.mask_decoder(
                    image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
                    image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                    multimask_output=False,
                )

                loss = seg_loss(mask_predictions, gt2D.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= step
            losses.append(epoch_loss)
            print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
            # save the latest model checkpoint
            torch.save(self.model.state_dict(), os.path.join(model_save_path, 'sam_model_latest.pth'))
            # save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), os.path.join(model_save_path, 'sam_model_best.pth'))


    def eval(self):
        predictor = SamPredictor(self.model)

        predictor.set_image(self.image)

        input_point = np.array([[500, 375]])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True)
        return (masks,)