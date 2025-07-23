# Custom Dataset Class

import os, h5py
import random

import cv2
import math

import torch
import torch.nn as nn
import torchvision.transforms as tf

from torch.utils.data import Dataset
from PIL import Image
import json

from torchvision.transforms import v2
from torchvision import tv_tensors

import pandas as pd

# from functools import lru_cache

## Frame dataset
class CustomDataset(Dataset):
    def __init__(self, dir_path, transform=None, time_window=3, buffer_num_sample=8, line_width=20, b_thres=False, det_num_classes=1, use_h5=True, add_t0=False, add_t0_per_frame=0):
        """
        Args:
            root (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            add_t0_per_frame (for video training only)
        """
        if line_width == 20 and b_thres:
            self.mask_name_suffix = "_bt_lw_20.png"
        elif line_width == 20:
            self.mask_name_suffix = "_lw_20.png"
        else:
            self.mask_name_suffix = ".png"

        self.dir_path = dir_path
        self.time_window = time_window
        self.buffer_num_sample = buffer_num_sample
        self.buffer_length = time_window + buffer_num_sample - 1
        self.dilate = 1
        self.file_names = sorted(os.listdir(self.dir_path))
        self.image_names = [f for f in self.file_names if f[0] == "a" and f.endswith(".jpg")]  # ["a0001.jpg", "a0002.jpg", ...]
        self.mask_names = ["m" + f[1:-4] + self.mask_name_suffix for f in self.image_names]  # ["m0001_lw_20.png", "m0002_lw_20.png", ...]
        self.json_names = [f.replace(".jpg", ".json") for f in self.image_names]  # ["a0001.json", "a0002.json", ...]
        self.image_names_origin = [f for f in self.file_names if f[0] == "a" and f.endswith(".jpg")]
        self.mask_names_origin = ["m" + f[1:-4] + self.mask_name_suffix for f in self.image_names]
        self.json_names_origin = [f.replace(".jpg", ".json") for f in self.image_names]
        self.set_origin_size()

        # -------------------------------------------------------------------------------
        # h5
        # -------------------------------------------------------------------------------        
        self.use_h5 = use_h5 #False #
        if use_h5:
            video_folder_name = os.path.basename(dir_path)
            grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))
            h5_dir = os.path.join(grandparent_dir, "h5files", f"{video_folder_name}_img_mask512.h5")
            
            if os.path.exists(h5_dir):
                f = h5py.File(h5_dir, 'r')
                self.image_stack = torch.from_numpy(f['images'][:])
                self.mask_stack = torch.from_numpy(f['masks'][:])
                self.image_stack = self.image_stack.float().div(255)  ## tf.ToTensor() includes normalize by div 255
                self.mask_stack = self.mask_stack.float().div(255)
                f.close()
                self.use_h5 = True
                print(f" use {h5_dir[40:]}", end='')
            else:
                self.use_h5 = False


        # creat T consecutive image & mask names list
        if time_window == 1:  ## duplicate
            self.consec_images_names = [
                [self.image_names[i]] * 3 for i in range(0, len(self.image_names) - self.time_window + 1)
            ]  # [["a0001.jpg", "a0001.jpg", "a0001.jpg"], ["a0002.jpg", "a0002.jpg", "a0002.jpg"], ...]
            self.consec_masks_names = [
                [self.mask_names[i]] * 3 for i in range(0, len(self.mask_names) - self.time_window + 1)
            ]  # [["m0001_lw_20.png", "m0001_lw_20.png", "m0001_lw_20.png"], ["m0002_lw_20.png", "m0002_lw_20.png", "m0002_lw_20.png"], ...]
            self.consec_json_names = [[self.json_names[i]] * 3 for i in range(0, len(self.json_names) - self.time_window + 1)]
        # -------------------------------------------------------------------------------
        # buffer for speed up
        # -------------------------------------------------------------------------------
        else:
            ## add black mask to every start of sequence
            if add_t0_per_frame > 0:
                self.image_names, self.mask_names, self.json_names = [], [], []
                for idx in range(0,len(self.image_names_origin), add_t0_per_frame):
                    self.image_names += ["", ""]
                    self.mask_names += ["", ""]
                    self.json_names += ["", ""]
                    seq_end_idx = min(len(self.image_names_origin), idx+add_t0_per_frame) ## avoid overflow
                    self.image_names += self.image_names_origin[idx: seq_end_idx]
                    self.mask_names += self.mask_names_origin[idx: seq_end_idx]
                    self.json_names += self.json_names_origin[idx: seq_end_idx]
            ## add black mask to the start
            elif add_t0:
                self.image_names, self.mask_names, self.json_names = [], [], []
                self.image_names = ["", ""]+self.image_names_origin
                self.mask_names = ["", ""]+self.mask_names_origin
                self.json_names = ["", ""]+self.json_names_origin
            
            if add_t0_per_frame > 0:
                total_num_buffer = math.ceil((len(self.image_names_origin)) / self.buffer_num_sample)
                self.consec_images_names = [
                    self.image_names[i * (self.buffer_num_sample+time_window-1) : i * (self.buffer_num_sample+time_window-1) + self.buffer_length] for i in range(0, total_num_buffer)
                ]  # [["a0001.jpg", "a0002.jpg", "a0003.jpg", ... ], ["a0003.jpg", "a0004.jpg", "a0005.jpg", ... ], ...]
                self.consec_masks_names = [
                    self.mask_names[i * (self.buffer_num_sample+time_window-1) : i * (self.buffer_num_sample+time_window-1) + self.buffer_length] for i in range(0, total_num_buffer)
                ]  # [["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png", ... ], ["m0003_lw_20.png", "m0004_lw_20.png", "m0005_lw_20.png", ... ], ...]
                self.consec_json_names = [
                self.json_names[i * (self.buffer_num_sample+time_window-1) : i * (self.buffer_num_sample+time_window-1) + self.buffer_length] for i in range(0, total_num_buffer)
            ]  # [["a0001.json", "a0002.json", "a0003.json", ... ], ["a0003.json", "a0004.json", "a0005.json", ... ], ...]
                
            else:
                total_num_buffer = 1 + math.ceil((len(self.image_names) - self.buffer_length) / self.buffer_num_sample)
                self.consec_images_names = [
                    self.image_names[i * self.buffer_num_sample : i * self.buffer_num_sample + self.buffer_length] for i in range(0, total_num_buffer)
                ]  # [["a0001.jpg", "a0002.jpg", "a0003.jpg", ... ], ["a0003.jpg", "a0004.jpg", "a0005.jpg", ... ], ...]
                self.consec_masks_names = [
                    self.mask_names[i * self.buffer_num_sample : i * self.buffer_num_sample + self.buffer_length] for i in range(0, total_num_buffer)
                ]  # [["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png", ... ], ["m0003_lw_20.png", "m0004_lw_20.png", "m0005_lw_20.png", ... ], ...]
                self.consec_json_names = [
                self.json_names[i * self.buffer_num_sample : i * self.buffer_num_sample + self.buffer_length] for i in range(0, total_num_buffer)
            ]  # [["a0001.json", "a0002.json", "a0003.json", ... ], ["a0003.json", "a0004.json", "a0005.json", ... ], ...]

            # drop the last buffer if the length is less than buffer_length (therefore the buffer_num_sample should not be too large to avoid significant dataset drop)
            if len(self.consec_images_names) > 0 and len(self.consec_images_names[-1]) < self.buffer_length:
                self.consec_images_names = self.consec_images_names[:-1]
                self.consec_masks_names = self.consec_masks_names[:-1]
                self.consec_json_names = self.consec_json_names[:-1]
        # -------------------------------------------------------------------------------
        # original version
        # -------------------------------------------------------------------------------
        # else:
        #     self.consec_images_names = [
        #         self.image_names[i : i + self.time_window] for i in range(0, len(self.image_names) - self.time_window + 1)
        #     ]  # [["a0001.jpg", "a0002.jpg", "a0003.jpg"], ["a0002.jpg", "a0003.jpg", "a0004.jpg"], ...]
        #     self.consec_masks_names = [
        #         self.mask_names[i : i + self.time_window] for i in range(0, len(self.mask_names) - self.time_window + 1)
        #     ]  # [["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png"], ["m0002_lw_20.png", "m0003_lw_20.png", "m0004_lw_20.png"], ...]
        #     self.consec_json_names = [self.json_names[i : i + self.time_window] for i in range(0, len(self.json_names) - self.time_window + 1)]
        # -------------------------------------------------------------------------------

        self.transform = transform
        self.trans_totensor = tf.Compose([tf.ToTensor()])
        self.det_num_classes = det_num_classes
        assert self.dilate >= 1

    def __len__(self):
        return len(self.consec_images_names)

    def set_origin_size(self):
        if len(self.image_names) >0 and os.path.exists(os.path.join(self.dir_path, self.image_names[0])):
            img = Image.open(os.path.join(self.dir_path, self.image_names[0])).convert("L")
            self.origin_img_size = img.size[-1]
            img.close()

    def endpoints_to_labels(self, consec_endpoints):
        consec_labels = []
        for t in range(consec_endpoints.shape[0]):
            if consec_endpoints[t, :].sum() == 0:
                consec_labels.append(torch.as_tensor(-1, dtype=torch.float32))  ## cls_id = -1: no needle
            else:
                if self.det_num_classes == 1:  ## 1 class for with needle or not
                    consec_labels.append(torch.as_tensor(0, dtype=torch.float32))  ## cls_id = 0: with needle
                else:  ## 2 classes for needles with different directions
                    if torch.sign(consec_endpoints[t][0] - consec_endpoints[t][2]) == torch.sign(consec_endpoints[t][1] - consec_endpoints[t][3]):
                        consec_labels.append(torch.as_tensor(0, dtype=torch.float32))  ## cls_id = 0: left-top to right-bottom needle
                    else:
                        consec_labels.append(torch.as_tensor(1, dtype=torch.float32))  ## cls_id = 1: right-top to left-bottom needle
        consec_labels = torch.stack(consec_labels, dim=0).long()  ## [T,]
        return consec_labels

    def get_cals_endpoints(self, json_dir, consec_json_name):
        """return consec_cals, consec_endpoints
        """
        ## Center, Angle, Length (cal) 
        consec_cals = []
        consec_endpoints = []
        for f_name in consec_json_name:
            cal = [0, 0, 0, 0]
            endpoint = [0, 0, 0, 0]
            if f_name != "":
                with open(os.path.join(json_dir, f_name), "r") as f:
                    js = json.load(f)
                    # print(js)
                # label = -1
                if len(js["shapes"]) != 0:  ## with needle
                    upper_needle_id, needle_id = -1, -1
                    for label_id in range(len(js["shapes"])):
                        if js["shapes"][label_id]["label"] == "needle" and js["shapes"][label_id].get("center"):
                            needle_id = label_id
                        elif js["shapes"][label_id]["label"] == "upper_needle" and js["shapes"][label_id].get("center"):
                            upper_needle_id = label_id
                    use_needle_label_id = 0
                    if upper_needle_id != -1:
                        use_needle_label_id = upper_needle_id
                    elif needle_id != -1:
                        use_needle_label_id = needle_id
                    if js["shapes"][use_needle_label_id].get("center"):
                        cal = [js["shapes"][use_needle_label_id]["center"][0], js["shapes"][use_needle_label_id]["center"][1], 
                            js["shapes"][use_needle_label_id]["theta"], js["shapes"][use_needle_label_id]["length"]]
                    endpoint = [
                        js["shapes"][use_needle_label_id]["points"][0][0],
                        js["shapes"][use_needle_label_id]["points"][0][1],
                        js["shapes"][use_needle_label_id]["points"][1][0],
                        js["shapes"][use_needle_label_id]["points"][1][1],
                    ]
                    # print('bbox', bbox, 'end', endpoint)
                f.close()
            consec_cals.append(torch.as_tensor(cal, dtype=torch.float32))
            # consec_endpoints.append(tv_tensors.BoundingBoxes(endpoint, format="XYXY", canvas_size=[1758,1758]) )
            consec_endpoints.append(torch.as_tensor(endpoint, dtype=torch.float32))
            # consec_labels.append(torch.as_tensor(label, dtype=torch.float32))
        consec_cals = torch.stack(consec_cals, dim=0)  ## [T, 4]
        consec_endpoints = torch.stack(consec_endpoints, dim=0)  ## [T, 4]
        return consec_cals, consec_endpoints
    
    # @lru_cache(256)
    def __getitem__(self, idx):
        consec_images_name = self.consec_images_names[idx]  # ["a0001.jpg", "a0002.jpg", "a0003.jpg", ...]
        if self.use_h5:
            if self.dilate == 1:
                if consec_images_name[1] == "": ## add empty img & mask
                    start_name_idx = self.image_names_origin.index(consec_images_name[2])
                elif consec_images_name[0] == "": ## add empty img & mask
                    start_name_idx = self.image_names_origin.index(consec_images_name[1])
                else:
                    start_name_idx = self.image_names_origin.index(consec_images_name[0])
                end_name_idx = self.image_names_origin.index(consec_images_name[-1])
                
                consec_images = self.image_stack[start_name_idx:end_name_idx+1]
                consec_masks = self.mask_stack[start_name_idx:end_name_idx+1]
            else: ## find each index since they are not sequential actually
                consec_images, consec_masks = [],[]
                for name in consec_images_name:
                    if name != "":
                        name_idx = self.image_names_origin.index(name)
                        consec_images.append(self.image_stack[name_idx])
                        consec_masks.append(self.mask_stack[name_idx])
                consec_images = torch.stack(consec_images)
                consec_masks = torch.stack(consec_masks)
            fname_list = [os.path.join(self.dir_path, f_name) if f_name != "" else "" for f_name in consec_images_name]
            curr_T = consec_images.shape[0]
            need_T = len(consec_images_name)
            if curr_T != need_T:  ## add empty img & mask
                _,h,w = consec_images.shape
                # consec_images = torch.cat([torch.zeros(need_T-curr_T,h,w), consec_images],dim=0)  ## comment this if not add 0
                consec_masks =  torch.cat([torch.zeros(need_T-curr_T,h,w), consec_masks],dim=0)
                first_image = consec_images[0].unsqueeze(0)
                if need_T-curr_T == 2:
                    consec_images = torch.cat([first_image, first_image, consec_images], dim=0)
                elif need_T-curr_T == 1:
                    consec_images = torch.cat([first_image, consec_images], dim=0)
        else:
            consec_images = []
            fname_list = []
            add_empty_mask = 0
            for f_name in consec_images_name:
                if f_name == "":
                    fname_list.append("")
                    add_empty_mask += 1
                    continue
                img = Image.open(os.path.join(self.dir_path, f_name)).convert("L")
                img_tensor = self.trans_totensor(img)
                consec_images.append(img_tensor)
                fname_list.append(os.path.join(self.dir_path, f_name))
                img.close()
            consec_images = torch.cat(consec_images, dim=0)  ## [T, H, W] float32
            if add_empty_mask>0:
                # _,h,w = img_tensor.shape
                # consec_images = torch.cat([torch.zeros(add_empty_mask,h,w),consec_images], dim=0)
                
                first_image = consec_images[0].unsqueeze(0)
                if add_empty_mask == 2:
                    consec_images = torch.cat([first_image, first_image, consec_images], dim=0)
                elif add_empty_mask == 1:
                    consec_images = torch.cat([first_image, consec_images], dim=0)

            consec_masks_name = self.consec_masks_names[idx]  # ["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png"]
            consec_masks = []
            for f_name in consec_masks_name:
                if f_name == "":
                    continue
                img = Image.open(os.path.join(self.dir_path, f_name)).convert("L")
                img_tensor = self.trans_totensor(img)
                # img_tensor.unsqueeze_(0)
                consec_masks.append(img_tensor)
                img.close()
            consec_masks = torch.cat(consec_masks, dim=0)  ## [T, H, W]
            if add_empty_mask>0:
                _,h,w = img_tensor.shape
                consec_masks = torch.cat([torch.zeros(add_empty_mask,h,w),consec_masks], dim=0)

        ## Center, Angle, Length (cal) and Cls
        consec_json_name = self.consec_json_names[idx]  # ["a0001.json", "a0002.json", "a0003.json"]
        consec_cals, consec_endpoints = self.get_cals_endpoints(self.dir_path, consec_json_name)

        # Unsqueeze
        # consec_images_copy = consec_images.clone()
        # consec_endpoints_copy = consec_endpoints.clone()
        consec_images = consec_images.unsqueeze(1)  # [T, 1, H, W]
        consec_masks = consec_masks.unsqueeze(1)  # [T, 1, H, W]

        # Apply transform
        if self.transform:
            consec_images, consec_masks, consec_endpoints, consec_cals, consec_softmasks = self.transform(consec_images, consec_masks, consec_endpoints, consec_cals, self.origin_img_size)

        # Assign labels based on the orientation of the needle
        consec_labels = self.endpoints_to_labels(consec_endpoints)

        # Squeeze
        consec_images = consec_images.squeeze(1)  # [T, H, W]
        consec_masks = consec_masks.squeeze(1)  # [T, H, W]
        sample = {
            "images": consec_images,
            "masks": consec_masks,
            "cals": consec_cals,  ## center, angle, length (x2, y2, angle, length)
            "endpoints": consec_endpoints,  ## (x1, y1, x3, y3) tensor, not tv_tensors.BoundingBoxes)
            "labels": consec_labels,  ## cls_id = -1: no needle, 0: left-top to right-bottom, 1: right-top to left-bottom
            "img_path": fname_list,  ## not sure why but this list will be size [18,B]
            "softmasks": consec_softmasks,
            "origin_img_size":self.origin_img_size,
            # "origin_images":consec_images_copy, #"origin_points":consec_endpoints_copy
        }
        return sample

## Video dataset
class VideoDataset(CustomDataset):
    def __init__(self, dir_path, transform=None, video_len=16, skip_frames=4, no_overlap=True, line_width=20, b_thres=False, det_num_classes=1, use_h5=True, add_t0=False, add_t0_per_frame=0,
                 dilate = 1):
        super(VideoDataset, self).__init__(dir_path, transform, 3, video_len, line_width, b_thres, det_num_classes, use_h5, add_t0, add_t0_per_frame)
        """
        Args:
            dir_path (string): Directory with the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            video_len (int): length of the a video sample.
                - if video_len == -1, set video_len to folder video length.
            skip_frames (int): Number of frames to skip between consecutive video samples.
            no_overlap (bool): No frame overlap between consecutive video samples.
            line_width (int): Line width of the mask image.
            b_thres (bool): Whether to use blur & thresholded mask image.
            det_num_classes (int): Number of classes for detection head.
        """
        self.add_t0_per_frame = add_t0_per_frame
        self.no_overlap = no_overlap
        self.skip_frames = skip_frames
        self.dilate = dilate  ## lower FPS
        assert self.dilate >= 1
        if video_len == -1:
            video_len = len(self.image_names) -2
            
        # 3 consecutive frames as a 3-channel input
        # n_frames: number of frames to load in a video sample
        # skip_frames: number of frames to skip between video samples loading. Set to time_len if no_overlap is True
        self.n_frames = video_len + 2
        if self.no_overlap:
            self.skip_frames = video_len

        self.set_seq_name_list()

    def set_seq_name_list(self):
        # add_t0 alraeady add "" in self.image_names in __init__
        if self.add_t0_per_frame > 0:  ## black mask added to every start of sequence
            total_num_samples = math.ceil((len(self.image_names_origin)) / self.skip_frames)
            self.consec_images_names = [
                self.image_names[i * (self.skip_frames+self.time_window-1) : i * (self.skip_frames+self.time_window-1) + self.n_frames] for i in range(0, total_num_samples)
            ]  # [["a0001.jpg", "a0002.jpg", "a0003.jpg", ... ], ["a0003.jpg", "a0004.jpg", "a0005.jpg", ... ], ...]
            self.consec_masks_names = [
                self.mask_names[i * (self.skip_frames+self.time_window-1) : i * (self.skip_frames+self.time_window-1) + self.n_frames] for i in range(0, total_num_samples)
            ]  # [["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png", ... ], ["m0003_lw_20.png", "m0004_lw_20.png", "m0005_lw_20.png", ... ], ...]
            self.consec_json_names = [
            self.json_names[i * (self.skip_frames+self.time_window-1) : i * (self.skip_frames+self.time_window-1) + self.n_frames] for i in range(0, total_num_samples)
            ]  # [["a0001.json", "a0002.json", "a0003.json", ... ], ["a0003.json", "a0004.json", "a0005.json", ... ], ...]
        elif self.dilate > 1:
            self.seq_range = self.n_frames+(self.dilate-1)*(self.n_frames-1)  ## real sequence length including dilated frames
            total_num_samples = math.ceil((len(self.image_names) - self.seq_range) / self.skip_frames)  ## number of sequence in a video
            self.consec_images_names, self.consec_masks_names, self.consec_json_names = [],[],[]
            for i in range(0, total_num_samples):  
                self.consec_images_names.append([])
                self.consec_masks_names.append([])
                self.consec_json_names.append([])
                j_start = i* self.skip_frames  ## the first idx in each sequence
                for j in range(j_start, j_start + self.seq_range, self.dilate):
                    self.consec_images_names[i].append(self.image_names[j])
                    self.consec_masks_names[i].append(self.mask_names[j])
                    self.consec_json_names[i].append(self.json_names[j])
                    if i <5:
                        print(f" {j}", end=" ")
                if i  <5:
                    print("\n", self.consec_json_names[i])
        else:    
            total_num_samples = math.ceil((len(self.image_names) - self.n_frames) / self.skip_frames) + 1
            self.consec_images_names = [self.image_names[i * self.skip_frames : i * self.skip_frames + self.n_frames] for i in range(0, total_num_samples)]
            self.consec_masks_names = [self.mask_names[i * self.skip_frames : i * self.skip_frames + self.n_frames] for i in range(0, total_num_samples)]
            self.consec_json_names = [self.json_names[i * self.skip_frames : i * self.skip_frames + self.n_frames] for i in range(0, total_num_samples)]

        # drop the last video sample if the length is less than n_frames
        if len(self.consec_images_names[-1]) < self.n_frames:
            print(f"Dataset {os.path.basename(self.dir_path)} drop the last video sample with {len(self.consec_images_names[-1])} frames.")
            self.consec_images_names = self.consec_images_names[:-1]
            self.consec_masks_names = self.consec_masks_names[:-1]
            self.consec_json_names = self.consec_json_names[:-1]        

    def revise_video_len(self, seq_len:int):
        if seq_len != self.buffer_num_sample:  ## video len
            self.buffer_num_sample = seq_len
            self.n_frames = seq_len + 2
            if self.no_overlap:
                self.skip_frames = seq_len
            self.set_seq_name_list()

## Video dataset
class SyntheticVideoDataset(CustomDataset):
    def __init__(self, dir_path, transform=None, video_len=3, line_width=20, b_thres=False, det_num_classes=1, use_h5=True, use_json=False, mask_name_suffix=None):
        super(SyntheticVideoDataset, self).__init__(dir_path, transform, 3, video_len, line_width, b_thres, det_num_classes, use_h5)
        """
        Args:
            dir_path (string): Directory with the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            video_len (int): length of the a video sample.
                - if video_len == -1, set video_len to folder video length.

            line_width (int): Line width of the mask image.
            b_thres (bool): Whether to use blur & thresholded mask image.
            det_num_classes (int): Number of classes for detection head.
        """
        if mask_name_suffix is not None:
            self.mask_name_suffix = mask_name_suffix
        self.image_names = [f for f in self.file_names if self.mask_name_suffix not in f and "Annotation" not in f]  # ["a0001.jpg", "a0002.jpg", ...]
        self.mask_names = [f for f in self.file_names if self.mask_name_suffix in f]  # ["m0001_lw_20.png", "m0002_lw_20.png", ...]
        self.image_names_origin = [f for f in self.file_names if self.mask_name_suffix not in f and "Annotation" not in f]
        self.mask_names_origin =  [f for f in self.file_names if self.mask_name_suffix in f]
        if use_json:
            self.json_names = [f.replace(".jpg", ".json") for f in self.image_names]  # ["a0001.json", "a0002.json", ...]
            self.json_names_origin = [f.replace(".jpg", ".json") for f in self.image_names]
        
        self.set_origin_size()

        # 3 consecutive frames as a 3-channel input
        # n_frames: number of frames to load in a video sample
        # skip_frames: number of frames to skip between video samples loading. Set to time_len if no_overlap is True
        self.skip_frames = 1
        self.video_len = video_len
        self.use_json = use_json
   
        # total_num_samples = math.ceil((len(self.image_names) - self.n_frames) / self.skip_frames) + 1
        self.consec_images_names = self.image_names
        self.consec_masks_names = self.mask_names
        if self.use_json:
            self.consec_json_names = self.json_names
        else:
            del self.consec_json_names
    
    def __getitem__(self, idx):
        consec_images_name = self.consec_images_names[idx]  # single str "a0001.jpg"
        if self.use_h5:
            start_name_idx = self.image_names_origin.index(consec_images_name) 
            consec_images = self.image_stack[start_name_idx]
            consec_masks = self.mask_stack[start_name_idx]
            consec_images = consec_images.expand(self.video_len,-1,-1)
            consec_masks = consec_masks.expand(self.video_len,-1,-1)
        else:
            img = Image.open(os.path.join(self.dir_path, consec_images_name)).convert("L")
            consec_images = self.trans_totensor(img).expand(self.video_len,-1,-1)  ## [T, H, W] float32
            img.close()
            
            consec_masks_name = self.consec_masks_names[idx]  # # single str "m0001_lw_20.png"
            img = Image.open(os.path.join(self.dir_path, consec_masks_name)).convert("L")
            consec_masks = self.trans_totensor(img).expand(self.video_len,-1,-1)
            img.close()

        fname_list=[os.path.join(self.dir_path, consec_images_name)] * self.video_len
            
        ## Center, Angle, Length (cal) and Cls
        if self.use_json:
            consec_json_name = self.consec_json_names[idx]  # "a0001.json"
            consec_cals, consec_endpoints = self.get_cals_endpoints(self.dir_path, [consec_json_name])
        else:
            consec_cals, consec_endpoints = [0]*self.video_len, [0]*self.video_len

        # Unsqueeze
        consec_images = consec_images.unsqueeze(1)  # [T, 1, H, W]
        consec_masks = consec_masks.unsqueeze(1)  # [T, 1, H, W]
        # consec_images_copy = consec_images.clone()
        # consec_endpoints_copy = consec_endpoints.clone()

        # Apply transform
        if self.transform:
            consec_images_list, consec_masks_list, consec_endpoints_list, consec_cals_list, consec_softmasks_list = [],[],[],[],[]
            for consec_image, consec_mask, consec_endpoint, consec_cal in zip(consec_images, consec_masks, consec_endpoints, consec_cals):
                consec_image = consec_image.unsqueeze(0)
                consec_mask = consec_mask.unsqueeze(0)
                if self.use_json:
                    consec_endpoint = consec_endpoint.unsqueeze(0)
                    consec_cal = consec_cal.unsqueeze(0)
                consec_image, consec_mask, consec_endpoint, consec_cal, _ = self.transform(consec_image, consec_mask, consec_endpoint, consec_cal, self.origin_img_size)

                consec_images_list.append(consec_image)
                consec_masks_list.append(consec_mask)
                if self.use_json:
                    consec_endpoints_list.append(consec_endpoint)
                    consec_cals_list.append(consec_cal)
                # consec_softmasks_list.append(consec_softmask)
        
            consec_images = torch.cat(consec_images_list, dim=0)
            consec_masks = torch.cat(consec_masks_list, dim=0)
            if self.use_json:
                consec_endpoints = torch.cat(consec_endpoints_list, dim=0)
                consec_cals = torch.cat(consec_cals_list, dim=0)
            # consec_softmasks = torch.cat(consec_softmasks_list, dim=0)
        # # Assign labels based on the orientation of the needle
        # consec_labels = self.endpoints_to_labels(consec_endpoints)

        # Squeeze
        consec_images = consec_images.squeeze(1)  # [T, H, W]
        consec_masks = consec_masks.squeeze(1)  # [T, H, W]
        sample = {
            "images": consec_images,
            "masks": consec_masks,
            # "labels": consec_labels,  ## cls_id = -1: no needle, 0: left-top to right-bottom, 1: right-top to left-bottom
            "img_path": fname_list,  ## not sure why but this list will be size [18,B]
            # "softmasks": consec_softmasks,
            "origin_img_size":self.origin_img_size,
            # "origin_images":consec_images_copy, "origin_points":consec_endpoints_copy
        }
        if self.use_json:
            sample["cals"]=consec_cals  ## center, angle, length (x2, y2, angle, length)]
            sample["endpoints"]=consec_endpoints  ## (x1, y1, x3, y3) tensor, not tv_tensors.BoundingBoxes)
        return sample

## Video dataset
class SyntheticNeedleVideoDataset(CustomDataset):
    def __init__(self, dir_path, transform=None, video_len=3, line_width=20, b_thres=False, det_num_classes=1, use_h5=True, use_json=False, mask_name_suffix=None):
        super(SyntheticNeedleVideoDataset, self).__init__(dir_path, transform, 3, video_len, line_width, b_thres, det_num_classes, use_h5)
        """
        Args:
            dir_path (string): Directory with the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            video_len (int): length of the a video sample.
                - if video_len == -1, set video_len to folder video length.

            line_width (int): Line width of the mask image.
            b_thres (bool): Whether to use blur & thresholded mask image.
            det_num_classes (int): Number of classes for detection head.
        """
        if mask_name_suffix is not None:
            self.mask_name_suffix = mask_name_suffix
        # self.image_names_origin = [f for f in self.file_names if self.mask_name_suffix not in f and "Annotation" not in f]
        # self.mask_names_origin =  [f for f in self.file_names if self.mask_name_suffix in f]
        # if use_json:
        #     self.json_names = [f.replace(".jpg", ".json") for f in self.image_names]  # ["a0001.json", "a0002.json", ...]
        #     self.json_names_origin = [f.replace(".jpg", ".json") for f in self.image_names]
        
        # self.set_origin_size()

        # 3 consecutive frames as a 3-channel input
        # n_frames: number of frames to load in a video sample
        # skip_frames: number of frames to skip between video samples loading. Set to time_len if no_overlap is True
        self.skip_frames = 1
        self.video_len = video_len
        self.use_json = use_json
   
        # total_num_samples = math.ceil((len(self.image_names) - self.n_frames) / self.skip_frames) + 1
        self.consec_images_names = self.image_names
        self.consec_masks_names = self.mask_names
        if self.use_json:
            self.consec_json_names = self.json_names
        else:
            del self.consec_json_names
    
    def __getitem__(self, idx):
        consec_images_name = self.consec_images_names[idx]  # single str "a0001.jpg"
        if self.use_h5:
            start_name_idx = self.image_names_origin.index(consec_images_name) 
            consec_images = self.image_stack[start_name_idx]
            consec_masks = self.mask_stack[start_name_idx]
            consec_images = consec_images.expand(self.video_len,-1,-1)
            consec_masks = consec_masks.expand(self.video_len,-1,-1)
        else:
            img = Image.open(os.path.join(self.dir_path, consec_images_name)).convert("L")
            consec_images = self.trans_totensor(img).expand(self.video_len,-1,-1)  ## [T, H, W] float32
            img.close()
            
            consec_masks_name = self.consec_masks_names[idx]  # # single str "m0001_lw_20.png"
            img = Image.open(os.path.join(self.dir_path, consec_masks_name)).convert("L")
            consec_masks = self.trans_totensor(img).expand(self.video_len,-1,-1)
            img.close()

        fname_list=[os.path.join(self.dir_path, consec_images_name)] * self.video_len
            
        ## Center, Angle, Length (cal) and Cls
        if self.use_json:
            consec_json_name = self.consec_json_names[idx]  # "a0001.json"
            consec_cals, consec_endpoints = self.get_cals_endpoints(self.dir_path, [consec_json_name])
        else:
            consec_cals, consec_endpoints = [0]*self.video_len, [0]*self.video_len

        # Unsqueeze
        consec_images_copy = consec_images.clone()
        # consec_endpoints_copy = consec_endpoints.clone()
        consec_images = consec_images.unsqueeze(1)  # [T, 1, H, W]
        consec_masks = consec_masks.unsqueeze(1)  # [T, 1, H, W]

        # Apply transform
        if self.transform:
            consec_images_list, consec_masks_list, consec_endpoints_list, consec_cals_list, consec_softmasks_list = [],[],[],[],[]
            for consec_image, consec_mask, consec_endpoint, consec_cal in zip(consec_images, consec_masks, consec_endpoints, consec_cals):
                consec_image = consec_image.unsqueeze(0)
                consec_mask = consec_mask.unsqueeze(0)
                if self.use_json:
                    consec_endpoint = consec_endpoint.unsqueeze(0)
                    consec_cal = consec_cal.unsqueeze(0)
                consec_image, consec_mask, consec_endpoint, consec_cal, _ = self.transform(consec_image, consec_mask, consec_endpoint, consec_cal, self.origin_img_size)

                consec_images_list.append(consec_image)
                consec_masks_list.append(consec_mask)
                if self.use_json:
                    consec_endpoints_list.append(consec_endpoint)
                    consec_cals_list.append(consec_cal)
                # consec_softmasks_list.append(consec_softmask)
        
            consec_images = torch.cat(consec_images_list, dim=0)
            consec_masks = torch.cat(consec_masks_list, dim=0)
            if self.use_json:
                consec_endpoints = torch.cat(consec_endpoints_list, dim=0)
                consec_cals = torch.cat(consec_cals_list, dim=0)
            # consec_softmasks = torch.cat(consec_softmasks_list, dim=0)
        # # Assign labels based on the orientation of the needle
        # consec_labels = self.endpoints_to_labels(consec_endpoints)

        # Squeeze
        consec_images = consec_images.squeeze(1)  # [T, H, W]
        consec_masks = consec_masks.squeeze(1)  # [T, H, W]
        sample = {
            "images": consec_images,
            "masks": consec_masks,
            # "labels": consec_labels,  ## cls_id = -1: no needle, 0: left-top to right-bottom, 1: right-top to left-bottom
            "img_path": fname_list,  ## not sure why but this list will be size [18,B]
            # "softmasks": consec_softmasks,
            "origin_img_size":self.origin_img_size,
            "origin_images":consec_images_copy, # "origin_points":consec_endpoints_copy
        }
        if self.use_json:
            sample["cals"]=consec_cals  ## center, angle, length (x2, y2, angle, length)]
            sample["endpoints"]=consec_endpoints  ## (x1, y1, x3, y3) tensor, not tv_tensors.BoundingBoxes)
        return sample



# Augmentation Class
class Augmentation(nn.Module):
    def __init__(self, crop=True, rotate=True, color_jitter=True, horizontal_flip=True, perspective=False, affine=False, gaussian_noise=False, image_size=224,
                        active_rotate = False, needle_cutmix = False, soft_mask=False):
        self.crop = crop
        self.rotate = rotate
        self.color_jitter = color_jitter
        self.gaussian_noise = gaussian_noise
        self.horizontal_flip = horizontal_flip
        self.image_size = image_size
        self.active_rotate = active_rotate
        self.max_delta_angle = 5
        self.perspective = perspective
        self.affine = affine
        self.max_perspective_dist = int(0.15 * self.image_size)
        self.needle_cutmix = needle_cutmix
        self.soft_mask = soft_mask

    def rotate_endpoints(self, endpoints, angle, center=(112, 112)):
        if endpoints.dim() == 1:
            x1, y1 = endpoints[0], endpoints[1]
            x3, y3 = endpoints[2], endpoints[3]
            x1, y1 = self.rotate_point(x1, y1, angle, center)
            x3, y3 = self.rotate_point(x3, y3, angle, center)
            new_coords = torch.tensor([x1, y1, x3, y3])
        else:
            x1, y1 = endpoints[:, 0], endpoints[:, 1]
            x3, y3 = endpoints[:, 2], endpoints[:, 3]
            x1, y1 = self.rotate_point(x1, y1, angle, center)
            x3, y3 = self.rotate_point(x3, y3, angle, center)
            new_coords = torch.stack([x1, y1, x3, y3], dim=1)
        return new_coords

    def rotate_point(self, x, y, angle, center=(112, 112)):
        x = x - center[0]
        y = y - center[1]
        theta = math.radians(angle)
        new_x = x * math.cos(theta) + y * math.sin(theta) + center[0]
        new_y = -(x * math.sin(theta)) + y * math.cos(theta) + center[1]
        return new_x, new_y

    def flip_endpoints(self, endpoints):  ## v2.functional.horizontal_flip gives wrong direction
        if endpoints.dim() == 1:
            new_x1 = self.image_size - endpoints[0]
            new_x3 = self.image_size - endpoints[2]
            new_coords = torch.tensor([new_x1, endpoints[1], new_x3, endpoints[3]])
        else:
            new_x1 = self.image_size - endpoints[:, 0]
            new_x3 = self.image_size - endpoints[:, 2]
            new_coords = torch.stack([new_x1, endpoints[:, 1], new_x3, endpoints[:, 3]], dim=1)
        return new_coords


    def compute_perspective_transform(self,startpoints, endpoints):
        """
        Compute the perspective transformation matrix given startpoints and endpoints.
        """
        # Convert points to tensor format
        startpoints = torch.tensor(startpoints, dtype=torch.float32)
        endpoints = torch.tensor(endpoints, dtype=torch.float32)

        # Compute the transformation matrix
        matrix = tf.functional._get_perspective_coeffs(startpoints, endpoints)
        matrix = list(matrix) + [1.]
        matrix = torch.tensor(matrix).reshape(3, 3)
        
        return matrix
    
    def transform_points_matrix(self, points_matrix, matrix):
        """
        Transform a matrix of points with shape (N, 4) using the given transformation matrix.
        Each row in the matrix contains two points [x1, y1, x2, y2].
        """
        # Reshape the points matrix to (2N, 2) to handle all points at once
        points = points_matrix.reshape(-1, 2)
        
        # Convert points to homogeneous coordinates by adding a column of ones
        ones = torch.ones(points.shape[0], 1, dtype=torch.float32)
        homogeneous_points = torch.cat([points, ones], dim=1)
        
        # Apply the transformation matrix to all points
        transformed_points = (matrix @ homogeneous_points.T).T
        
        denom = transformed_points[:, 2].clone()
        # Convert back from homogeneous to Cartesian coordinates
        transformed_points /= denom.unsqueeze(1)  # Normalize by the third coordinate
        
        # Reshape back to (N, 4)
        transformed_points = transformed_points[:, :2].reshape(-1, 4)
        
        return transformed_points

    def get_random_affine_params(self, degrees, translate=None, scale_ranges=None, shear=None):
        # Generate random rotation angle
        angle = torch.empty(1).uniform_(-degrees, degrees).item()
        
        # Generate random translation
        if translate is not None:
            max_dx = translate[0]
            max_dy = translate[1]
            translations = (torch.empty(1).uniform_(-max_dx, max_dx).item(),
                            torch.empty(1).uniform_(-max_dy, max_dy).item())
        else:
            translations = (0, 0)
        
        # Generate random scale
        if scale_ranges is not None:
            scale = torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item()
        else:
            scale = 1.0
        
        # Generate random shear
        if shear is not None:
            shear_x = torch.empty(1).uniform_(-shear, shear).item()
            shear_y = torch.empty(1).uniform_(-shear, shear).item()
            shear = (shear_x, shear_y)
        else:
            shear = (0.0, 0.0)
        
        return angle, translations, scale, shear

    def __call__(self, images, masks, endpoints, cals, origin_img_size=None):  ## endpoints type: tv_tensors.BoundingBoxes

        softmasks = 0
        if isinstance(endpoints, torch.Tensor):
            if origin_img_size is not None:
                endpoints = tv_tensors.BoundingBoxes(endpoints, format="XYXY", canvas_size=[origin_img_size, origin_img_size])  ## [T, 4] 
            else:
                endpoints = tv_tensors.BoundingBoxes(endpoints, format="XYXY", canvas_size=images.shape[-2:])  ## [T, 4] 

            ## Resize bbox first if size not match
            if origin_img_size is not None and origin_img_size != images.shape[-1]:
                endpoints = v2.functional.resize(endpoints, (images.shape[-2], images.shape[-1]), interpolation=tf.InterpolationMode.BILINEAR)

        """ Random Resized Crop """
        if self.crop and random.random() < 0.5:

            # Get parameters for RandomResizedCrop
            ### Note: Due to bias in mask and upper points
            ### large ratio may provide more vertical bias; while small ratio may provide horizontal bias
            top, left, height, width = tf.RandomResizedCrop.get_params(images, scale=(0.3, 1.0), ratio=(0.9, 1.1))  ## ratio:w/h

            # Apply Crop
            images = images[:, :, top : top + height, left : left + width]  ## (... ,Y1:Y2 , X1:X2)
            masks = masks[:, :, top : top + height, left : left + width]

            if isinstance(endpoints, torch.Tensor):
                for r in range(endpoints.shape[0]):
                    newpoints = 按斜率滑動到裁剪範圍內(endpoints[r].tolist(), left, top, left + width, top + height)
                    newpoints = 轉换到裁剪後座標系(newpoints, left, top)
                    endpoints[r][0], endpoints[r][1], endpoints[r][2], endpoints[r][3] = (
                        newpoints[0][0],
                        newpoints[0][1],
                        newpoints[1][0],
                        newpoints[1][1],
                    )

                endpoints.canvas_size = (height, width)  ## reset the world size of bbox
                # print(endpoints)

        """Resize"""  
        ## do not resize at the begining to avoid distortion
        ## soft mask resize before mask resize
        images = v2.functional.resize(images, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.BILINEAR, antialias=True)
        if self.soft_mask:
            softmasks = v2.functional.resize(masks, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.BILINEAR)
        masks = v2.functional.resize(masks, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.NEAREST)
        if isinstance(endpoints, torch.Tensor):
            endpoints = v2.functional.resize(endpoints, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.BILINEAR)

        """Perspective"""
        if self.perspective and random.random() < 0.5:
            ## top-left, top-right, bottom-right, bottom-left
            corner_delta = [random.randint(0,self.max_perspective_dist) for i in range(8)]
            startpoints = [[0,0],[self.image_size,0],[self.image_size,self.image_size],[0,self.image_size]] 
            perspective_endpoints = [[corner_delta[0],corner_delta[1]],
                                     [self.image_size - corner_delta[2],corner_delta[3]],
                                     [self.image_size - corner_delta[4],self.image_size - corner_delta[5]],
                                     [corner_delta[6],self.image_size - corner_delta[7]]] 

            # print(corner_delta)
            # print(perspective_endpoints)
            images = v2.functional.perspective(images, startpoints, perspective_endpoints, interpolation=tf.InterpolationMode.BILINEAR)
            masks = v2.functional.perspective(masks, startpoints, perspective_endpoints, interpolation=tf.InterpolationMode.NEAREST)
            if self.soft_mask:
                softmasks = v2.functional.perspective(softmasks, startpoints, perspective_endpoints, interpolation=tf.InterpolationMode.BILINEAR)

            ## v2.functional.perspective(endpoints) does not work
            if isinstance(endpoints, torch.Tensor):
                matrix = self.compute_perspective_transform(startpoints, perspective_endpoints)
                inverse_matrix = torch.inverse(matrix)
                endpoints = self.transform_points_matrix(endpoints, inverse_matrix)

        """Affine"""
        # Function to generate random affine parameters
        if self.affine and random.random() < 0.5:
            # Generate random affine parameters
            angle, translations, scale, shear_params = self.get_random_affine_params(degrees=0, translate=(0.1, 0.1), 
                                                                                    scale_ranges=(0.9, 1.1), shear=10)
            # Apply the same affine transform to both images
            images = tf.functional.affine(
                images, angle, translations, scale, shear_params, interpolation=tf.InterpolationMode.BILINEAR
            )
            masks = tf.functional.affine(
                masks, angle, translations, scale, shear_params, interpolation=tf.InterpolationMode.NEAREST
            )

        """ Random Rotation """
        rotate_random = random.random()
        if self.rotate and rotate_random < 0.5:
            angle = random.randint(-15, 15)
            images = v2.functional.rotate(images, angle, interpolation=tf.InterpolationMode.BILINEAR)
            masks = v2.functional.rotate(masks, angle, interpolation=tf.InterpolationMode.NEAREST)
            if self.soft_mask:
                softmasks = v2.functional.rotate(softmasks, angle, interpolation=tf.InterpolationMode.BILINEAR)
            if isinstance(endpoints, torch.Tensor):
                for r in range(endpoints.shape[0]):
                    newpoints = self.rotate_endpoints(endpoints[r], angle, center=(self.image_size // 2, self.image_size // 2))
                    newpoints = 按斜率滑動到裁剪範圍內(newpoints, 0, 0, self.image_size, self.image_size)
                    endpoints[r][0], endpoints[r][1], endpoints[r][2], endpoints[r][3] = (
                        newpoints[0][0],
                        newpoints[0][1],
                        newpoints[1][0],
                        newpoints[1][1],
                    )

        # """Random Active Rotation"""
        # elif self.rotate and rotate_random >= 0.7:  ##0.7
        #     ## update active angle
        #     # angles = []
        #     cur_angle = random.uniform(-15, 15)
        #     clockwise = True
        #     for i in range(endpoints.shape[0]):
        #         if i != 0:
        #             if clockwise:
        #                 if random.random() > 0.25:
        #                     angle_delta = random.uniform(0, self.max_delta_angle)
        #                 else:
        #                     angle_delta = random.uniform(-self.max_delta_angle, 0)
        #                     clockwise = False
        #             else:
        #                 if random.random() < 0.25:
        #                     angle_delta = random.uniform(0, self.max_delta_angle)
        #                     clockwise = True
        #                 else:
        #                     angle_delta = random.uniform(-self.max_delta_angle, 0)
        #             cur_angle += angle_delta # random.uniform(-10, 10)
        #         ## one by one rotate
        #         cur_img, cur_mask = images[i], masks[i]
        #         cur_img = v2.functional.rotate(cur_img, cur_angle, interpolation=tf.InterpolationMode.BILINEAR)
        #         cur_mask = v2.functional.rotate(cur_mask, cur_angle, interpolation=tf.InterpolationMode.NEAREST)
        #         images[i] = cur_img
        #         masks[i] = cur_mask

        #         newpoints = self.rotate_endpoints(endpoints[i], cur_angle, center=(self.image_size // 2, self.image_size // 2))
        #         newpoints = 按斜率滑動到裁剪範圍內(newpoints, 0, 0, self.image_size, self.image_size)
        #         endpoints[i][0], endpoints[i][1], endpoints[i][2], endpoints[i][3] = (
        #             newpoints[0][0],
        #             newpoints[0][1],
        #             newpoints[1][0],
        #             newpoints[1][1],
        #         )
        #     #     angles.append(cur_angle)
        #     # print(angles)

        """ Random Color Jitter"""
        if self.color_jitter and random.random() < 0.5:
            color_transform = tf.Compose(
                [
                    tf.ColorJitter(brightness=0.5, contrast=0.3),
                ]
            )
            images = color_transform(images)

        if self.gaussian_noise and random.random():
            std, mean = 0.05, 0.0
            noise = torch.randn_like(images) * std + mean
            # Add noise to the image
            images = images + noise
            images = torch.clamp(images, 0.0, 1.0)

        """ Random Horizontal Flip """
        if self.horizontal_flip and random.random() < 0.5:
            images = tf.functional.hflip(images)
            masks = tf.functional.hflip(masks)
            if self.soft_mask:
                softmasks = tf.functional.hflip(softmasks)
            if isinstance(endpoints, torch.Tensor):
                endpoints = self.flip_endpoints(endpoints)  ## tensor

        ## endpoints (0,0,0,0) may be scaled to (224,0,224,0)
        ## check if endpoints should not exists, reset to (0,0,0,0)
        if isinstance(endpoints, torch.Tensor):
            for r in range(endpoints.shape[0]):
                if endpoints[r][0] == endpoints[r][2] and endpoints[r][1] == endpoints[r][3]:
                    endpoints[r][0], endpoints[r][1], endpoints[r][2], endpoints[r][3] = 0.0, 0.0, 0.0, 0.0

            ## update bbox (center x, center y, theta, len)
            cals = get_center_angle_length(endpoints)

        return images, masks, endpoints, cals, softmasks


# Augmentation Class if no detection head
class AugmentationImgMaskOnly(nn.Module):
    def __init__(self, resized_crop=True, color_jitter=True, horizontal_flip=True, rotate=True, image_size=224):
        self.resized_crop = resized_crop
        self.color_jitter = color_jitter
        self.horizontal_flip = horizontal_flip
        self.image_size = image_size
        self.rotate = rotate

    def __call__(self, images, masks):

        masks = v2.functional.resize(masks, (images.shape[-2], images.shape[-1]), interpolation=tf.InterpolationMode.NEAREST)
        """ Random Resized Crop """
        if self.resized_crop and random.random() < 0.5:

            # Get parameters for RandomResizedCrop
            ### Note: Due to bias in mask and upper points
            ### large ratio may provide more vertical bias; while small ratio may provide horizontal bias
            top, left, height, width = tf.RandomResizedCrop.get_params(images, scale=(0.9, 1.0), ratio=(0.9, 1.1))  ## ratio:w/h

            # Apply Crop
            images = images[:, :, top : top + height, left : left + width]  ## (... ,Y1:Y2 , X1:X2)
            masks = masks[:, :, top : top + height, left : left + width]

        """Resize"""  ## do not resize at the begining to avoid distortion
        images = v2.functional.resize(images, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.BILINEAR, antialias=True)
        softmasks = v2.functional.resize(masks, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.BILINEAR)
        masks = v2.functional.resize(masks, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.NEAREST)
        
        if self.rotate and random.random() < 0.5:
            angle = random.randint(-10,10)
            images = v2.functional.rotate(images, angle, interpolation=tf.InterpolationMode.BILINEAR)
            masks = v2.functional.rotate(masks, angle, interpolation=tf.InterpolationMode.NEAREST)
            softmasks = v2.functional.rotate(softmasks, angle,  interpolation=tf.InterpolationMode.BILINEAR)
        
        """ Random Color Jitter"""
        if self.color_jitter and random.random() < 0.5:
            color_transform = tf.Compose(
                [
                    tf.ColorJitter(brightness=0.5, contrast=0.3),
                ]
            )
            images = color_transform(images)

        """ Random Horizontal Flip """
        if self.horizontal_flip and random.random() < 0.5:
            images = tf.functional.hflip(images)
            masks = tf.functional.hflip(masks)
            softmasks = tf.functional.hflip(softmasks)

        return images, masks


"""For Semi Supervise Pseudo Labeling"""


class UnlabeledDataset(CustomDataset):
    def __init__(self, dir_path, transform=None, time_window=3, buffer_num_sample=8, line_width=20, b_thres=False, det_num_classes=1, use_h5=False, add_t0=False):
        super(UnlabeledDataset, self).__init__(dir_path, transform, time_window, buffer_num_sample,
                                               line_width, b_thres, det_num_classes, use_h5, add_t0)
        """
        Args:
            dir_path (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        del self.consec_masks_names, self.consec_json_names, self.mask_names, self.json_names
        self.labeled = False
        if self.transform and self.transform.needle_cutmix:
            self.labeled = True
            self.pre_transform = AugmentationImgMaskOnly(image_size=1758)
            # self.needle_pre_transform = AugmentationImgMaskOnly(resized_crop=False, horizontal_flip=False,rotate=False,image_size=1758)

    def __len__(self):
        return len(self.consec_images_names)

    def __getitem__(self, idx):
        consec_images_name = self.consec_images_names[idx]  # ["a0001.jpg", "a0002.jpg", "a0003.jpg"]
        consec_images = []
        fname_list = []
        add_empty_mask = 0
        for f_name in consec_images_name:
            if f_name == "":
                fname_list.append("")
                add_empty_mask += 1
                continue
            img = Image.open(os.path.join(self.dir_path, f_name)).convert("L")
            img_tensor = self.trans_totensor(img)
            consec_images.append(img_tensor)
            fname_list.append(os.path.join(self.dir_path, f_name))
            img.close()
        consec_images = torch.stack(consec_images, dim=0)  ## [T, 1, H, W]
        consec_images_copy = consec_images.clone().squeeze(1)
        
        if add_empty_mask>0:
            first_image = consec_images[0].unsqueeze(0)
            if add_empty_mask == 2:
                consec_images = torch.stack([first_image, first_image, consec_images], dim=0)
            elif add_empty_mask == 1:
                consec_images = torch.stack([first_image, consec_images], dim=0)
        origin_img_size = consec_images.shape[-1]

        ## Synthesis from other folder
        if self.labeled:
            
            ## transform background
            temp = torch.zeros_like(consec_images)
            consec_images, _ = self.pre_transform(consec_images, temp)

            ## random pick other folder sequence
            ref_folder_path, consec_cut_images_name, consec_cut_masks_name, consec_masks_name, consec_json_name = self.get_cut_imgs_gts_name(seq_len=len(consec_images_name))

            ## get images to cut
            consec_cut_images = []
            for f_name in consec_cut_images_name:
                img = Image.open(os.path.join(ref_folder_path, f_name)).convert("L")
                img_tensor = self.trans_totensor(img)
                consec_cut_images.append(img_tensor)
                img.close()
            consec_cut_images = torch.stack(consec_cut_images, dim=0)  ## [T, 1, H, W]
            # consec_cut_images, _ = self.needle_pre_transform(consec_cut_images, temp)

            ## get thresholded gt masks
            consec_thres_masks = []
            for f_name in consec_cut_masks_name:
                img = Image.open(os.path.join(ref_folder_path, f_name)).convert("L")
                img_tensor = self.trans_totensor(img)
                consec_thres_masks.append(img_tensor)
                img.close()
            consec_thres_masks = torch.stack(consec_thres_masks, dim=0)  ## [T, 1, H, W]

            ## add needle pixels to img
            consec_images = needle_cutmix(consec_images, consec_cut_images, consec_thres_masks)
            # del consec_thres_masks, consec_cut_images

            ## get gt masks
            consec_masks = []
            for f_name in consec_masks_name:
                img = Image.open(os.path.join(ref_folder_path, f_name)).convert("L")
                img_tensor = self.trans_totensor(img)
                consec_masks.append(img_tensor)
                img.close()
            consec_masks = torch.stack(consec_masks, dim=0)  ## [T, 1, H, W]

            ## get gt Center, Angle, Length (cal)
            consec_cals, consec_endpoints = self.get_cals_endpoints(ref_folder_path, consec_json_name) 

        if not self.labeled:
            consec_masks = torch.zeros_like(consec_images)
            consec_endpoints, consec_cals = torch.zeros([3, 4]), torch.zeros([3, 4])
        
            # Apply transform
            if self.transform:
                consec_images, _, _, _, _ = self.transform(consec_images, consec_masks, consec_endpoints, consec_cals)
        
        elif self.transform:
            consec_images, consec_masks, consec_endpoints, consec_cals, consec_softmasks = self.transform(consec_images, consec_masks, consec_endpoints, consec_cals)
            consec_masks = consec_masks.squeeze(1)  # [T, H, W]
            consec_labels = self.endpoints_to_labels(consec_endpoints)

        # Squeeze
        consec_images = consec_images.squeeze(1)  # [T, H, W]

        sample = {
            "images": consec_images,
            "img_names": " ".join(consec_images_name),  ## "a0000.jpg a0001.jpg a0002.jpg"
            "img_folder_dir": self.dir_path,  ## folder directory of images
            "img_path": fname_list,
            "origin_img_size": origin_img_size,
            "origin_images":consec_images_copy,
        }
        if self.labeled:
            sample = {"images": consec_images,
                        "masks": consec_masks,
                        "cals": consec_cals,  ## center, angle, length (x2, y2, angle, length)
                        "endpoints": consec_endpoints,  ## (x1, y1, x3, y3) tensor, not tv_tensors.BoundingBoxes)
                        "labels": consec_labels,  ## cls_id = -1: no needle, 0: left-top to right-bottom, 1: right-top to left-bottom
                        "img_path": fname_list,
                        "softmasks": consec_softmasks,
                        "origin_img_size": origin_img_size,
                        # "consec_thres_masks":consec_thres_masks, "consec_cut_images": consec_cut_images
                        }
        return sample

    def set_reference_candidate_folder(self, folders, config):
        self.ref_root_path = config["Data"]["folder_dir"]
        if isinstance(folders, list) and isinstance(folders[0], dict):
            self.ref_candidate_folders = []
            for folder_dict in folders:
                self.ref_candidate_folders += list(folder_dict.values())
        elif isinstance(folders, list) and isinstance(folders[0], str):
            self.ref_candidate_folders = folders
        elif isinstance(folders, dict):
            self.ref_candidate_folders = list(folders.values())

    def get_cut_imgs_gts_name(self, seq_len=3):
        ## random select a folder
        ref_folder = random.choice(self.ref_candidate_folders)
        ref_folder_path = os.path.join(self.ref_root_path, ref_folder)
        
        ## get seq from ref folder
        file_names = sorted(os.listdir(ref_folder_path))
        candidate_image_names = [f for f in file_names if f[0] == "a" and f.endswith(".jpg")]  # ["a0001.jpg", "a0002.jpg", ...]
        start_cut_frame = random.randint(0, len(candidate_image_names)-seq_len)
        
        consec_cut_images_name = candidate_image_names[start_cut_frame:start_cut_frame+seq_len]
        consec_cut_masks_name = ["m" + f[1:-4] + "_bt_lw_20.png" for f in consec_cut_images_name]  # ["m0001_lw_20.png", "m0002_lw_20.png", ...]

        consec_masks_name = ["m" + f[1:-4] + self.mask_name_suffix for f in consec_cut_images_name]  # ["m0001_lw_20.png", "m0002_lw_20.png", ...]
        consec_json_name = [f.replace(".jpg", ".json") for f in consec_cut_images_name]  # ["a0001.json", "a0002.json", ...]
        
        return ref_folder_path, consec_cut_images_name, consec_cut_masks_name, consec_masks_name, consec_json_name


class UnlabeledLCRDataset(Dataset):
    def __init__(self, dir_path, transform=None, time_window=3):
        super(UnlabeledLCRDataset, self).__init__()
        """
        Args:
            dir_path (str): Path to the dataset directory. Should include L, C, R subfolders
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir_path = dir_path
        self.lcr_dir_path = [dir_path+"/L", dir_path+"/C", dir_path+"/R"]
        self.transform = transform
        self.time_window = time_window
        self.dilate = 1
        self.file_names = sorted(os.listdir(self.dir_path+"/L"))
        self.image_names = [f for f in self.file_names if f[0] == "a" and f.endswith(".jpg")]  # ["a0001.jpg", "a0002.jpg", ...]
        self.image_names_origin = [f for f in self.file_names if f[0] == "a" and f.endswith(".jpg")]
        self.set_origin_size()

        # creat T consecutive image & mask names list
        self.consec_images_names = [
            [self.image_names[i]]*3 for i in range(0, len(self.image_names))
        ]  # [["a0001.jpg", "a0001.jpg", "a0001.jpg"], ["a0002.jpg", "a0002.jpg", "a0002.jpg"], ...]
            
        self.labeled = False
        self.trans_totensor = tf.Compose([tf.ToTensor()])

    def __len__(self):
        return len(self.consec_images_names)

    def __getitem__(self, idx):
        consec_images_name = self.consec_images_names[idx]  # ["a0001.jpg", "a0002.jpg", "a0003.jpg"]
        consec_images = []
        fname_list = []
        for f_name, dir_path in zip(consec_images_name, self.lcr_dir_path):
            img = Image.open(os.path.join(dir_path, f_name)).convert("L")
            img_tensor = self.trans_totensor(img)
            consec_images.append(img_tensor)
            fname_list.append(os.path.join(dir_path, f_name))
            img.close()
        consec_images = torch.stack(consec_images, dim=0)  ## [T, 1, H, W]
        consec_images_copy = consec_images.clone().squeeze(1)
        
        origin_img_size = consec_images.shape[-1]
        print('[fname_list]', fname_list)

        if not self.labeled:
            consec_masks = torch.zeros_like(consec_images)
            consec_endpoints, consec_cals = torch.zeros([3, 4]), torch.zeros([3, 4])
        
            # Apply transform
            if self.transform:
                consec_images, _, _, _, _ = self.transform(consec_images, consec_masks, consec_endpoints, consec_cals)
        
        # Squeeze
        consec_images = consec_images.squeeze(1)  # [T, H, W]

        sample = {
            "images": consec_images,
            "img_names": " ".join(consec_images_name),  ## "a0000.jpg a0001.jpg a0002.jpg"
            "img_folder_dir": self.dir_path,  ## folder directory of images
            "img_path": fname_list,
            "origin_img_size": origin_img_size,
            "origin_images":consec_images_copy,
        }
        return sample
    
    def set_origin_size(self):
        if len(self.image_names) > 0:
            assert os.path.exists(os.path.join(self.dir_path+"/L", self.image_names[0]))
            img = Image.open(os.path.join(self.dir_path+"/L", self.image_names[0])).convert("L")
            self.origin_img_size = img.size[-1]
            img.close()


class PseudoDataset(Dataset):
    def __init__(self, csv_dir=None, transform=None, time_window=3):
        """
        Args:
            csv_dir (str): The csv that updates image, mask path and confidence,
                            with columns <img_root> <img_names> <mask_path> <confidence>.
            transform (callable, optional): Optional transform to be applied on a sample.

            add <pl> in mask or json name for pseudo label
        """
        self.csv_dir = csv_dir
        self.time_window = time_window

        ## initialize consec names and confidence
        self.update_df_from_csv()

        self.transform = transform
        self.trans_totensor = tf.Compose([tf.ToTensor()])

    def __len__(self):
        return len(self.consec_images_names)

    def update_df_from_csv(self):
        self.df = pd.read_csv(self.csv_dir)
        self.img_roots = self.df["img_root"].tolist()
        self.image_names = self.df["img_names"].tolist()
        self.mask_paths = self.df["mask_path"].tolist()
        # self.json_names = [f.replace(".png", ".json") for f in self.mask_names]   ### Note: assume same name with mask
        self.pl_confidence = self.df["confidence"].tolist()

        # creat T consecutive image & mask & json names list
        self.consec_images_names = [names.split(" ") for names in self.image_names]
        # [["a0001.jpg", "a0002.jpg", "a0003.jpg"], ["a0002.jpg", "a0003.jpg", "a0004.jpg"], ...]
        self.consec_masks_names = [
            [self.mask_paths[i]] * 3 for i in range(0, len(self.mask_paths))
        ]  # [["root/m0001_pl.png", "root/m0002_pl.png", "root/m0003_pl.png"], ["root/m0002_pl.png", "root/m0003_pl.png", "root/m0004_pl.png"], ...]
        # self.consec_json_names = [[self.json_names[i]] * 3 for i in range(0, len(self.json_names))]
        #   ## [["m0001_pl.json", "m0002_pl.json", "m0003_pl.json"], ["m0002_pl.json", "m0003_pl.json", "m0004_pl.json"], ...]

        return

    def __getitem__(self, idx):
        consec_images_name = self.consec_images_names[idx]  # ["a0001.jpg", "a0002.jpg", "a0003.jpg"]
        consec_images = []
        fname_list = []
        for f_name in consec_images_name:
            img = Image.open(os.path.join(self.img_roots[idx], f_name)).convert("L")
            img_tensor = self.trans_totensor(img)
            consec_images.append(img_tensor)
            fname_list.append(os.path.join(self.img_roots[idx], f_name))
            img.close()
        consec_images = torch.cat(consec_images, dim=0)  ## [T, H, W]

        consec_masks_name = self.consec_masks_names[idx]  # ["root/m0001_pl.png", "root/m0001_pl.png", "root/m0001_pl.png"]
        consec_masks = []
        for f_name in consec_masks_name:
            img = Image.open(f_name).convert("L")
            img_tensor = self.trans_totensor(img)
            consec_masks.append(img_tensor)
            img.close()
        consec_masks = torch.cat(consec_masks, dim=0)  ## [T, H, W]

        # ## TODO: pseudo label only train on seg branch?
        # ## Center, Angle, Length (cal) and Cls
        # consec_json_name = self.consec_json_names[idx]  # ["m0001_pl.json", "m0002_pl.json", "m0003_pl.json"]
        # consec_cals = []
        # consec_endpoints = []
        # consec_labels = []
        # for f_name in consec_json_name:
        #     with open(os.path.join(self.pl_dir, f_name), "r") as f:
        #         js = json.load(f)
        #         # print(js)
        #     if "shapes" in js and len(js["shapes"]) >= 0:
        #         cal = [js["shapes"]["center"][0], js["shapes"]["center"][1], js["shapes"]["theta"], js["shapes"]["length"]]
        #         endpoint = [
        #             js["shapes"]["points"][0][0],
        #             js["shapes"]["points"][0][1],
        #             js["shapes"]["points"][1][0],
        #             js["shapes"]["points"][1][1],
        #         ]
        #     else:
        #         cal = [0, 0, 0, 0]
        #         endpoint = [0, 0, 0, 0]
        #         # label = -1
        #     consec_cals.append(torch.as_tensor(cal, dtype=torch.float32))
        #     consec_endpoints.append(torch.as_tensor(endpoint, dtype=torch.float32))
        #     # consec_labels.append(torch.as_tensor(label, dtype=torch.float32))
        #     f.close()
        # consec_cals = torch.stack(consec_cals, dim=0)  ## [T, 4]
        # consec_endpoints = torch.stack(consec_endpoints, dim=0)  ## [T, 4]
        # # consec_labels = torch.stack(consec_labels, dim=0).long()  ## [T,]

        # Unsqueeze
        consec_images = consec_images.unsqueeze(1)  # [T, 1, H, W]
        consec_masks = consec_masks.unsqueeze(1)  # [T, 1, H, W]

        # consec_endpoints = torch.zeros([self.time_window, 4])
        # consec_cals = torch.zeros([self.time_window, 4])

        # Apply transform
        if self.transform:
            consec_images, consec_masks = self.transform(consec_images, consec_masks)

        # # Assign labels based on the orientation of the needle
        # for t in range(consec_endpoints.shape[0]):
        #     if consec_endpoints[t, :].sum() == 0:
        #         consec_labels.append(torch.as_tensor(-1, dtype=torch.float32))  ## cls_id = -1: no needle
        #     elif torch.sign(consec_endpoints[t][0] - consec_endpoints[t][2]) == torch.sign(consec_endpoints[t][1] - consec_endpoints[t][3]):
        #         consec_labels.append(torch.as_tensor(0, dtype=torch.float32))  ## cls_id = 0: left-top to right-bottom
        #     else:
        #         consec_labels.append(torch.as_tensor(1, dtype=torch.float32))  ## cls_id = 1: right-top to left-bottom
        # consec_labels = torch.stack(consec_labels, dim=0).long()  ## [T,]

        # Squeeze
        consec_images = consec_images.squeeze(1)  # [T, H, W]
        consec_masks = consec_masks.squeeze(1)  # [T, H, W]

        sample = {
            "images": consec_images,
            "masks": consec_masks,
            "cals": torch.tensor([-1, -1, -1, -1]),  ## center, angle, length (x2, y2, angle, length)
            "endpoints": torch.tensor([-1, -1, -1, -1]),  ## (x1, y1, x3, y3) tensor, not tv_tensors.BoundingBoxes)
            "labels": -2,  ## cls_id = -1: no needle, 0: left-top to right-bottom, 1: right-top to left-bottom
            "img_path": fname_list,  ## (path_t1, path_t2, path_t3)
        }
        return sample

def needle_cutmix(images_a, images_b, masks):
    """
    Apply CutMix augmentation by cutting a masked area from images B and pasting it onto images A.

    Args:
    - images_a (list of torch.Tensor): List of images A (NxCxHxW).
    - images_b (list of torch.Tensor): List of images B (NxCxHxW).
    - masks (list of torch.Tensor): List of binary masks (Nx1xHxW) with values 0 or 1.

    Returns:
    - final_images_a (list of torch.Tensor): List of images A after applying CutMix.
    """

    # Ensure the lists have the same length
    assert len(images_a) == len(images_b) == len(masks), "All lists must have the same length."

    final_images_a = []

    for img_a, img_b, mask in zip(images_a, images_b, masks):
        # Ensure the images and mask have the same height and width
        # assert img_a.shape == img_b.shape == mask.shape, "Images and masks must have the same dimensions."
        img_a = v2.functional.resize(img_a, (1758, 1758))
        img_b = v2.functional.resize(img_b, (1758, 1758))
        mask = v2.functional.resize(mask, (1758, 1758), interpolation=tf.InterpolationMode.BILINEAR)  ## change back to nearest?

        # Cut the masked area from image B
        cut_area = img_b * mask

        # Paste it onto image A
        combined_img = img_a * (1 - mask) + cut_area

        # Append the modified image A to the list
        final_images_a.append(combined_img)
    
    final_images_a = torch.stack(final_images_a)  ## [T,1,H,W]
    return final_images_a


"""Augmentation and endpoints calculation helper"""
def 按斜率滑動到裁剪範圍內(points, X1, Y1, X2, Y2):  # points「按斜率滑動」到crop範圍內
    x1, y1 = points[0], points[1]
    x2, y2 = points[2], points[3]
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    slope = max(y2 - y1, 1) / (x2 - x1) if x2 != x1 else float("inf")

    if x1 < X1:
        y1 += slope * (X1 - x1)
        x1 = X1
    if x1 > X2:
        y1 += slope * (X2 - x1)
        x1 = X2
    if y1 < Y1:
        x1 += (Y1 - y1) / slope if slope != float("inf") else 0
        y1 = Y1
    if y1 > Y2:
        x1 += (Y2 - y1) / slope if slope != float("inf") else 0
        y1 = Y2
    if x2 < X1:
        y2 -= slope * (x2 - X1)
        x2 = X1
    if x2 > X2:
        y2 -= slope * (x2 - X2)
        x2 = X2
    if y2 < Y1:
        x2 -= (y2 - Y1) / slope if slope != float("inf") else 0
        y2 = Y1
    if y2 > Y2:
        x2 -= (y2 - Y2) / slope if slope != float("inf") else 0
        y2 = Y2

    return [[max(min(x1, X2), X1), max(min(y1, Y2), Y1)], [max(min(x2, X2), X1), max(min(y2, Y2), Y1)]]


def 轉换到裁剪後座標系(points, X1, Y1):
    # 用新的原點描述滑動好的points 在前面輸入裁剪範圍時就排序大小X1<X2, Y1<Y2, (X1,Y1)top-left corner of the crop area becomes the new origin (0, 0)
    return [[x - X1, y - Y1] for x, y in points]


def 計算中心點和角度和長度(points):
    x1, y1 = points[0], points[1]
    x2, y2 = points[2], points[3]
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    if x2 == x1:  # 確保不會除以零
        theta = math.pi / 2 if y2 > y1 else -math.pi / 2
    else:
        theta = math.atan((y2 - y1) / (x2 - x1))
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return torch.tensor([x_center, y_center, theta, length])


def get_center_angle_length(points):
    cals = torch.zeros_like(points, dtype=torch.float32)  ## set dtype to float to avoid error
    x1, y1 = points[:, 0], points[:, 1]
    x2, y2 = points[:, 2], points[:, 3]
    cals[:, 0] = (x1 + x2) / 2  ## center x
    cals[:, 1] = (y1 + y2) / 2  ## center y
    # bboxs[:,2] = torch.where(x1 == x2 , torch.sign(y2 - y1) * math.pi / 2, torch.atan2(y2 - y1 , x2 - x1))
    cals[:, 2] = torch.where(x1 == x2, torch.sign(y2 - y1) * math.pi / 2, torch.atan((y2 - y1) / (x2 - x1)))   ## rad
    cals[:, 3] = torch.sqrt(torch.pow(x2 - x1, 2) + torch.pow(y2 - y1, 2))  ## length
    return cals