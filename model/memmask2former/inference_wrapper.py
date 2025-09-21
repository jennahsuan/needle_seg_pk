from typing import List, Optional, Iterable, Dict, Union
import logging
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn.functional as F

from .mem_m2f import MemMask2Former

class MemoryFeatureManager:
    """
    Feature of a frame should be associated with a unique index -- typically the frame id.
    - dict _bank:
        - s0 (s1, s2)
            - cond_frame_outputs (first pred)
                - frame_id: 
                    - maskmem_features
                    - maskmem_pos_enc
            - non_cond_frame_outputs
                - frame_id: 
                    - maskmem_features
                    - maskmem_pos_enc
    - dict _last_cache ###
        - ms_features: multi scale features
        - pix_feat
        - current_vision_feats
        - current_vision_pos_embeds
    """
    def __init__(self, network: MemMask2Former, no_warning: bool = False):
        self.network = network
        self.no_warning = no_warning
        self.init_memory()
        
    def init_memory(self):
        self.engaged = False
        self._bank = {}
        if self.network.track_model_cfg.memory_attention.method == "sam2":
            for s in range(self.network.multi_scale_memory):
                self._bank[f's{str(s)}'] = {"cond_frame_outputs":{}, "non_cond_frame_outputs":{}}

        self._last_cache = {}

        ## init memory for vis_query_att
        ## Stores past queries in decoder. Different from memory bank!
        if self.network.cfg["Model"]["track_modules"].get("vis_query_att") or self.network.cfg["Model"]["track_modules"]["moe_q"] is not None:
            self.dec_query_pre_memory = {"k": [], "v": []}  
        elif self.network.cfg["Model"]["track_modules"].get("query_membank_att"):
            self.dec_query_pre_memory = {}
        else:
            self.dec_query_pre_memory = None

    def _encode_feature(self, index: int, image: torch.Tensor) -> None:
        """
        - index: current t
        Encode image and store.
        Return last feature from backbone.
        """
        ms_features, mask_feature = self.network.unet_backbone(image)  ## mask_feature [B, C=72, H, W]  ## multi_scale_features [B, C', H', W']*3 (1/32, 1/16, 1/8)
        # ms_features, pix_feat = self.network.encode_image(image)
        
        if self.network.track_model_cfg.mask_encoder.method == "sam2":
            self._last_cache["ms_features"] = ms_features
        return mask_feature

    def get_curr_features(self, index: int,
                     image: torch.Tensor): ## -> (Iterable[torch.Tensor], torch.Tensor)
        """
        Return ms_features, pix_feat, mask_feature
        - index: current t
        """
        mask_feature = self._encode_feature(index, image)
        pix_feat = None
        if index in self._bank:
            pix_feat = self._bank[index].get("pix_feat")

        return (self._last_cache.get("ms_features"), 
                pix_feat, 
                mask_feature)
    

    def delete(self, index: int) -> None:
        if index in self._bank:
            del self._bank[index]
        if self._bank.get("non_cond_frame_outputs") and self._bank["non_cond_frame_outputs"].get(index):
            del self._bank["non_cond_frame_outputs"][index]

    def remove_old(self):
        for s in range(self.network.multi_scale_memory):
            while len(self._bank[f"s{s}"]["non_cond_frame_outputs"]) > self.network.track_eval_cfg.long_term.max_mem_frames:
                stored_keys = list(self._bank[f"s{s}"]["non_cond_frame_outputs"].keys())
                min_key = min(stored_keys)
                self._bank[f"s{s}"]["non_cond_frame_outputs"].pop(min_key)

    def __len__(self):
        return len(self._bank["s0"]["non_cond_frame_outputs"])

    def __del__(self):
        if len(self._bank) > 0 and not self.no_warning:
            print(f'[IMAGE FEATURE STORE WARN] Leaking {self._bank.keys()} in the image feature store')

class MemInferenceWrapper(MemMask2Former):

    def __init__(self,
                #  network: MemMask2Former,
                 cfg, track_model_cfg, track_eval_cfg,
                 *,
                 mem_manager: MemoryFeatureManager = None):
        super().__init__(cfg, track_model_cfg)
        # self.network = network
        # self.network.eval()
        self.cfg = cfg
        self.track_eval_cfg = track_eval_cfg
        self.mem_every = track_eval_cfg.mem_every
        self.flip_aug = track_eval_cfg.flip_aug  ## default False

        self.curr_ti = -3
        self.last_mem_ti = 0
        ## scale in memory
        self.multi_scale_memory = track_model_cfg.mem_scale
        
        self.object_manager = ObjectManager()
        if mem_manager is None:
            self.mem_manager = MemoryFeatureManager(self)
        else:
            self.mem_manager = mem_manager

        self.last_mask = None
        self.device = "cuda"

        ## SAM2 param
        if self.track_model_cfg.memory_attention.method == "sam2":
            self.memory_temporal_stride_for_eval = self.track_eval_cfg.mem_every
        
        
        ## resample memory bank
        if self.multi_scale_memory == 3:
            self.past_readouts_list = [[],[],[]]
        elif self.multi_scale_memory == 2:
            self.past_readouts_list = [[],[]]
        else:
            self.past_readouts_list = []

    def clear_memory(self):
        """
        init a new MemoryManager
        """
        # print('[MEM] reset')
        self.curr_ti = -3
        self.last_mem_ti = 0
        self.object_manager = ObjectManager()
        if self.track_model_cfg.memory_attention.method == "sam2":
            self.past_readouts_list = []
            if self.multi_scale_memory > 1: 
                for s in range(self.multi_scale_memory):
                    self.past_readouts_list.append([])
            self.mem_manager.init_memory()


    def memory_engaged(self) -> bool:
        return self.mem_manager.engaged

    def _add_memory(self,
                    image: torch.Tensor,
                    pred_mask: torch.Tensor,
                    *,
                    force_permanent: bool = False,
                    object_score_logits:torch.Tensor = None) -> None:
        """
        Memorize the given segmentation in all memory stores.

        The batch dimension is 1 if flip augmentation is not used. (1/2) means 1 or 2
        - image: RGB image, (1/2)*3*H*W
        - pred_mask: prediction mask logits, (1/2)*num_objects*H*W', in [0, 1]
        - selection can be None if not using long-term memory
        - force_permanent: whether to force the memory to be permanent
        """
        if pred_mask.shape[1] == 0 or torch.all(pred_mask < 0.5):
            # nothing to add
            # print('[WARN] Trying to add empty mask to memory!', end='')
            return

        if self.track_model_cfg.mask_encoder.method == "sam2":
            msk_value, _, _, _, maskmem_pos_enc = self.encode_memory(
                image,        ## [B,3,H,W]
                self.mem_manager._last_cache.get("current_vision_feats"),  ## flattented feature
                pred_mask,         ## [B,1,H,W]
                object_score_logits=object_score_logits) 
            # if isinstance(maskmem_pos_enc,list):
            #     maskmem_pos_enc = maskmem_pos_enc[-1]  ## This is in _encode_new_memory
            del self.mem_manager._last_cache["current_vision_feats"], self.mem_manager._last_cache["current_vision_pos_embeds"]


        if self.track_model_cfg.memory_attention.method == "sam2":
            if self.track_model_cfg.mask_encoder.method != "sam2":
                maskmem_pos_enc = self.memory_pos_enc(msk_value)
            resample_add=True
            if len(self.mem_manager._bank['s0']["cond_frame_outputs"]) == 0:
                store_key = "cond_frame_outputs"
                # k_idx = 0
            else:
                store_key = "non_cond_frame_outputs"
                # k_idx = len(self.mem_manager._bank[store_key])+1

            ## Save memory
            if not self.resample_memory or (self.resample_memory and resample_add):
                for s in range(self.multi_scale_memory):
                    if self.multi_scale_memory == 1:
                        self.mem_manager._bank[f"s0"][store_key][self.curr_ti] = {"maskmem_features": msk_value.squeeze(1),  # B*C*H'*W'
                                                                    "maskmem_pos_enc": maskmem_pos_enc}    
                    else:
                        self.mem_manager._bank[f"s{s}"][store_key][self.curr_ti] = {"maskmem_features": msk_value[s].squeeze(1),  # B,C,H',W'
                                                                "maskmem_pos_enc": maskmem_pos_enc[s]}
                ## remove old features
                self.mem_manager.remove_old()
            else:
                return
        
        if self.track_model_cfg.memory_attention.method == "sam2":
            self.mem_manager.engaged = True

        self.last_mem_ti = self.curr_ti

    def _segment(self,
                 ms_features: Iterable[torch.Tensor],
                 image: torch.Tensor,
                 mask_feature: torch.Tensor,) -> torch.Tensor:
        """
        Produce a segmentation using the given features and the memory

        The batch dimension is 1 if flip augmentation is not used.
        Input:
        - ms_features: an iterable of multiscale features from the encoder, each is (1/2)*_*H*W
                      with strides 16, 8, and 4 respectively
        - mask_feature: from pixel decoder
        

        Returns: (num_objects+1)*H*W normalized probability; the first channel is the background
        """
        bs = ms_features[0].shape[0]
        if self.flip_aug:  ## Test time augmentation
            assert bs == 2
        else:
            assert bs == 1

        if not self.memory_engaged():  ## true if memory.add_memory() has been called
            out, _ = self.transformer_decoder(ms_features, mask_feature)
            self.rescale_pred_mask(out, image)
            # if self.curr_ti <2:
            print(f"t{self.curr_ti} Segment w/o any memory!", end=" ")
            return out

        ## last_mask: mask from the last frame
        # --------------------------------------------------------------
        # Get "Readout": Read from memories to get a memory affected feature
        ## memory_readout [B, 1, embed_dim, H/32, W/32]
        # --------------------------------------------------------------
        if self.track_model_cfg.memory_attention.method == "sam2":
            if self.multi_scale_memory > 1:
                memory_readout, scale_list = [],[32,16,8]
                if self.track_model_cfg.shallow_backbone: #C2:
                    scale_list = [16,8,4]
                for s in range(self.multi_scale_memory):
                    readout_s, bank_memory = self._prepare_memory_conditioned_features(
                                    frame_idx = self.curr_ti,
                                    is_init_cond_frame = False,  ## SAM2 only treat frames with prompt as cond_frame
                                    current_vision_feats = [self.mem_manager._last_cache["current_vision_feats"][3-s]],  
                                    current_vision_pos_embeds = [self.mem_manager._last_cache["current_vision_feats"][3-s]],
                                    feat_sizes = (self.img_size//scale_list[s], self.img_size//scale_list[s]),
                                    output_dict = self.mem_manager._bank[f"s{s}"],
                                    past_readouts=self.past_readouts_list[s])
                    memory_readout.append(readout_s) ## smallest (1/32) to largest(1/8) scale
            else:
                memory_readout, bank_memory = self._prepare_memory_conditioned_features(
                        frame_idx = self.curr_ti,
                        is_init_cond_frame = False,
                        current_vision_feats = self.mem_manager._last_cache["current_vision_feats"][-1:],  ## smallest 3 scale instead of largest
                        current_vision_pos_embeds = self.mem_manager._last_cache["current_vision_pos_embeds"][-1:],
                        feat_sizes = (self.img_size//16, self.img_size//16),
                        output_dict = self.mem_manager._bank["s0"],
                        past_readouts=self.past_readouts_list)  ## TODO memory manager?
                # print('[memory_readout]', memory_readout.shape)
            if self.cfg["Model"]["track_modules"].get("query_membank_att"):
                self.mem_manager.dec_query_pre_memory["bank_memory"] = bank_memory
        
        # --------------------------------------------------------------
        # Decode with multi scale features and "Readout"
        # --------------------------------------------------------------
        ## out: dict("pred_masks","pred_class", "aux_outputs")
        if self.multi_scale_memory > 1:#C1
            if self.track_model_cfg.ms_readout is not None and "norm" in self.track_model_cfg.ms_readout:
                memory_readout = [self.read_norm[i](memory_readout[i]) for i in range(len(memory_readout))]
                ms_features = [self.feat_norm[i](ms_features[i]) for i in range(len(memory_readout))]
            if self.track_model_cfg.ms_readout is not None and "sum" in self.track_model_cfg.ms_readout:
                memory_readout = [mem_read + ms_feat for mem_read, ms_feat in zip(memory_readout, ms_features)]  # [B,256,H',W']  .squeeze(0)
                if self.track_model_cfg.ms_readout is not None and "relu" in self.track_model_cfg.ms_readout:
                    memory_readout = [self.activ(mem_read) for mem_read in memory_readout]
                if len(memory_readout) == 2:  ## mem_scale: 2
                    memory_readout = memory_readout + [ms_features[2]]
                out, _ = self.transformer_decoder(memory_readout, mask_feature) #### readout is a list of augmented ms_feature (small to big)
            else:
                if len(memory_readout) == 2:  ## mem_scale: 2
                    memory_readout = memory_readout + [ms_features[2]]
                out, _ = self.transformer_decoder(memory_readout, mask_feature, pre_memory=self.mem_manager.dec_query_pre_memory) #### readout is a list of augmented ms_feature (small to big) .squeeze(0)
        elif self.multi_scale_memory == 1:#A2:
            if self.track_model_cfg.ms_readout == "avg":
                decoder_input_readout = (memory_readout + ms_features[1]) /2  ## .squeeze(0)
            else:
                decoder_input_readout = memory_readout  ##.squeeze(0)
            if self.track_model_cfg.B1:
                ## MaskUpsampleBlock only needs readout with num_obj dim, ouput is also with num_obj dim
                ms_features[2] = self.up_16_8(decoder_input_readout.unsqueeze(1), ms_features[2]).squeeze(1)  
            if self.track_model_cfg.shallow_backbone: #.C2:
                readout = [decoder_input_readout, ms_features[1], ms_features[2]]
            else:
                readout = [ms_features[0], decoder_input_readout, ms_features[2]]
            out, _ = self.transformer_decoder(readout, mask_feature, pre_memory=self.mem_manager.dec_query_pre_memory)  #### A2
        else:
            raise NotImplementedError
        self.rescale_pred_mask(out, image)

        # vis_query_att accumulate query
        if (self.mem_manager.dec_query_pre_memory != None and 
            (self.cfg["Model"]["track_modules"].get("vis_query_att")
            or (self.cfg["Model"]["track_modules"]["moe_q"] is not None and 'past' in self.cfg["Model"]["track_modules"]["moe_q"]))):
            self.mem_manager.dec_query_pre_memory["k"].append(out["pre_memory"]["k"])
            self.mem_manager.dec_query_pre_memory["v"].append(out["pre_memory"]["v"])
            # if len(self.mem_manager.dec_query_pre_memory["k"])  > 300: ## maintain a queue ## TODO priority queue?
            #     self.mem_manager.dec_query_pre_memory["k"] = self.mem_manager.dec_query_pre_memory["k"][1:]
            # if len(self.mem_manager.dec_query_pre_memory["v"])  > 300:
            #     self.mem_manager.dec_query_pre_memory["v"] = self.mem_manager.dec_query_pre_memory["v"][1:]
        
        return out

    ## origin step()
    def forward(self,
             sample,
             mask: Optional[torch.Tensor] = None,
             objects: Optional[List[int]] = [1], # None
             *,
             idx_mask: bool = True,
             end: bool = False,
             delete_buffer: bool = True,
             force_permanent: bool = False) -> torch.Tensor:
        """
        Take a step with a new incoming image.
        If there is an incoming mask with new objects, we will memorize them.
        If there is no incoming mask, we will segment the image using the memory.
        In both cases, we will update the memory and return a segmentation.

        sample: 'image' b*3*H*W
        mask: None
        objects: list of object ids that are valid in the mask Tensor.
                The ids themselves do not need to be consecutive/in order, but they need to be 
                in the same position in the list as the corresponding mask
                in the tensor in non-idx-mask mode.
                objects is ignored if the mask is None. 
                If idx_mask is False and objects is None, we sequentially infer the object ids.
        idx_mask: if True, mask is expected to contain an object id at every pixel.
                  If False, mask should have multiple channels with each channel representing one object.
        end: if we are at the end of the sequence, we do not need to update memory
            if unsure just set it to False 
        delete_buffer: whether to delete the image feature buffer after this step
        force_permanent: the memory recorded this frame will be added to the permanent memory
        """

        # resize input if needed -- currently only used for the GUI  ## Not used
        if isinstance(sample, dict):
            image = sample["images"].to(self.device)
        elif isinstance(sample, torch.Tensor):
            image = sample.to(self.device)

        # --------------------------------------------------------------------------
        # Check update timing
        # --------------------------------------------------------------------------
        self.curr_ti += 1
        # whether to update the memory
        is_mem_frame = ((self.curr_ti - self.last_mem_ti >= self.mem_every) or
                        (mask is not None)) and (not end)
        # segment when there is no input mask or when the input mask is incomplete
        need_segment = (mask is None) or (self.object_manager.num_obj > 0
                                          and not self.object_manager.has_all(objects))
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Encode the image
        # M2f: backbone encoder + pixel decoder
        # Write key memory, get pix_feat
        # --------------------------------------------------------------------------
        ms_feat, pix_feat, mask_feature = self.mem_manager.get_curr_features(self.curr_ti, image)
        if self.track_model_cfg.memory_attention.method == "sam2":
            (current_vision_feats, 
            current_vision_pos_embeds,
            pix_feat) = self.get_curr_vision_feat_pos(ms_feat, mask_feature)
            ## test for simplicity.. but this should be simpler, no need to save curr feat in dict
            self.mem_manager._last_cache["pix_feat"] = pix_feat 
            self.mem_manager._last_cache["current_vision_feats"] = current_vision_feats 
            self.mem_manager._last_cache["current_vision_pos_embeds"] = current_vision_pos_embeds 
        # --------------------------------------------------------------------------

        # segmentation from memory if needed   ## Should always be true
        if need_segment:  ## TODO check foward
            out_dict = self._segment(ms_feat,
                                    image,
                                    mask_feature)

        # --------------------------------------------------------------
        # Postprocess: no mask pred if cls pred < 0.5
        # --------------------------------------------------------------
        if out_dict["pred_class"].item() < 0.5:
            out_dict["pred_masks"].zero_()
        
        # --------------------------------------------------------------
        # Add object to object manager
        # --------------------------------------------------------------
        """
        If think easy, then object ID is always only 1, no need position or temp ids
        Future work: View more frames at th begining to make better prediction?
        """
        if torch.all(out_dict["pred_masks"] < 0.5):
            # print(f"No pred at {self.curr_ti}\tpred_class", out_dict.get("pred_class").item(), " |", end='\t')
            objects = None
        else:
            objects = [1]
            corresponding_tmp_ids, _ = self.object_manager.add_new_objects(objects)

        # --------------------------------------------------------------
        # Reset the latest pred mask
        # --------------------------------------------------------------
        self.last_mask = out_dict["pred_masks"]  ## [B,1,H,W]
        # self.last_mask = pred_prob_with_bg[1:].unsqueeze(0)

        # --------------------------------------------------------------
        # Encode memory if memory bank is empty or it's time to encode
        # --------------------------------------------------------------
        if is_mem_frame or force_permanent or \
            (objects is not None and not self.memory_engaged()):  ## if memory is empty, add first prediction  #  and self.curr_ti > -1
            self._add_memory(image,
                             self.last_mask,
                             force_permanent=force_permanent,
                             object_score_logits = out_dict.get("pred_class", None))

        if delete_buffer:
            self.mem_manager.delete(self.curr_ti)

        return out_dict




# from ..Cutie.cutie.inference.object_manager import ObjectManager

from typing import Union, List, Dict

# from .object_info import ObjectInfo

class ObjectInfo:
    """
    Store meta information for an object
    """
    def __init__(self, id: int):
        self.id = id
        self.poke_count = 0  # count number of detections missed

    def poke(self) -> None:
        self.poke_count += 1

    def unpoke(self) -> None:
        self.poke_count = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if type(other) == int:
            return self.id == other
        return self.id == other.id

    def __repr__(self):
        return f'(ID: {self.id})'

class ObjectManager:
    """
    Object IDs are immutable. The same ID always represent the same object.
    Temporary IDs are the positions of each object in the tensor. It changes as objects get removed.
    Temporary IDs start from 1.
    """

    def __init__(self):
        self.obj_to_tmp_id: Dict[ObjectInfo, int] = {}
        self.tmp_id_to_obj: Dict[int, ObjectInfo] = {}
        self.obj_id_to_obj: Dict[int, ObjectInfo] = {}

        self.all_historical_object_ids: List[int] = []

    def _recompute_obj_id_to_obj_mapping(self) -> None:
        self.obj_id_to_obj = {obj.id: obj for obj in self.obj_to_tmp_id}

    def add_new_objects(
            self, objects: Union[List[ObjectInfo], ObjectInfo,
                                 List[int]]) -> (List[int], List[int]):
        if not isinstance(objects, list):
            objects = [objects]

        corresponding_tmp_ids = []
        corresponding_obj_ids = []
        for obj in objects:
            if isinstance(obj, int):
                obj = ObjectInfo(id=obj)

            if obj in self.obj_to_tmp_id:
                # old object
                corresponding_tmp_ids.append(self.obj_to_tmp_id[obj])
                corresponding_obj_ids.append(obj.id)
            else:
                # new object
                new_obj = ObjectInfo(id=obj.id)

                # new object
                new_tmp_id = len(self.obj_to_tmp_id) + 1
                self.obj_to_tmp_id[new_obj] = new_tmp_id
                self.tmp_id_to_obj[new_tmp_id] = new_obj
                self.all_historical_object_ids.append(new_obj.id)
                corresponding_tmp_ids.append(new_tmp_id)
                corresponding_obj_ids.append(new_obj.id)
        # print("self.obj_to_tmp_id", self.obj_to_tmp_id)
        self._recompute_obj_id_to_obj_mapping()
        assert corresponding_tmp_ids == sorted(corresponding_tmp_ids)
        return corresponding_tmp_ids, corresponding_obj_ids

    def delete_objects(self, obj_ids_to_remove: Union[int, List[int]]) -> None:
        # delete an object or a list of objects
        # re-sort the tmp ids
        if isinstance(obj_ids_to_remove, int):
            obj_ids_to_remove = [obj_ids_to_remove]

        new_tmp_id = 1
        total_num_id = len(self.obj_to_tmp_id)

        local_obj_to_tmp_id = {}
        local_tmp_to_obj_id = {}

        for tmp_iter in range(1, total_num_id + 1):
            obj = self.tmp_id_to_obj[tmp_iter]
            if obj.id not in obj_ids_to_remove:
                local_obj_to_tmp_id[obj] = new_tmp_id
                local_tmp_to_obj_id[new_tmp_id] = obj
                new_tmp_id += 1

        self.obj_to_tmp_id = local_obj_to_tmp_id
        self.tmp_id_to_obj = local_tmp_to_obj_id
        self._recompute_obj_id_to_obj_mapping()

    def purge_inactive_objects(self,
                               max_missed_detection_count: int) -> (bool, List[int], List[int]):
        # remove tmp ids of objects that are removed
        obj_id_to_be_deleted = []
        tmp_id_to_be_deleted = []
        tmp_id_to_keep = []
        obj_id_to_keep = []

        for obj in self.obj_to_tmp_id:
            if obj.poke_count > max_missed_detection_count:
                obj_id_to_be_deleted.append(obj.id)
                tmp_id_to_be_deleted.append(self.obj_to_tmp_id[obj])
            else:
                tmp_id_to_keep.append(self.obj_to_tmp_id[obj])
                obj_id_to_keep.append(obj.id)

        purge_activated = len(obj_id_to_be_deleted) > 0
        if purge_activated:
            self.delete_objects(obj_id_to_be_deleted)
        return purge_activated, tmp_id_to_keep, obj_id_to_keep

    def tmp_to_obj_cls(self, mask) -> torch.Tensor:
        # remap tmp id cls representation to the true object id representation
        new_mask = torch.zeros_like(mask)
        for tmp_id, obj in self.tmp_id_to_obj.items():
            new_mask[mask == tmp_id] = obj.id
        return new_mask

    def get_tmp_to_obj_mapping(self) -> Dict[int, ObjectInfo]:
        # returns the mapping in a dict format for saving it with pickle
        return {obj.id: tmp_id for obj, tmp_id in self.tmp_id_to_obj.items()}

    def realize_dict(self, obj_dict, dim=1) -> torch.Tensor:
        # turns a dict indexed by obj id into a tensor, ordered by tmp IDs
        output = []
        for _, obj in self.tmp_id_to_obj.items():
            if obj.id not in obj_dict:
                raise NotImplementedError
            output.append(obj_dict[obj.id])
        output = torch.stack(output, dim=dim)
        return output

    def make_one_hot(self, cls_mask) -> torch.Tensor:
        output = []
        for _, obj in self.tmp_id_to_obj.items():
            output.append(cls_mask == obj.id)
        if len(output) == 0:
            output = torch.zeros((0, *cls_mask.shape), dtype=torch.bool, device=cls_mask.device)
        else:
            output = torch.stack(output, dim=0)
        return output

    @property
    def all_obj_ids(self) -> List[int]:
        return [k.id for k in self.obj_to_tmp_id]

    @property
    def num_obj(self) -> int:
        return len(self.obj_to_tmp_id)

    def has_all(self, objects: List[int]) -> bool:
        for obj in objects:
            if obj not in self.obj_to_tmp_id:
                return False
        return True

    def find_object_by_id(self, obj_id) -> ObjectInfo:
        return self.obj_id_to_obj[obj_id]

    def find_tmp_by_id(self, obj_id) -> int:
        return self.obj_to_tmp_id[self.obj_id_to_obj[obj_id]]
