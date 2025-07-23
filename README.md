# Ultrasound Video Needle Segmentation and Detection

<!-- ## Code Directory Layout
```
dlmi_final/
    ├── README.md
    ├── environment.yml
    ├── config.json
    ├── dataset.py
    ├── model.py
    ├── upsampling_blocks.py
    ├── swin_unetr.py
    ├── transnext_unetr.py
    ├── loss.py
    ├── evaluation.py
    ├── visualization.py
    ├── train.py
    ├── test.py
    ├── pseudo_label.py
    ├── post_processing.py
    ├── anchors.py
    ├── pseudo_label/
    │   └── ...
    ├── video_unetr_checkpoints/
    │   └── ...
    ├── video_retina-unetr_checkpoints/
    │   └── ...
    ├── pretrained_weight/
    │   └── mae_pretrain_vit_base.pth
    │   └── swin_small_patch4_window7_224.pth
        └── transnext_tiny_224_1k.pth
``` -->

## Set up
For RTX5090, ignore the below instructpions and check out `MEMO5090.md`
1. build conda env
  ```bash
  conda env create -f environment.yml
  conda activate needle_seg
  ```

2. other packages
  ```bash
  pip install psutil ninja omegaconf
  mim install mmcv-full==1.6.0
  pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html  ## install the version based on your own device (https://mmcv.readthedocs.io/en/v1.6.0/get_started/installation.html)
  pip install mmengine
  ```

3. Extension in TransNeXt
Follow https://github.com/DaiShiResearch/TransNeXt?tab=readme-ov-file#cuda-implementation,
download and move swattention_extension folder to `./model/mask2former`
  ```bash
  cd ./model/mask2former/swattention_extension
  pip install .
  ```
NOTE: This should be done under `conda activate needle_seg` 
If "error: Microsoft Visual C++ 14.0 or greater is required", download Microsoft C++ Build Tools.

## Segmentation Implementation
### Trainig
  ```bash
  python train.py
  ```
### Testing
  ```bash
  python test.py "<run id in wandb>" "<checkpoint path>"
  # e.g. python test_v2.py "cjz1f4ne" "C:/Dropbox/家庭資料室/demo/transnext_mask2former_cls_T512_pix_add1_ema.pth"
  ```
### Inference
  Run inference.ipynb (ckpt in Dropbox/家庭資料室/demo)

### Inference and PK left right frame
  Run inference_PK.ipynb with **TransNeXt-Mask2former**, **ConvNeXt-Mask2former** or **ConvNeXt-Memory-Mask2former** (ckpts in Dropbox/家庭資料室/demo/inference_PK or https://drive.google.com/drive/folders/1CUVhjczbpiTFq_Ar6mqX1GpmTr5B3NHI?usp=drive_link).
  Follow instructions in ipynb to set configurations for the segementation or track model.

## File Functionalities
- `config.json`
  - Training hyperparameters and model architecture designs can be modified in this file.
- `dataset.py`
  - Contains the custom dataset and augmentation classes.
  - Image reading can be speeded up with larger number of samples in a buffer. 
  - The masks are redundant in the current implementation, as only the mask from the last frame are used in the training process.
  - To enable prediction on the first & second frames, set `add_t0` to True in CustomDataset
- `model`
  - `mask2former`
  <!-- - `upsampling_blocks.py`
    - Contains the upsampling blocks for the U-Net architecture.
  - `swin_unter.py`
    - Contains the Swin-UNETR model.
  - `transnext_unetr.py`
    - Contains the TransNeXt-UNETR model. -->
- `loss.py`
  - Loss functions for segmentation and detection tasks.
- `evaluation_v2.py`
  - Evaluation functions based on the segmentation and detection results.
- `visualization.py`
  - Visualization function for showing the training data before training & prediction results during training.
  - The visualization function can be disabled by setting `visualize = False` in `config.json`.
- `train_v2.py`
  - The main training script. Run `python train.py` to train the model.
  - Hyperparameters can be modified in `config.json`.
  - Pseudo label training is only activated if the mask threshold is set to non `null` value in `config.json`.
- `test_v2.py`
  - Pass `run id` in wandb and checkpoint path when running, so that the model follows the setting during training.
  - Inference medium and hard test folders in `config.json`
  - Record results into `run.summary` in wandb.
- `pseudo_label.py`
  - Evaluate the predicted masks of unlabeled dataset.
  - If the confidence of the mask is high enough, then `mask_XXXX_pl.png` is saved and recorded to `pl.csv` in the `pseudo_label` folder.
- `post_processing.py`
  - Functions for detection head output post-processing.
- `anchors.py`
  - Functions for anchor generation in detection head.
<!-- - `video_unetr_checkpoints/`
  - Directory to save the trained model checkpoints.
- `video_retina-unetr_checkpoints/`
  - Directory to save the trained model checkpoints. -->
- `pretrained_weight/`
  - Directory to pre-trained model weights, including:

  | Backbone | Pretrained Model | Download |
  | :-----: | :----: | :----: |
  | UNETR | mae_pretrain_vit_base | [link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) |
  | Swin Transformer | swin_small_patch4_window7_224 | [link](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth) |
  | TransNeXt | transnext_tiny_224_1k | [link](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true) |

## Progress
### Stage 1
- [x] Retina-UNETR structure for segmentation, detection, and classification head
- [x] Pseudo labeling for training large amount of unlabeled data
- [Slides Report](https://docs.google.com/presentation/d/1LSqBBR_9WsUj8z3wzu-srwehbbd8ft77PaHadLxRxG8/edit?usp=sharing)
### Stage 2 (Current)
- [ ] Short- and long-term information extraction
  
