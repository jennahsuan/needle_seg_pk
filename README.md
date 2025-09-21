# Ultrasound Video Needle Segmentation + PK

## Code Directory Layout
```
needle_seg_pk/
    ├── pretrained_weight/
    ├── configs/
    ├── lib/
    ├── model/
    │   └── mask2former/
    │   └── memmask2former/ 
    │   └── DeAOT/ (support functions from https://github.com/yoxu515/aot-benchmark) 
    │   └── SAM2/ 
    ├── .gitignore
    ├── dataset.py (dataset and augmentation)
    ├── environment.yml
    ├── environment_linux.yml (this is for linux only)
    ├── inference_PK.ipynb
    ├── MEMO5090.md  (installation & developement guideline for RTX5090 laptop)
    ├── README.md
```

## Set up
For RTX5090, ignore the below instructpions and check out `MEMO5090.md`
1. Build conda env
	```bash
	conda env create -f environment.yml
	conda activate needle_seg
	```
2. Install other packages
	```bash
	pip install psutil ninja omegaconf
	mim install mmcv-full==1.6.0
	pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html  ## install the version based on your own device (https://mmcv.readthedocs.io/en/v1.6.0/get_started/installation.html)
	pip install mmengine
	```
<!-- 
3. Extension in TransNeXt
Follow https://github.com/DaiShiResearch/TransNeXt?tab=readme-ov-file#cuda-implementation,
download and move swattention_extension folder to `./model/mask2former`
	```bash
	cd ./model/mask2former/swattention_extension
	pip install .
	```
NOTE: This should be done under `conda activate needle_seg` 
If "error: Microsoft Visual C++ 14.0 or greater is required", download Microsoft C++ Build Tools. -->

3. **Checkpoints**: stored in `Dropbox\家庭資料室_Developer\Prodigy管理\inference_PK\ckpt`, run `bash pretrained_weight/download.sh` to place them in the `pretrain_weight` folder.

## Inference and PK Guidance
**NOTE**: important code is highlight with `=======`
1. Set `configs/config_PK.yml`
	* If input is raw prodigy video, set `raw_video_dir` with the video directory, and `sonosite_frame_dir` and `prodigy_frame_dir` to `null`
	* Tune the hyperparamters `left_bright_weight`, `left_shadow_weight` ...
	* `Detection_model.name` options: `m2f` for mask2former, `mem_m2f` for mask2former model with memory modules (time-cost is not tested yet)
2. Run `inference_PK.ipynb`
	* **Tip**: fold(close) the code when you see `# region ## not used now` if the code is too redundant for you (most of them are code for testing or from source code by Wang Phd.)
	* In **`Set Coordinates`**, if the video contains 3 frames in the same position as in `家庭資料室_Developer/Prodigy管理/zipper array data for PK豬肉打針/ultrasound_2025-06-13-15-07.mp4`, then just run the 2nd cell (use default values). Otherwise, run the first cell to check out the coordinates and set the coordinate values manually in the 2nd cell.
	* In `PK` function, paramters such as `left_bright_weight` are used to compute on the bright and shadow pixel average of each carriage.
	* In `inference_LCR_image_with_flag` function, the model outputs endpoints `x1`, `y1` & `x3`, `y3`. The depth `x1y1_regression_depth` & `x3y3_regression_depth` (z1 & z3) is estimated by `PK()`.

## Other notes:
- Before cropping the frames online in the code, a pre-cropped video experiment can also be tested. You can crop a single video into 3 videos by `Microsoft Clipchamp`, capture the frames of them and save them in three folders `L`, `C` and `R`. Then, set the `Data.prodigy_frame_dir` to the root of these three folders.
- Original code crop frame and json: [colab](https://colab.research.google.com/drive/1IDEzVgIBcPq9fVombAORn-TQ8SG4xLlT?usp=sharing#scrollTo=55a55FjP1onU)
