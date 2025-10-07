# Official repository of Open-Vocabulary Online Semantic Mapping for SLAM

<a href="https://tberriel.github.io/">Tomas Berriel-Martins</a>,
<a href="https://oswaldm.github.io/">Martin R. Oswald</a>,
<a href="https://scholar.google.com/citations?user=j_sMzokAAAAJ&hl=en">Javier Civera</a>,

<div align="left">
    <a href='https://arxiv.org/abs/2411.15043'><img src='https://img.shields.io/badge/arXiv-2404.06836-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <a href='https://tberriel.github.io/ovo/'><img src='https://img.shields.io/badge/Web-Page-green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</div>


## Installation
Following instruction are for an Ubuntu>=20.04 system, with installed Conda and CUDA support.

Clone repository with recursive flag:
```
    git clone git@github.com:tberriel/OVO.git --recursive
```

And set up the conda environment:
```
# conda env
conda create -n ovo python=3.10.13
conda activate ovo

conda install pyyaml tqdm psutil wandb plyfile numpy=1.26.4 matplotlib seaborn opencv=4.11 imageio scipy scikit-learn pandas -c conda-forge
# torch
pip install torch==2.5.1 torchvision==0.20.1 transformers==4.51.0 open_clip_torch==2.32.0 open3d==0.19.0 huggingface-hub==0.30.1

#sam2
cd /<ovo_path>/thirdParty/segment-anything-2
pip install -e .
```
By default we use Segment Anything 2, but OVO is also compatible with SAM 1. If you want, you can install it following the <a href="https://github.com/facebookresearch/segment-anything">official instructions</a>.


### Gaussian-SLAM (Optional)
If you want to run OVO using Gaussian-SLAM backbone, you will have to additionally run:
```
cd /<ovo_path>/
conda activate ovo
conda install faiss-gpu=1.8.0 cudnn -c pytorch -c conda-forge
conda install cuda-toolkit=12.1 -c nvidia/label/cuda-12.1.0
pip install git+https://github.com/VladimirYugay/simple-knn.git@c7e51a06a4cd84c25e769fee29ab391fe5d5ff8d git+https://github.com/VladimirYugay/gaussian_rasterizer.git@9c40173fcc8d9b16778a1a8040295bc2f9
```
### ORB-SLAM2 (Optional)
Clone the repository:
```
cd thirdParty/
git clone https://github.com/tberriel/ORB_SLAM2
cd ORB_SLAM2
git checkout ovo-mapping
```
Manually install ORB-SLAM2 dependencies into the conda environment:
```
conda activate ovo
# Instal conda C compilers to avoid relying on system defaults
conda install cxx-compiler -c conda-forge # generic version hardware agnostic
# Install OpenGL
conda install libegl libegl-devel libgl libgl-devel libgles libgles-devel libglvnd libglvnd-devel libglx libglx-devel libopengl libopengl-devel -c conda-forge
# Install Eigen, Pangolin, OpenCV, Numpy
conda install glew eigen=3.4 pangolin-opengl=0.9.2 libopencv=4.11 numpy=1.26.4 boost -c conda-forge 
```
And finally run the script `build.sh` to build the *ORB-SLAM2* and the python bindings.

## Data
See <a href="./data/input/ReadMe.md">data instructions</a>.

## Run OVO
To run OVO, and compute evaluation metrics use `run_eval.py`. Running flags are:
* `--dataset_name` (required): Dataset used. Choose either `Replica`, `ScanNet`.
* `--experiment_name`: name of the folder used to store the experiment in `data/output/<dataset_name>/<experiment_name>`.
* `--run` If set, run OVO on specified scenes
* `--segment` If set, use the reconstructed scene to segment the gt point-cloud, after running OVO.
* `--eval` If set, compute the final metrics, after runing OVO and segmenting.
* `--dataset_info_file`: file that stores dataset info. By default is `eval_info.yaml`, to evaluate scannet200 set as `eval_info_200.yaml`.
* `--scenes`: List of scenes from given dataset to run. If `--scenes_list` is set, this flag will be ignored.
* `--scenes_list`: Path to a txt containing a scene name on each line. If set, `--scenes` is ignored. If neither `--scenes` nor `--scenes_list` are set, the scene list will be loaded from `data/working/config/<dataset_name>/<dataset_info_file>`

OVO configuration can be modified in `data/working/configs/ovo.yaml`. To speedup SAM models, you can set compile mode modifying corresponding config files in `/<conda_path>/envs/ovo/lib/python3.10/site-packages/sam2/configs/sam2.1/`. 
### Examples
* To run, segment GT, and compute metrics, on Replica Office0:
    ```
    python run_eval.py --dataset_name Replica --experiment_name ovo_mapping --run --segment --eval --scenes office0
    ```

* To run on the full ScanNet scene0011_00:

    ```
    python run_eval.py --dataset_name ScanNet --experiment_name ovo_mapping --run --segment --eval --scenes scene0011_00
    ```

## Changelog:
- 7, October, 2025  - Switched from ORB-SLAM2 to ORB-SLAM3 to minimize segmentation fault errors. 
- 10, June, 2025 - Improved integration with ORB-SLAM2 and added loop closure support.

## Citation
If you found our work useful, please cite us.
```
    @article{martins2024ovo,
    title={Open-Vocabulary Online Semantic Mapping for SLAM},
    author={Martins, Tomas Berriel and Oswald, Martin R. and Civera, Javier},
    journal={IEEE Robotics and Automation Letters}, 
    year={2025},
    }
```