from typing import Dict
from datetime import datetime
from pathlib import Path
import argparse
import wandb
import torch
import numpy as np
import time
import yaml
import uuid
import gc
import os
import shutil

from ovo.utils import io_utils, gen_utils, eval_utils
from ovo.entities.ovomapping import OVOSemMap
from ovo.entities.ovo import OVO
from PIL import Image
import torchvision.transforms as T

def debug_architecture_flow(scene_path: Path, image_path: str):
    """
    아키텍처의 핵심 계층을 한 단계씩 훑어보기 위한 디버그 함수
    """
    print(f"🚀 [Debug] Starting architecture walkthrough with: {image_path}")
    
    # 1. 모델 로드 과정 확인
    # 여기서 Step Into(F11)를 하면 OVO 클래스의 생성자와 CLIP/Backbone 로드 과정을 볼 수 있습니다.
    ovo, _ = load_representation(scene_path, eval=True, debug_info=True)
    
    # 2. 이미지 입력 및 특징 추출 (Vision Encoder)
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).cuda()
    
    print("🔍 [Step 1] Checking Vision Encoder...")
    # 이 지점에서 breakpoint를 걸고 ovo 내부의 encoder를 확인하세요.
    breakpoint() 
    
    # 3. 텍스트-이미지 정렬 및 분류 (Open-Vocabulary Logic)
    print("🔍 [Step 2] Checking Semantic Classification...")
    # ovo.classify_instances 내부에서 CLIP 텍스트 임베딩과 매칭되는 로직이 실행됩니다.
    classes = ["desk", "chair", "monitor", "world map", "clock"]
    results = ovo.classify_instances(classes)
    
    print(f" Detection Results: {results['classes']}")

def load_representation(scene_path: Path, eval: bool=False, debug_info: bool=False) -> OVO:
    config = io_utils.load_config(scene_path / "config.yaml", inherit=False)
    submap_ckpt = torch.load(scene_path /"ovo_map.ckpt" )
    map_params = submap_ckpt.get("map_params", None)
    if map_params is None:
        map_params = submap_ckpt["gaussian_params"]        
    config["semantic"]["verbose"] = False 
    ovo = OVO(config["semantic"],None, config["data"]["scene_name"], eval=eval, device=config.get("device", "cuda"))
    ovo.restore_dict(submap_ckpt["ovo_map_params"], debug_info=debug_info)
    return ovo, map_params


def compute_scene_labels(scene_path: Path, dataset_name: str, scene_name: str, data_path:str, dataset_info: Dict) -> None:

    ovo, map_params = load_representation(scene_path, eval=True)
    pcd_pred = map_params["xyz"]
    points_obj_ids = map_params["obj_ids"]

    _, pcd_gt = io_utils.load_scene_data(dataset_name, scene_name, data_path, dataset_info, False)
    classes = dataset_info["class_names"] if dataset_info.get("map_to_reduced", None) is None else dataset_info["class_names_reduced"]
    pred_path = scene_path.parent / dataset_info["dataset"]
    os.makedirs(pred_path, exist_ok=True)
    pred_path = pred_path / (scene_name+".txt")

    # It may happen that all the points associated to an object where prunned, such that the number of unique labels in points_obj_ids, is different from the number of semantic module instances
    print("Computing predicted instances labels ...")

    instances_info = ovo.classify_instances(classes)

    mesh_semantic_labels = dict()
    print("Matching instances to ground truth mesh ...")
    mesh_instance_labels, mesh_instances_masks, matched_instances_ids = eval_utils.match_labels_to_vtx(points_obj_ids[:,0], pcd_pred, pcd_gt)
    
    map_id_to_idx = {id: i for i, id in enumerate(ovo.objects.keys())}
    mesh_semantic_labels = instances_info["classes"][np.vectorize(map_id_to_idx.get)(mesh_instance_labels)]
    instances_info["masks"] = mesh_instances_masks.int().numpy()

    print(f"Writing prediction to {pred_path}!")
    io_utils.write_labels(pred_path, mesh_semantic_labels)
    io_utils.write_instances(scene_path.parent, scene_name, instances_info)

    ovo.cpu()
    del ovo


def run_scene(scene: str, dataset: str, experiment_name: str, tmp_run: bool = False, depth_filter: bool = None) -> None:

    config = io_utils.load_config("data/working/configs/ovo.yaml")
    map_module = config["slam"]["slam_module"]
    if map_module.startswith("orbslam"):
        map_module = "vanilla"
        
    config_slam = io_utils.load_config(os.path.join(config["slam"]["config_path"],  map_module, dataset.lower()+".yaml"))
    io_utils.update_recursive(config, config_slam)

    config_dataset = io_utils.load_config(f"data/working/configs/{dataset}/{dataset.lower()}.yaml")
    io_utils.update_recursive(config, config_dataset)
    
    if os.path.exists(f"data/working/configs/{dataset}/{scene}.yaml"):
        config_scene = io_utils.load_config(f"data/working/configs/{dataset}/{scene}.yaml")
        io_utils.update_recursive(config, config_scene)
        
    if "data" not in config:
        config["data"] = {}
    config["data"]["scene_name"] = scene
    config["data"]["input_path"] = f"data/input/Datasets/{dataset}/{scene}"

    output_path = Path(f"data/output/{dataset}/")

    if tmp_run:
        output_path = output_path / "tmp"

    output_path = output_path / experiment_name / scene

    if depth_filter is not None:
        config["semantic"]["depth_filter"] = depth_filter

    if os.getenv('DISABLE_WANDB') == 'true':
        config["use_wandb"] = False
    elif config["use_wandb"]:
        wandb.init(
            project=config["project_name"],
            config=config,
            dir="data/working/output/wandb",
            group=config["data"]["scene_name"]
            if experiment_name != ""
            else experiment_name,
            name=f'{config["data"]["scene_name"]}_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}_{str(uuid.uuid4())[:5]}',
        )

    gen_utils.setup_seed(config["seed"])
    gslam = OVOSemMap(config, output_path=output_path)
    gslam.run()

    if tmp_run:
        final_path = Path(f"data/output/{dataset}/") / experiment_name / scene
        shutil.move(output_path, final_path)

    if config["use_wandb"]:
        wandb.finish()
    print("Finished run.✨")

def main(args):
    if args.experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M")
        tmp_run = True
    else:
        assert len(args.experiment_name) > 0, "Experiment name cannot be '' "
        experiment_name = args.experiment_name
        tmp_run = False

    experiment_path = Path("data/output") / args.dataset_name / experiment_name

    if args.scenes_list is not None:
        with open(args.scenes_list, "r") as f:
            scenes = f.read().splitlines() 
    else:
        scenes = args.scenes

    if len(scenes) == 0 or args.segment or args.eval:
        path = Path("data/working/configs/") / args.dataset_name / args.dataset_info_file
        with open(path, 'r') as f:
            dataset_info = yaml.full_load(f)

        if len(scenes) == 0:
            scenes = dataset_info["scenes"]

    for scene in scenes:        
        input_path = f"./data/input/Datasets/{args.dataset_name}/{scene}"

        ### debugging code
        if scene == "office0": # 혹은 원하는 씬 이름
            # experiment_path는 main 상단에서 정의됨
            scene_path = experiment_path / scene 
            image_path = "images/office.jpg" # 아까 만드신 폴더 구조 기준
            
            # 모델 결과물(.ckpt)이 있는 경로인지 확인 후 실행
            if scene == "office0":
                scene_path = experiment_path / scene 
                image_path = "images/office.jpg"
                
                # 폴더가 아니라 실제 .ckpt 파일이 있는지 확인!
                if (scene_path / "ovo_map.ckpt").exists():
                    debug_architecture_flow(scene_path, image_path)
                    print("🛑 Architecture debugging complete. Exiting...")
                    return
                else:
                    print(f"ℹ️ {scene_path / 'ovo_map.ckpt'}가 아직 없습니다. 모델 생성을 시작합니다.")
        ### end of debugging code

        if args.run:
            t0 = time.time()
            run_scene(scene, args.dataset_name, experiment_name, tmp_run = tmp_run)
            t1 = time.time()
            print(f"Scene {scene} took: {t1-t0:.2f}")
        gc.collect()
 
    if args.segment: 
        data_path ="data/input/Datasets/"
        for scene in scenes:    
            scene_path = experiment_path / scene
            compute_scene_labels(scene_path, args.dataset_name, scene, data_path, dataset_info)

    if args.eval:
        if dataset_info["dataset"] == "scannet200":
            gt_path = Path(input_path).parent / "scannet200_gt"
        else:
            gt_path = Path(input_path).parent / "semantic_gt"
        eval_utils.eval_semantics(experiment_path / dataset_info["dataset"], gt_path, scenes, dataset_info, ignore_background=args.ignore_background)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments to run and evaluate over a dataset')
    parser.add_argument('--dataset_name', help="Dataset used. Choose either `Replica`, `ScanNet`")
    parser.add_argument('--scenes', nargs="+", type=str, default=[], help=" List of scenes from given dataset to run.  If `--scenes_list` is set, this flag will be ignored.")
    parser.add_argument('--scenes_list',type=str, default=None, help="Path to a txt containing a scene name on each line. If set, `--scenes` is ignored. If neither `--scenes` nor `--scenes_list` are set, the scene list will be loaded from `data/working/config/<dataset_name>/<dataset_info_file>`")
    parser.add_argument('--dataset_info_file',type=str, default="eval_info.yaml")
    parser.add_argument('--experiment_name', default=None, type=str)
    parser.add_argument('--run', action='store_true', help="If set, compute the final metrics, after running OVO and segmenting.")
    parser.add_argument('--segment', action='store_true', help="If set, use the reconstructed scene to segment the gt point-cloud, after running OVO.")
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ignore_background', action='store_true',help="If set, does not use background ids from eval_info to compute metrics.")
    args = parser.parse_args()
    main(args)