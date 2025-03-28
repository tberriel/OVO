from typing import Any, Dict
from pathlib import Path
import numpy as np
import psutil
import pprint
import torch
import wandb

class Logger:
    def __init__(self, output_path: str, pid: int | None = None, use_wandb: bool = False) -> None:
        self.output_path = Path(output_path)
        (self.output_path / "logger").mkdir(exist_ok=True, parents=True)
        (self.output_path / "logger" / "segment_vis").mkdir(exist_ok=True, parents=True)
        stat_keys = [
            "frame_id", "t_sam", "t_obj","n_obj", "n_matches", "t_up", "t_seg",   "t_clip", "avg_fps", "ram", "vram"]
        
        self.stats ={key: [] for key in stat_keys}
        self.python_process = psutil.Process(pid)
        self.use_wandb = use_wandb

    def log_ovo_stats(self, stats: Dict[str, Any], print_output=False) -> None:
        """
        Log CLIP extraction, fusion, and association info.

        Args:
            stats (Dict[str, Any]):
            print_output (bool = False): if True prints logged statistics
        """
        for key, item in stats.items():
            self.stats[key].append(item)
        if self.use_wandb:
            wandb.log({f'Semantic/{key}': value for key, value in stats.items()})
            if "n_obj" in stats.keys():
                for i in range(len(stats["n_obj"])):
                    wandb.log(
                        {
                            "Semantic/Frame": stats["frame_id"],
                            f"Semantic/n_obj_{i}":stats["n_obj"][i],
                        })
            
        if print_output:
            pprint.pprint(stats, width = 160, compact=True)

    def log_fps(self, avg_fps: float):
        self.stats["avg_fps"].append(avg_fps)
        if self.use_wandb:
            wandb.log(
                {
                    "Semantic/avg_fps": avg_fps
                }
            )
            
    def log_memory_usage(self, frame_id: int):
        """
        Logs the memory usage, VRAM and RAM in gigabytes, for a given frame and process ID.
        If `self.use_wandb` is enabled, logs the memory usage statistics to Weights & Biases (wandb).
        Args:
            frame_id (int): Frame associtaed to statistics
        """
        torch.cuda.synchronize()
        vram_used = torch.cuda.memory_allocated("cuda") / (1000 ** 3)
        ram_used = self.python_process.memory_info().rss/(1000 ** 3)
        self.stats["vram"].append(vram_used)
        self.stats["ram"].append(ram_used)
        if self.use_wandb:
            wandb.log(
                {
                    "Semantic/Frame": frame_id,
                    "Semantic/vram": vram_used,
                    "Semantic/ram": ram_used,
                }
            )

    def log_max_memory_usage(self) -> None:
        """
        Logs the max memory usage, VRAM and RAM in gigabytes, from stored memory statistics.
        """
        torch.cuda.synchronize()
        self.stats["max_vram"] = [torch.cuda.max_memory_allocated("cuda") / (1000 ** 3)]
        self.stats["max_ram"] = [np.asarray(self.stats["ram"]).max()]

    def write_stats(self) -> None:
        """
        Writes statistics to log files. writes each statistic to a separate log file. The log files are named after the keys in the 
        dictionary, except for the key "n_obj", which is skipped. The log files are created with the ".log" extension.
        """

        for key, stat in self.stats.items():
            if key == "n_obj":
                continue
            stat_list = [str(i) for i in stat]
            with open(self.output_path/"logger"/f"{key}.log", "w") as f:
                f.write('\n'.join(stat_list))

    def print_final_stats(self) -> None:
        """
        Print logged statistics
        """
        stats = {f"Avg {key}": np.asarray(stat).mean().round(3) for key, stat in self.stats.items() if key not in ["frame_id", "max_vram", "max_ram"] }
        if "max_ram" in self.stats:
            stats["Max RAM"] = round(self.stats["max_ram"][0],2)
            stats["Max vRAM"] = round(self.stats["max_vram"][0],2)
        print("Final statistics:")
        pprint.pprint(stats, compact=True)