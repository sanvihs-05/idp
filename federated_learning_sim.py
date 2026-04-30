"""
Real YOLOv8 Federated Learning simulation using Flower (FedAvg).

What this script does:
1) Merges frontside + backside datasets into one unified YOLO dataset.
2) Splits merged training set into 3 balanced non-overlapping sites (A/B/C).
3) Runs Flower server + 3 Flower clients on localhost.
4) Each client trains YOLOv8 locally for N epochs per round.
5) Server aggregates model parameters via FedAvg and evaluates global mAP.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import yaml
from flwr.common import Context, NDArrays, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO


CLASS_NAMES = ["defect", "good_pack", "blister", "capsule", "no_pill", "no_pack"]
FRONTSIDE_MAP = {0: 2, 1: 3, 2: 0, 3: 4}  # blister,capsule,defect,no_pill
BACKSIDE_MAP = {0: 0, 1: 1, 2: 5}  # defect,good_pack,no_pack
SITE_IDS = ["A", "B", "C"]


@dataclass
class SiteData:
    site_id: str
    image_files: List[Path]
    label_files: List[Path]

    @property
    def num_examples(self) -> int:
        return len(self.image_files)


def _read_label_classes(label_path: Path) -> List[int]:
    if not label_path.exists():
        return []
    classes = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            classes.append(int(float(parts[0])))
        except Exception:
            continue
    return classes


def _remap_label_file(src: Path, dst: Path, class_map: Dict[int, int]) -> None:
    if not src.exists():
        dst.write_text("", encoding="utf-8")
        return
    remapped = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_old = int(float(parts[0]))
        if cls_old not in class_map:
            continue
        parts[0] = str(class_map[cls_old])
        remapped.append(" ".join(parts))
    dst.write_text("\n".join(remapped), encoding="utf-8")


def prepare_combined_dataset(
    front_root: Path,
    back_root: Path,
    output_root: Path,
) -> Dict[str, Path]:
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ["train", "valid"]:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    source_specs = [
        ("front", front_root, FRONTSIDE_MAP),
        ("back", back_root, BACKSIDE_MAP),
    ]
    split_map = {"train": "train", "valid": "valid"}

    for tag, root, cls_map in source_specs:
        for src_split, out_split in split_map.items():
            src_img_dir = root / src_split / "images"
            src_lbl_dir = root / src_split / "labels"
            for src_img in src_img_dir.glob("*.jpg"):
                new_name = f"{tag}_{src_img.name}"
                dst_img = output_root / out_split / "images" / new_name
                dst_lbl = output_root / out_split / "labels" / f"{Path(new_name).stem}.txt"
                shutil.copy2(src_img, dst_img)
                _remap_label_file(src_lbl_dir / f"{src_img.stem}.txt", dst_lbl, cls_map)

    global_yaml = output_root / "global_data.yaml"
    global_cfg = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    global_yaml.write_text(yaml.safe_dump(global_cfg, sort_keys=False), encoding="utf-8")
    return {"global_yaml": global_yaml, "dataset_root": output_root}


def build_site_splits(combined_root: Path) -> Dict[str, SiteData]:
    train_img_dir = combined_root / "train" / "images"
    train_lbl_dir = combined_root / "train" / "labels"
    all_images = sorted(train_img_dir.glob("*.jpg"))
    buckets: Dict[int, List[Path]] = {i: [] for i in range(len(CLASS_NAMES))}
    unlabeled: List[Path] = []

    for img in all_images:
        lbl = train_lbl_dir / f"{img.stem}.txt"
        classes = _read_label_classes(lbl)
        if not classes:
            unlabeled.append(img)
            continue
        buckets[classes[0]].append(img)

    site_imgs: Dict[str, List[Path]] = {sid: [] for sid in SITE_IDS}
    for cls_id, cls_imgs in buckets.items():
        random.Random(42 + cls_id).shuffle(cls_imgs)
        for idx, img in enumerate(cls_imgs):
            site_imgs[SITE_IDS[idx % 3]].append(img)
    for idx, img in enumerate(unlabeled):
        site_imgs[SITE_IDS[idx % 3]].append(img)

    site_data: Dict[str, SiteData] = {}
    for sid in SITE_IDS:
        imgs = sorted(site_imgs[sid])
        lbls = [train_lbl_dir / f"{img.stem}.txt" for img in imgs]
        site_data[sid] = SiteData(site_id=sid, image_files=imgs, label_files=lbls)
    return site_data


def write_site_datasets(site_data: Dict[str, SiteData], combined_root: Path, output_root: Path) -> Dict[str, Path]:
    yaml_paths: Dict[str, Path] = {}
    valid_img_dir = combined_root / "valid" / "images"
    valid_lbl_dir = combined_root / "valid" / "labels"

    for sid, data in site_data.items():
        site_root = output_root / f"site_{sid}"
        for split in ["train", "valid"]:
            (site_root / split / "images").mkdir(parents=True, exist_ok=True)
            (site_root / split / "labels").mkdir(parents=True, exist_ok=True)

        for img, lbl in zip(data.image_files, data.label_files):
            shutil.copy2(img, site_root / "train" / "images" / img.name)
            if lbl.exists():
                shutil.copy2(lbl, site_root / "train" / "labels" / lbl.name)
            else:
                (site_root / "train" / "labels" / lbl.name).write_text("", encoding="utf-8")

        for vimg in valid_img_dir.glob("*.jpg"):
            shutil.copy2(vimg, site_root / "valid" / "images" / vimg.name)
            vlbl = valid_lbl_dir / f"{vimg.stem}.txt"
            if vlbl.exists():
                shutil.copy2(vlbl, site_root / "valid" / "labels" / vlbl.name)
            else:
                (site_root / "valid" / "labels" / f"{vimg.stem}.txt").write_text("", encoding="utf-8")

        data_yaml = site_root / "data.yaml"
        cfg = {
            "path": str(site_root.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "nc": len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }
        data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        yaml_paths[sid] = data_yaml
    return yaml_paths


def get_model_parameters(model: YOLO) -> NDArrays:
    state_dict = model.model.state_dict()
    return [tensor.detach().cpu().numpy() for tensor in state_dict.values()]


def set_model_parameters(model: YOLO, parameters: NDArrays) -> None:
    state_dict = model.model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(parameters):
        raise ValueError(f"Parameter count mismatch (model={len(keys)} vs fed={len(parameters)}).")

    new_state = {}
    mismatches = []
    for key, arr in zip(keys, parameters):
        target = state_dict[key]
        src = torch.as_tensor(arr, dtype=target.dtype, device=target.device)
        if tuple(src.shape) != tuple(target.shape):
            mismatches.append(f"{key}: expected {tuple(target.shape)}, got {tuple(src.shape)}")
            continue
        new_state[key] = src
    if mismatches:
        preview = "; ".join(mismatches[:5])
        extra = f"; ... {len(mismatches) - 5} more" if len(mismatches) > 5 else ""
        raise ValueError(f"Federated tensor shape mismatch: {preview}{extra}")
    model.model.load_state_dict(new_state, strict=True)


def build_project_yolo(pretrained_weights: Path, class_names: List[str]) -> YOLO:
    """Build a YOLOv8 model with the project class head, then transfer compatible pretrained weights."""
    source = YOLO(str(pretrained_weights))
    cfg = copy.deepcopy(source.model.yaml)
    cfg["nc"] = len(class_names)

    model = YOLO(str(pretrained_weights))
    model.model = DetectionModel(cfg=cfg, nc=len(class_names), verbose=False)
    model.task = "detect"
    model.overrides["task"] = "detect"
    model.model.nc = len(class_names)
    model.model.names = {idx: name for idx, name in enumerate(class_names)}
    model.model.args = {**getattr(model.model, "args", {}), **model.overrides}
    model.load(str(pretrained_weights))
    return model


def ensure_project_seed_weights(pretrained_weights: Path, output_dir: Path, class_names: List[str]) -> Path:
    """Create a reusable checkpoint whose detection head already matches the federated dataset."""
    seed_path = output_dir / f"{pretrained_weights.stem}_{len(class_names)}cls_seed.pt"
    model = build_project_yolo(pretrained_weights, class_names)
    model.save(str(seed_path))
    return seed_path


def train_local_client(
    data_yaml: Path,
    base_weights: Path,
    rounds_dir: Path,
    site_id: str,
    round_idx: int,
    local_epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> Tuple[NDArrays, float]:
    model = YOLO(str(base_weights))
    train_name = f"site_{site_id}_round_{round_idx}"
    train_result = model.train(
        data=str(data_yaml),
        epochs=local_epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(rounds_dir.resolve()),
        name=train_name,
        verbose=False,
        plots=False,
        save=True,
        val=True,
        workers=0,
    )
    save_dir = Path(getattr(train_result, "save_dir", rounds_dir / train_name))
    weights_dir = save_dir / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"
    chosen_weights = best_weights if best_weights.exists() else last_weights
    if not chosen_weights.exists():
        raise FileNotFoundError(f"No trained weights found in {weights_dir}")
    trained_model = YOLO(str(chosen_weights))
    params = get_model_parameters(trained_model)
    val_metrics = trained_model.val(data=str(data_yaml), imgsz=imgsz, batch=batch, device=device, verbose=False, workers=0)
    local_loss = float(max(0.0, 1.0 - float(val_metrics.box.map50)))
    return params, local_loss


def evaluate_global_model(
    base_weights: Path,
    global_parameters: NDArrays,
    global_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
) -> Dict[str, float]:
    model = YOLO(str(base_weights))
    set_model_parameters(model, global_parameters)
    metrics = model.val(data=str(global_yaml), imgsz=imgsz, batch=batch, device=device, verbose=False, workers=0)
    return {
        "global_map50": round(float(metrics.box.map50), 4),
        "global_map50_95": round(float(metrics.box.map), 4),
        "global_precision": round(float(metrics.box.mp), 4),
        "global_recall": round(float(metrics.box.mr), 4),
    }


class PharmaYoloClient(fl.client.NumPyClient):
    def __init__(
        self,
        site_id: str,
        site_yaml: Path,
        num_examples: int,
        base_weights: Path,
        rounds_dir: Path,
        local_epochs: int,
        imgsz: int,
        batch: int,
        device: str,
    ) -> None:
        self.site_id = site_id
        self.site_yaml = site_yaml
        self.num_examples = num_examples
        self.base_weights = base_weights
        self.rounds_dir = rounds_dir
        self.local_epochs = local_epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.seed_model = YOLO(str(base_weights))

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_model_parameters(self.seed_model)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        round_idx = int(config.get("server_round", 1))
        try:
            local_start = self.rounds_dir / f"site_{self.site_id}_start_round_{round_idx}.pt"
            model = YOLO(str(self.base_weights))
            set_model_parameters(model, parameters)
            model.save(str(local_start))
            updated_params, local_loss = train_local_client(
                data_yaml=self.site_yaml,
                base_weights=local_start,
                rounds_dir=self.rounds_dir,
                site_id=self.site_id,
                round_idx=round_idx,
                local_epochs=self.local_epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                device=self.device,
            )
        except Exception as exc:
            # Never crash the federated round because one site hit a local training error.
            print(f"[WARN] Site {self.site_id} failed at round {round_idx}: {exc}")
            updated_params = parameters
            local_loss = 1.0
        return updated_params, self.num_examples, {"site_id": self.site_id, "local_loss": local_loss}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        return 0.0, self.num_examples, {}


class TrackingFedAvg(FedAvg):
    def __init__(self, eval_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_fn = eval_fn
        self.round_history: List[Dict[str, object]] = []

    def aggregate_fit(self, server_round, results: List[Tuple[ClientProxy, fl.common.FitRes]], failures):
        agg_result = super().aggregate_fit(server_round, results, failures)
        if agg_result is None:
            # Keep history consistent even when a round fails.
            self.round_history.append(
                {"round": server_round, "global_map50": 0.0, "global_map50_95": 0.0, "global_precision": 0.0, "global_recall": 0.0, "client_losses": {}}
            )
            return None
        aggregated_parameters, _ = agg_result
        if aggregated_parameters is None:
            self.round_history.append(
                {"round": server_round, "global_map50": 0.0, "global_map50_95": 0.0, "global_precision": 0.0, "global_recall": 0.0, "client_losses": {}}
            )
            return agg_result
        ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        eval_metrics = self.eval_fn(ndarrays)
        client_losses: Dict[str, float] = {}
        for _, fit_res in results:
            sid = str(fit_res.metrics.get("site_id", "unknown"))
            client_losses[sid] = float(fit_res.metrics.get("local_loss", 0.0))
        self.round_history.append({"round": server_round, **eval_metrics, "client_losses": client_losses})
        return aggregated_parameters, {}


def save_split_manifest(site_data: Dict[str, SiteData], output_dir: Path) -> None:
    manifest = {}
    for sid, data in site_data.items():
        manifest[f"Site {sid}"] = {"total_images": data.num_examples}
    with (output_dir / "fl_site_split.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run_fl_simulation(
    front_root: Path,
    back_root: Path,
    rounds: int = 5,
    local_epochs: int = 2,
    output_dir: Path = Path("fl_results"),
    imgsz: int = 640,
    batch: int = 4,
    device: str = "cpu",
    initial_weights: Path = Path("yolov8n.pt"),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_root = output_dir / "combined_dataset"
    prep = prepare_combined_dataset(front_root, back_root, combined_root)
    global_yaml = prep["global_yaml"]
    seed_weights = ensure_project_seed_weights(initial_weights, output_dir, CLASS_NAMES)
    site_data = build_site_splits(combined_root)
    save_split_manifest(site_data, output_dir)
    site_yaml_map = write_site_datasets(site_data, combined_root, output_dir / "sites")
    rounds_dir = output_dir / "round_artifacts"
    if rounds_dir.exists():
        shutil.rmtree(rounds_dir)
    rounds_dir.mkdir(parents=True, exist_ok=True)

    def eval_fn(params: NDArrays) -> Dict[str, float]:
        return evaluate_global_model(
            base_weights=seed_weights,
            global_parameters=params,
            global_yaml=global_yaml,
            imgsz=imgsz,
            batch=batch,
            device=device,
        )

    strategy = TrackingFedAvg(
        eval_fn=eval_fn,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=3,
        min_available_clients=3,
        min_evaluate_clients=0,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )

    site_idx_to_id = {str(i): sid for i, sid in enumerate(SITE_IDS)}

    def client_fn(context: Context):
        node_id = str(context.node_config.get("partition-id", context.node_id))
        sid = site_idx_to_id[node_id]
        client = PharmaYoloClient(
            site_id=sid,
            site_yaml=site_yaml_map[sid],
            num_examples=site_data[sid].num_examples,
            base_weights=seed_weights,
            rounds_dir=rounds_dir,
            local_epochs=local_epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
        )
        return client.to_client()

    client_resources = {"num_cpus": 2, "num_gpus": 1.0} if device != "cpu" else {"num_cpus": 1, "num_gpus": 0.0}
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    history_path = output_dir / "fl_metrics.json"
    payload = {
        "rounds": rounds,
        "local_epochs": local_epochs,
        "server_address": "in-process-simulation",
        "initial_weights": str(initial_weights),
        "architecture_seed_weights": str(seed_weights),
        "class_names": CLASS_NAMES,
        "history": strategy.round_history,
        "privacy_note": (
            "Privacy-preserving FL: each site keeps raw images local; "
            "only model weights/updates are shared for FedAvg aggregation."
        ),
    }
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return history_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real YOLOv8 federated learning simulation")
    parser.add_argument("--front-root", type=Path, default=Path("BLISTER-1"), help="Frontside dataset root")
    parser.add_argument(
        "--back-root",
        type=Path,
        default=Path("Larger-Blister-Pack-Defect--1"),
        help="Backside dataset root",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Federated rounds")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--output-dir", type=Path, default=Path("fl_results"), help="FL output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO training image size")
    parser.add_argument("--batch", type=int, default=4, help="YOLO batch size")
    parser.add_argument("--device", type=str, default=("0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--initial-weights", type=Path, default=Path("yolov8n.pt"), help="Base model weights")
    args = parser.parse_args()

    out_path = run_fl_simulation(
        front_root=args.front_root,
        back_root=args.back_root,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        initial_weights=args.initial_weights,
    )
    print(f"Federated simulation completed. Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
