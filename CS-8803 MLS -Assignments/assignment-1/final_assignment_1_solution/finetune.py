#!/usr/bin/env python3
# fine_tune_resnet.py
# Q4: Adversarial training / fine-tuning script for ResNet-50

import argparse
import glob
import os
from typing import Optional, Tuple, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
from torchvision import datasets, transforms, models
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224):
    # NOTE: match your assignment's preprocessing
    return transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_clean_dataset(clean_root: Optional[str], batch_size: int, num_workers: int, shuffle: bool) -> Optional[DataLoader]:
    if clean_root is None or not os.path.isdir(clean_root):
        return None
    tfm = build_transforms(224)
    ds = datasets.ImageFolder(root=clean_root, transform=tfm)
    if len(ds) == 0:
        return None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def load_adv_tensors_from_results(results_dir: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    results_dir should contain one or more .pt files with a dict that includes:
        - 'adv_images': Tensor [N, 3, 224, 224] (already normalized!)
        - 'labels': Tensor [N]
    We will concatenate all found files.
    """
    files = sorted(glob.glob(os.path.join(results_dir, "*.pt")))
    if not files:
        return None, None

    adv_list, lab_list = [], []
    for f in files:
        try:
            data = torch.load(f, map_location="cpu")
            if isinstance(data, dict) and ("adv_images" in data) and ("labels" in data):
                adv = data["adv_images"]
                lab = data["labels"]
                if isinstance(adv, torch.Tensor) and isinstance(lab, torch.Tensor) and adv.ndim == 4:
                    # Expect normalized images already (from the attack pipeline)
                    adv_list.append(adv)
                    lab_list.append(lab)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")

    if not adv_list:
        return None, None

    adv_all = torch.cat(adv_list, dim=0)
    lab_all = torch.cat(lab_list, dim=0)
    return adv_all, lab_all


def make_adv_dataset(adv_images: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool, num_workers: int):
    ds = TensorDataset(adv_images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


class MixedBatcher:
    """
    Simple iterator that interleaves/concatenates batches from clean and adv loaders for each epoch.
    If one runs out, it continues with the other.
    """
    def __init__(self, clean_loader: Optional[DataLoader], adv_loader: Optional[DataLoader]):
        self.clean_loader = clean_loader
        self.adv_loader = adv_loader

    def __iter__(self):
        clean_iter = iter(self.clean_loader) if self.clean_loader is not None else None
        adv_iter = iter(self.adv_loader) if self.adv_loader is not None else None
        while True:
            got_any = False
            if clean_iter is not None:
                try:
                    imgs, labels = next(clean_iter)
                    got_any = True
                    yield imgs, labels
                except StopIteration:
                    clean_iter = None
            if adv_iter is not None:
                try:
                    imgs, labels = next(adv_iter)
                    got_any = True
                    yield imgs, labels
                except StopIteration:
                    adv_iter = None
            if not got_any:
                break


def build_model(lr: float, weight_decay: float, freeze_backbone: bool = False):
    # Pretrained ResNet-50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        # Fine-tune only the final layer if desired:
        for p in model.fc.parameters():
            p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str = "Eval") -> float:
    model.eval()
    correct, total = 0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        if isinstance(batch, dict):  # safety for other loaders
            images, labels = batch["image"], batch["label"]
        else:
            images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)


def load_optional_adv_eval(eval_adv_pt: Optional[str], batch_size: int, num_workers: int) -> Optional[DataLoader]:
    if eval_adv_pt is None or not os.path.isfile(eval_adv_pt):
        return None
    data = torch.load(eval_adv_pt, map_location="cpu")
    if not (isinstance(data, dict) and "adv_images" in data and "labels" in data):
        print(f"[WARN] {eval_adv_pt} not in expected format. Skipping eval loader.")
        return None
    adv = data["adv_images"]
    lab = data["labels"]
    ds = TensorDataset(adv, lab)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def main():
    ap = argparse.ArgumentParser(description="Q4: Adversarial fine-tuning for ResNet-50")
    # Data
    ap.add_argument("--clean_root", type=str, default=None,
                    help="Path to clean images (ImageFolder). Optional but recommended.")
    ap.add_argument("--adv_results_dir", type=str, required=True,
                    help="Directory containing .pt files with {'adv_images','labels'} (from Q3).")
    ap.add_argument("--eval_adv_pt", type=str, default=None,
                    help="Optional .pt file with {'adv_images','labels'} for evaluation (e.g., adv_images_test_100.pt).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)

    # Train
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="If set, freezes all but the final FC layer.")

    # Save
    ap.add_argument("--output_path", type=str, default="fine_tuned_resnet",
                    help="Path to save the fine-tuned state_dict wrapped as {'model_state_dict': ...}")

    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---- Load adversarial tensors (normalized already) ----
    adv_images, adv_labels = load_adv_tensors_from_results(args.adv_results_dir)
    if adv_images is None:
        raise RuntimeError(f"No adversarial tensors found in {args.adv_results_dir}")

    adv_loader = make_adv_dataset(adv_images, adv_labels, args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"[INFO] Adversarial train samples: {len(adv_images)}")

    # ---- Load clean dataset (optional) ----
    clean_loader = load_clean_dataset(args.clean_root, args.batch_size, args.num_workers, shuffle=True)
    if clean_loader is None:
        print("[WARN] No clean dataset provided/found. Training will use only adversarial examples.")

    # ---- Build model, loss, optimizer ----
    model, criterion, optimizer = build_model(args.lr, args.weight_decay, args.freeze_backbone)
    model.to(device)

    # ---- Optional adv eval loader ----
    eval_adv_loader = load_optional_adv_eval(args.eval_adv_pt, args.batch_size, args.num_workers)
    if eval_adv_loader is None:
        print("[INFO] No adv eval set provided. Skipping adversarial eval.")
    else:
        print("[INFO] Adversarial eval set loaded.")

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_count = 0.0, 0

        mixed_iter = MixedBatcher(clean_loader, adv_loader)
        for images, labels in tqdm(mixed_iter, desc=f"Epoch {epoch}/{args.epochs} - train", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            epoch_count += labels.size(0)

        avg_loss = epoch_loss / max(1, epoch_count)
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}")

        # Optional: quick checks each epoch
        if clean_loader is not None:
            clean_acc = evaluate(model, clean_loader, device, desc=f"Epoch {epoch} clean")
            print(f"[Epoch {epoch}] clean_acc={clean_acc:.4f}")
        if eval_adv_loader is not None:
            adv_acc = evaluate(model, eval_adv_loader, device, desc=f"Epoch {epoch} adv_eval")
            print(f"[Epoch {epoch}] adv_eval_acc={adv_acc:.4f}")

    # ---- Save model exactly as required ----
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, args.output_path)
    print(f"[OK] Saved fine-tuned model to: {args.output_path}")


if __name__ == "__main__":
    main()
