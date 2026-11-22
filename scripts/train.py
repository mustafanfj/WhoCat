"""E3 (and the E7 student): fine-tune a face embedder with ArcFace + batch-hard triplet.

The only difference between the two E3 runs is the batch sampler:
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py --name arc_random --sampler random
  CUDA_VISIBLE_DEVICES=1 python scripts/train.py --name arc_coat   --sampler coat
Edge student (E7):
  python scripts/train.py --name student_coat --sampler coat --backbone mobilenetv3_large_100 --lr-bb 1e-3

Checkpoints: checkpoints/<name>/{best.pt,last.pt,config.json,log.csv}
best.pt is chosen on val cats only (mean TAR@FAR=1% of P1 and P3).
"""
import _boot  # noqa: F401
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from whocat import metrics as M, paths, protocols
from whocat.data import load_eval_images, load_ids, load_images
from whocat.degrade import BgSwap, TrainDoorAug
from whocat.embed import embed_paths, n_workers
from whocat.losses import ArcFace, batch_hard_triplet
from whocat.models import FaceNet
from whocat.samplers import PKSampler


class TrainSet(Dataset):
    def __init__(self, rel_paths, labels, tf):
        self.p, self.y, self.tf = rel_paths, labels, tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        img = Image.open(paths.IMAGES / self.p[i]).convert("RGB")
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.BICUBIC)
        return self.tf(img), self.y[i]


@torch.no_grad()
def validate(net, ev, val_trials, img_size):
    net.eval()
    val_paths = ev[ev.split == "val"]
    e = np.zeros((len(ev), net.neck[0].out_features), np.float32)
    e[val_paths.index.to_numpy()] = embed_paths(net, val_paths.path.tolist(), batch=256, desc="val")
    out = {}
    for k, t in val_trials.items():
        out[k] = M.tar_at_far(protocols.score(t, e, e), t["label"], 0.01)[0]
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sampler", choices=["random", "coat"], default="coat")
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--P", type=int, default=64, help="cats per batch")
    ap.add_argument("--K", type=int, default=2, help="photos per cat in a batch")
    ap.add_argument("--groups", type=int, default=4, help="coat groups per batch (coat sampler)")
    ap.add_argument("--lr-bb", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--arc-s", type=float, default=32.0)
    ap.add_argument("--arc-m", type=float, default=0.4)
    ap.add_argument("--tri-w", type=float, default=1.0)
    ap.add_argument("--tri-m", type=float, default=0.3)
    ap.add_argument("--door-aug", type=float, default=0.0, help="probability of training door augmentation")
    ap.add_argument("--bg-swap", type=float, default=0.0,
                    help="E12b: probability of replacing the background outside the head oval")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing: same maths, ~40%% less GPU memory, ~25%% slower")
    ap.add_argument("--max-cats", type=int, default=0, help="debug: limit training cats")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = paths.CKPT / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(args), dim=512)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    imgs, ids = load_images(), load_ids()
    tr = imgs[imgs.split == "train"].reset_index(drop=True)
    if args.max_cats:
        keep = tr.id.drop_duplicates().iloc[: args.max_cats]
        tr = tr[tr.id.isin(keep)].reset_index(drop=True)
    labels, uniq = pd.factorize(tr.id)
    imgs_by_label = list(pd.Series(np.arange(len(tr))).groupby(labels).apply(np.array))
    coat_groups = None
    if args.sampler == "coat":
        coat = ids[ids.split == "train"].set_index("id").coat.reindex(uniq).fillna("unknown").to_numpy()
        coat_groups = {c: np.flatnonzero(coat == c) for c in np.unique(coat)}
        print("coat groups:", {c: len(v) for c, v in coat_groups.items()})
    print(f"{len(uniq)} training cats, {len(tr)} images, sampler={args.sampler}")

    net = FaceNet(args.backbone, 512, pretrained=True, img_size=args.img_size).to(dev)
    if args.grad_ckpt:
        net.backbone.set_grad_checkpointing(True)
    tf = [T.RandomResizedCrop(args.img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05),
                              interpolation=T.InterpolationMode.BICUBIC),
          T.RandomHorizontalFlip(),
          T.ColorJitter(brightness=0.2, contrast=0.2)]          # no hue/saturation: coat colour is identity
    if args.door_aug > 0:
        tf.insert(0, TrainDoorAug(args.door_aug))
    if args.bg_swap > 0:                                         # before the door aug: swap, then degrade
        tf.insert(0, BgSwap(tr.path.tolist(), paths.IMAGES, args.bg_swap))
    tf = T.Compose(tf + [T.ToTensor(), T.Normalize(*net.norm)])

    sampler = PKSampler(imgs_by_label, args.P, args.K, coat_groups, args.groups, args.seed)
    dl = DataLoader(TrainSet(tr.path.tolist(), labels, tf), batch_sampler=sampler,
                    num_workers=n_workers(), pin_memory=dev == "cuda", persistent_workers=True)
    head = ArcFace(512, len(uniq), args.arc_s, args.arc_m).to(dev)
    opt = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": args.lr_bb, "weight_decay": args.wd},
        {"params": net.neck.parameters(), "lr": args.lr_head, "weight_decay": 5e-4},
        {"params": head.parameters(), "lr": args.lr_head, "weight_decay": 5e-4},
    ])
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps, warm = args.epochs * len(sampler), min(500, len(sampler))

    ev = load_eval_images()
    val_trials = {p: protocols.load(p, "val") for p in ("P1", "P3")}
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else dict(device_type="cpu", enabled=False)
    log, best, step = [], -1, 0
    for ep in range(args.epochs):
        t0, run = time.time(), dict(arc=0.0, tri=0.0, viol=0.0, n=0)
        for x, y in dl:
            frac = step / steps
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * (step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * frac)))
            head.m = args.arc_m * min(1.0, step / max(1, len(sampler)))   # margin warm-up over epoch 1
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(**amp):
                emb = net(x)
            l_arc = head(emb.float(), y)
            l_tri, viol = batch_hard_triplet(emb, y, args.tri_m)
            loss = l_arc + args.tri_w * l_tri
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            step += 1
            run["arc"] += l_arc.item(); run["tri"] += l_tri.item(); run["viol"] += viol.item(); run["n"] += 1
            if step % 100 == 0:
                n = run["n"]
                print(f"ep {ep} step {step}/{steps} arc {run['arc']/n:.3f} tri {run['tri']/n:.3f} "
                      f"hard-neg-violations {run['viol']/n:.2%} {n*x.shape[0]/(time.time()-t0):.0f} img/s", flush=True)
        v = validate(net, ev, val_trials, args.img_size)
        score = float(np.mean(list(v.values())))
        n = max(run["n"], 1)
        log.append(dict(epoch=ep, arc=run["arc"] / n, tri=run["tri"] / n, val_P1=v["P1"], val_P3=v["P3"],
                        minutes=(time.time() - t0) / 60))
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        state = {"model": net.state_dict(), "epoch": ep, "val": v}
        torch.save(state, out / "last.pt")
        if score > best:
            best = score
            torch.save(state, out / "best.pt")
        print(f"== epoch {ep}: val TAR@1% P1 {v['P1']:.4f} P3 {v['P3']:.4f} (best {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
