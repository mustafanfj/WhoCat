"""Face embedders behind one interface: model(x) -> (B, D) embedding; .transform for PIL input.

Baselines (frozen, used as-is):
  petface   PetFace's ArcFace cat model (ResNet-50, ECCV 2024)            224px
  megadesc  MegaDescriptor-L-384 (WildlifeDatasets, Swin-L)               384px
  dinov2    plain DINOv2-Base, CLS token (general-purpose features)       224px
  miewid    MiewID-msv3 (optional; needs trust_remote_code)              440px
Helper (not a model under test):
  clip      CLIP ViT-L/14; used to mine P3 look-alikes and label coats    224px
Ours:
  ckpt:<path>  a checkpoint written by scripts/train.py
"""
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from . import paths

IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIPN = ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def eval_tf(size, norm):
    return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                      T.ToTensor(), T.Normalize(*norm)])


class PetFaceR50(nn.Module):
    def __init__(self, weights=paths.PETFACE_WEIGHTS):
        super().__init__()
        from torchvision.models import resnet50
        m = resnet50(weights=None)
        m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 512), nn.BatchNorm1d(512))
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        m.load_state_dict(sd.get("state_dict_backbone", sd))
        self.m = m
        self.transform = eval_tf(224, IMNET)

    def forward(self, x):
        return self.m(x)


class TimmHub(nn.Module):
    def __init__(self, name, size, norm):
        super().__init__()
        import timm
        self.m = timm.create_model(name, pretrained=True, num_classes=0)
        self.transform = eval_tf(size, norm)

    def forward(self, x):
        return self.m(x)

    def _fake(self, name):
        import timm
        nn.Module.__init__(self)
        self.m = timm.create_model("resnet18", pretrained=False, num_classes=0)
        self.transform = eval_tf(224, IMNET)
        return self


class HFDino(nn.Module):
    def __init__(self, name="facebook/dinov2-base"):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(name)
        self.transform = eval_tf(224, IMNET)

    def forward(self, x):
        return self.m(pixel_values=x).pooler_output


def _feat(o):
    """transformers<5 returns a tensor, >=5 a ModelOutput whose pooler_output is the projected feature."""
    return o if torch.is_tensor(o) else o.pooler_output


class HFClip(nn.Module):
    def __init__(self, name="openai/clip-vit-large-patch14"):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer
        self.m = CLIPModel.from_pretrained(name)
        self.tok = CLIPTokenizer.from_pretrained(name)
        self.transform = eval_tf(224, CLIPN)

    def forward(self, x):
        return _feat(self.m.get_image_features(pixel_values=x))

    @torch.no_grad()
    def text(self, prompts, device):
        t = self.tok(prompts, padding=True, return_tensors="pt").to(device)
        return F.normalize(_feat(self.m.get_text_features(**t)), dim=-1)


class MiewID(nn.Module):
    def __init__(self, name="conservationxlabs/miewid-msv3"):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(name, trust_remote_code=True)
        self.transform = eval_tf(440, IMNET)

    def forward(self, x):
        return self.m(x)


# ---------------------------------------------------------------- our trained models
class FaceNet(nn.Module):
    """timm backbone + linear/BN neck -> 512-d face embedding (trained by scripts/train.py)."""

    def __init__(self, backbone="vit_base_patch14_dinov2.lvd142m", dim=512, pretrained=True, img_size=224):
        super().__init__()
        import timm
        kw = {"img_size": img_size} if "vit" in backbone or "dinov2" in backbone else {}
        self.backbone = timm.create_model(backbone, pretrained=pretrained and not FAKE, num_classes=0, **kw)
        cfg = self.backbone.pretrained_cfg
        self.norm = (tuple(cfg.get("mean", IMNET[0])), tuple(cfg.get("std", IMNET[1])))
        with torch.no_grad():   # true output width (MobileNetV3 has a 1280-d head conv after num_features)
            was = self.backbone.training
            out_dim = self.backbone.eval()(torch.zeros(1, 3, img_size, img_size)).shape[-1]
            self.backbone.train(was)
        self.neck = nn.Sequential(nn.Linear(out_dim, dim), nn.BatchNorm1d(dim))
        self.img_size = img_size
        self.transform = eval_tf(img_size, self.norm)

    def forward(self, x):
        return self.neck(self.backbone(x))


def load_checkpoint(path):
    path = Path(path)
    cfg = json.loads((path.parent / "config.json").read_text())
    net = FaceNet(cfg["backbone"], cfg["dim"], pretrained=False, img_size=cfg["img_size"])
    sd = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"])
    return net


FAKE = os.environ.get("WHOCAT_FAKE_HUB") == "1"   # offline smoke tests: random weights, no downloads


class _FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.m = timm.create_model("resnet18", pretrained=False, num_classes=64)
        self.transform = eval_tf(224, CLIPN)

    def forward(self, x):
        return self.m(x)

    @torch.no_grad()
    def text(self, prompts, device):
        g = torch.Generator().manual_seed(len(prompts[0]))
        return F.normalize(torch.randn(len(prompts), 64, generator=g), dim=-1).to(device)


def build(name):
    if FAKE and name in ("megadesc", "dinov2", "miewid"):
        return TimmHub.__new__(TimmHub)._fake(name)
    if FAKE and name == "clip":
        return _FakeClip()
    if name == "petface":
        return PetFaceR50()
    if name == "megadesc":
        return TimmHub("hf-hub:BVRA/MegaDescriptor-L-384", 384, HALF)
    if name == "dinov2":
        return HFDino()
    if name == "clip":
        return HFClip()
    if name == "miewid":
        return MiewID()
    if name.startswith("ckpt:"):
        return load_checkpoint(name[5:])
    raise ValueError(name)


def cache_name(name):
    """File-safe key for a model name (ckpt:/a/b/run/best.pt -> run_best)."""
    if name.startswith("ckpt:"):
        p = Path(name[5:])
        return f"{p.parent.name}_{p.stem}"
    return name
