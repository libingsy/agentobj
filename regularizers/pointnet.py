"""Standard PointNet classification encoder (Qi et al., 2017).

The architecture matches the widely used PyTorch implementation
`yanx27/Pointnet_Pointnet2_pytorch` (STN3d + STNkd input/feature transforms,
per-point 1x1 convs 3->64->128->1024, global max-pool -> 1024-d feature), so a
classification checkpoint from that repo loads into `PointNetEncoder` after the
`feat.` prefix is stripped. The 1024-d global feature is the naturalness
embedding used by `regularizers/naturalness.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        b = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(b, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(3, device=x.device, dtype=x.dtype).view(1, 9).repeat(b, 1)
        return (x + iden).view(-1, 3, 3)


class STNkd(nn.Module):
    def __init__(self, k: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        b = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(b, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k).repeat(b, 1)
        return (x + iden).view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    """PointNet backbone returning the 1024-d global feature.

    `feature_dim` is 1024 (the global feature). Matches the `feat` module of the
    yanx27 classification model, so its checkpoint loads here after stripping
    the `feat.` prefix.
    """

    feature_dim = 1024

    def __init__(self, channel: int = 3, feature_transform: bool = True):
        super().__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.feature_transform = feature_transform
        if feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        # x: (B, channel, N)
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        return x


def load_encoder_checkpoint(encoder: PointNetEncoder, ckpt_path: str) -> None:
    """Load a classification checkpoint into `encoder` (strict).

    Accepts a raw state dict or a dict with `model_state_dict` / `state_dict`.
    The classification model wraps the backbone as `feat.*`, so those keys are
    used (prefix stripped). Raises if the checkpoint does not match.
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    feat = {k[len("feat."):]: v for k, v in obj.items() if k.startswith("feat.")}
    state = feat or obj
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    # Only the backbone is needed; the classifier head keys are expected extras.
    hard_missing = [k for k in missing]
    if hard_missing:
        raise RuntimeError(
            f"PointNet checkpoint {ckpt_path} is missing encoder weights "
            f"{hard_missing[:6]}... Does it match the yanx27 PointNet architecture?"
        )
