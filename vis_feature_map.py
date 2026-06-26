import argparse
import os
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

import utils.misc as utils
from datasets import build_dataset
from models_mmca_vector_based import build_model


def get_args_parser():
    parser = argparse.ArgumentParser("Generate 4-route + fused heatmaps")

    parser.add_argument("--dataset", default="Aquaov255", type=str)
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--data_root", default="./ln_data/", type=str)
    parser.add_argument("--split_root", default="data", type=str)

    parser.add_argument("--eval_model", required=True, type=str)
    parser.add_argument("--out_dir", default="./outputs/route_heatmaps", type=str)
    parser.add_argument("--heatmap_gamma", default=1.8, type=float, help="Contrast enhancement. Larger value suppresses weak responses.")
    parser.add_argument("--heatmap_threshold", default=0.15, type=float, help="Suppress weak responses below this threshold.")
    parser.add_argument("--overlay_alpha", default=0.0, type=float, help="If >0, blend heatmap with image. Default 0 saves pure heatmap.")
    parser.add_argument("--image_darken", default=1.0, type=float, help="Only used when overlay_alpha > 0.")
    parser.add_argument("--invert_heatmap", action="store_true", help="Invert activation values if the heatmap appears reversed.")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)

    parser.add_argument("--backbone", default="resnet50", type=str)
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument("--position_embedding", default="sine", type=str)
    parser.add_argument("--imsize", default=320, type=int)

    parser.add_argument("--max_query_len", default=128, type=int)
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
    parser.add_argument("--detr_model", default="./saved_models/detr-r50.pth", type=str)

    parser.add_argument("--enc_layers", default=6, type=int)
    parser.add_argument("--dec_layers", default=0, type=int)
    parser.add_argument("--dim_feedforward", default=2048, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--num_queries", default=100, type=int)
    parser.add_argument("--pre_norm", action="store_true")

    parser.add_argument("--emb_size", default=512, type=int)
    parser.add_argument("--bert_enc_num", default=12, type=int)
    parser.add_argument("--detr_enc_num", default=6, type=int)

    parser.add_argument("--vl_dropout", default=0.1, type=float)
    parser.add_argument("--vl_nheads", default=8, type=int)
    parser.add_argument("--vl_hidden_dim", default=256, type=int)
    parser.add_argument("--vl_dim_feedforward", default=2048, type=int)
    parser.add_argument("--vl_enc_layers", default=6, type=int)

    parser.add_argument("--light", default=False, action="store_true")
    parser.add_argument("--aug_blur", action="store_true")
    parser.add_argument("--aug_crop", action="store_true")
    parser.add_argument("--aug_scale", action="store_true")
    parser.add_argument("--aug_translate", action="store_true")
    parser.add_argument("--lr_visu_cnn", default=0.0, type=float)
    parser.add_argument("--lr_bert", default=0.0, type=float)
    parser.add_argument("--masks", action="store_true")
    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument("--eos_coef", default=0.1, type=float)
    parser.add_argument("--aux_loss", action="store_true")
    parser.add_argument("--lr_visu_tra",default=0.0, type=float)



    return parser





def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return img


def _normalize_cam(cam, gamma=2.2, low_percentile=1, high_percentile=99, threshold=0.10, invert=False):
    """
    Normalize activation map for pure CAM visualization.

    Desired style:
        low response  -> dark blue
        high response -> yellow/red

    This function avoids transparent overlay and prevents the whole image from
    becoming red by suppressing weak responses and stretching only strong areas.
    """
    cam = cam.astype(np.float32)

    # Robust normalization.
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)

    if invert:
        cam = 1.0 - cam

    # Percentile stretch.
    lo = np.percentile(cam, low_percentile)
    hi = np.percentile(cam, high_percentile)
    cam = (cam - lo) / (hi - lo + 1e-6)
    cam = np.clip(cam, 0.0, 1.0)

    # Keep background dark blue: remove weak responses.
    cam[cam < threshold] = 0.0

    # Make target peaks more obvious.
    cam = np.power(cam, gamma)
    cam = cam / (cam.max() + 1e-6)

    return cam

def make_heatmap(visu_feat, text_feat, img_h, img_w, gamma=2.2, threshold=0.10, invert=False):
    """
    Build a paper-style heatmap from visual-token/text-feature similarity.

    Output convention:
        background / low response: deep blue
        target / high response: yellow-red
    """
    visu_feat = torch.nn.functional.normalize(visu_feat, dim=-1)
    text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

    score = torch.matmul(visu_feat, text_feat)  # [N]
    score = score.detach().float().cpu().numpy()

    side = int(np.sqrt(score.shape[0]))
    if side * side != score.shape[0]:
        raise ValueError(f"Visual token number {score.shape[0]} is not square.")

    cam = score.reshape(side, side)
    cam = _normalize_cam(cam, gamma=gamma, threshold=threshold, invert=invert)

    # Resize to image size and smooth slightly.
    cam = cv2.resize(cam, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=2.0, sigmaY=2.0)

    # Re-normalize after interpolation/blur and keep weak responses dark.
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    cam[cam < threshold] = 0.0
    cam = np.power(cam, gamma)
    cam = cam / (cam.max() + 1e-6)

    heat_uint8 = np.uint8(cam * 255)

    # OpenCV JET: low=blue, medium=green/yellow, high=red.
    heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    # Force zero-response pixels to dark navy blue, making the background clean.
    bg_mask = heat_uint8 <= 2
    heatmap[bg_mask] = (90, 0, 0)  # BGR: dark blue

    return heatmap

def overlay_heatmap(img_rgb, heatmap, alpha=0.0, darken=1.0):
    """
    Save pure heatmap by default.

    alpha <= 0: return heatmap only, fully covering the image.
    alpha > 0 : blend heatmap with the original image.
    """
    if alpha is None or alpha <= 0:
        return heatmap

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if darken is not None and darken < 0.999:
        img_bgr = np.uint8(img_bgr.astype(np.float32) * darken)

    out = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap, alpha, 0)
    return out

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.eval_model, map_location="cpu")
    state_dict = checkpoint["model"]

    # Robust checkpoint loading:
    # Some visualization runs may use a different text-branch length or number of
    # text routes, which changes vl_pos_embed.weight. PyTorch still raises an
    # error for size mismatches even when strict=False, so mismatched parameters
    # must be removed manually before loading.
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_state and tuple(v.shape) != tuple(model_state[k].shape):
            skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered_state[k] = v

    if skipped:
        print("Skip mismatched checkpoint parameters:")
        for k, old_shape, new_shape in skipped:
            print(f"  {k}: checkpoint {old_shape} -> model {new_shape}")

    msg = model.load_state_dict(filtered_state, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)
    model.eval()

    print("Loading dataset...")
    dataset = build_dataset(args.split, args)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
    )

    print("Generating heatmaps for all images...")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader)):
            img_data, text_data, target = batch

            img_data = img_data.to(device)
            text_data = text_data.to(device)

            _ = model(img_data, text_data)

            if not hasattr(model, "debug_branch_text_feat"):
                raise RuntimeError(
                    "debug_branch_text_feat not found. "
                    "Please add debug saving code into trans_vg.py."
                )

            visu_feat = model.debug_visu_feat[:, 0, :]          # [N, C]
            branch_text = model.debug_branch_text_feat[0]       # [R, C]
            fused_text = model.debug_fused_text_feat[0]         # [C]

            img_rgb = denormalize(img_data.tensors[0])
            h, w = img_rgb.shape[:2]

            sample_dir = out_dir / f"{idx:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # save original image
            cv2.imwrite(
                str(sample_dir / "00_original.jpg"),
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            )

            # four branch heatmaps
            num_branches = branch_text.shape[0]
            for r in range(num_branches):
                heat = make_heatmap(visu_feat, branch_text[r], h, w, gamma=args.heatmap_gamma, threshold=args.heatmap_threshold, invert=args.invert_heatmap)
                overlay = overlay_heatmap(img_rgb, heat, alpha=args.overlay_alpha, darken=args.image_darken)

                cv2.imwrite(str(sample_dir / f"branch_{r}.jpg"), overlay)

            # fused heatmap
            heat_fused = make_heatmap(visu_feat, fused_text, h, w, gamma=args.heatmap_gamma, threshold=args.heatmap_threshold, invert=args.invert_heatmap)
            overlay_fused = overlay_heatmap(img_rgb, heat_fused, alpha=args.overlay_alpha, darken=args.image_darken)
            cv2.imwrite(str(sample_dir / "fused.jpg"), overlay_fused)

            # make one combined panel
            imgs = []
            imgs.append(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            for r in range(num_branches):
                imgs.append(cv2.imread(str(sample_dir / f"branch_{r}.jpg")))
            imgs.append(cv2.imread(str(sample_dir / "fused.jpg")))

            imgs = [cv2.resize(x, (w, h)) for x in imgs]
            panel = np.concatenate(imgs, axis=1)

            cv2.imwrite(str(sample_dir / "panel.jpg"), panel)

    print(f"Done. Heatmaps saved to: {out_dir}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)