# UMVG

Official implementation of **UMVG: A Multi-branch Cross-modal Framework for Underwater Visual Grounding**.

This repository provides the inference code and pretrained model weights for underwater visual grounding on the AquaOV255-VG and NautData-VG benchmarks.

## Model Weights

The pretrained checkpoints are provided via Git LFS:

```text
aquaov255_best_checkpoint.pth
nautdata_best_checkpoint.pth
```

Please install Git LFS before cloning this repository:

```bash
git lfs install
git clone https://github.com/zhangjinkai43-collab/UMVG.git
cd UMVG
git lfs pull
```

## Environment

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, please install the main dependencies manually, including PyTorch, torchvision, transformers, numpy, opencv-python, and PIL.

## Inference

Run inference with the provided checkpoint:

```bash
python eval.py \
  --checkpoint aquaov255_best_checkpoint.pth \
  --image path/to/image.jpg \
  --text "a grey fish rests among rocks, covered with algae, on the seafloor"
```

For the NautData-VG checkpoint:

```bash
python eval.py \
  --checkpoint nautdata_best_checkpoint.pth \
  --image path/to/image.jpg \
  --text "a target object described by the input expression"
```

The output includes the predicted grounding result for the input image and referring expression.

## Repository Structure

```text
UMVG/
├── models/                         # Model definitions
├── models_mmca_vector_based/         # MMCA-based modules
├── aquaov255_best_checkpoint.pth     # Checkpoint for AquaOV255-VG
├── nautdata_best_checkpoint.pth      # Checkpoint for NautData-VG
├── eval.py                           # Inference / evaluation script
├── visualize.py                      # Visualization script
└── README.md
```

## Citation

If you use this code or model weights, please cite our paper:

```bibtex
@article{zhang2026umvg,
  title={UMVG: A Multi-branch Cross-modal Framework for Underwater Visual Grounding},
  author={Zhang, Jinkai and Zhao, Qi and Wang, Chunlei and Liu, Binghao and Zhang, Yutang and Chen, Lijiang},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2026}
}
```

## License

This repository is released for academic research only.
