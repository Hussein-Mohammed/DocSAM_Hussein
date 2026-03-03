# DocSAM on Windows 11 (single GPU)

This guide provides a stable setup for running inference/training on Windows with Conda + PyCharm.

## 1) Create environment

```powershell
conda create -n DocSAM python=3.11 -y
conda activate DocSAM
```

## 2) Install Visual C++ runtime (required by PyTorch DLLs)

Install **Microsoft Visual C++ Redistributable 2015-2022 (x64)**, then reopen terminal.

If you get this error:

```
OSError: [WinError 127] ... fbgemm.dll ...
```

it is usually caused by missing VC++ runtime or an incompatible torch build.

## 3) Install PyTorch first (matching your CUDA)

For CUDA 12.4:

```powershell
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

## 4) Install remaining dependencies

```powershell
pip install -r requirements.windows.txt
```


## 5) OpenMP duplicate runtime error (libomp.dll / libiomp5md.dll)

If you see:

```
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```

Use this order:

1. Ensure your env only has project deps (avoid mixing unrelated Conda packages in this env).
2. Reinstall torch/torchvision from the official index (step 3 above).
3. In PyCharm Run Configuration (or shell), set environment variables:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"
```

Project entry scripts (`train.py`, `test.py`) also apply this guard automatically on Windows.

## 6) Quick torch sanity check

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## 7) Download model/data assets (one-shot)

```powershell
python scripts/setup_assets.py
```

## 8) Run simple inference

```powershell
python -B -u test.py --eval-path ./data/demo_data/Layout/PubLayNet/data/PubLayNet/test/ --stage inference --restore-from ./pretrained_model/docsam_large_all_dataset.pth --model-size large --save-path ./outputs/outputs_test/demo/ --max-num 10 --short-range 704,896 --patch-size 640,640 --patch-num 1 --keep-size False --gpus 0
```

Outputs are saved under `./outputs/outputs_test/demo/`.
