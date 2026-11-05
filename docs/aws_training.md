# AWS training box

Vaakku does not create, start, or stop EC2 instances. You launch the instance yourself, then use the scripts in this repository.

Recommended starting point: `g5.xlarge` in the region you choose. One NVIDIA A10G, 24 GB GPU memory, 16 GB RAM. The code does not require that exact machine.

## Disk layout

The root volume on the instance used for setup is 64 GiB. The extra disk is the g5 NVMe instance store, about 233 GiB, mounted at `/home/ubuntu/vaaku`.

| Mount | Size | What goes there |
|---|---|---|
| `/` | 64 GiB | OS, NVIDIA driver, git checkout, Python virtualenv |
| `/home/ubuntu/vaaku` | 233 GiB | Datasets, checkpoints, Hugging Face cache, temp files |

Instance-store data survives a reboot. It is deleted when the instance stops or is terminated. Copy checkpoints off the box before you stop it.

## One-time setup

From a clone of this repository on the instance:

```bash
bash scripts/setup_aws_box.sh
source /etc/profile.d/vaaku.sh
bash scripts/verify_gpu.sh
```

`setup_aws_box.sh` does four things:

1. Installs git, ffmpeg, libsndfile, Python 3, and build tools.
2. Installs the Ubuntu NVIDIA driver branch `580-server` and the kernel module package that matches `uname -r`.
3. Stops the `nouveau` driver from holding the A10G, and loads the NVIDIA driver.
4. Formats the empty instance-store disk only when it has no filesystem, mounts it at `/home/ubuntu/vaaku`, and creates the data directories.

It will not format a disk that already has a filesystem unless that filesystem is already labeled `vaaku-data`.

After a stop/start the instance store comes back empty. Run `scripts/setup_aws_box.sh` again before training. The script mounts and, only if the disk is blank, formats it again.

## Check the GPU

`nvidia-smi` should show an NVIDIA A10G and about 23028 MiB. Driver 580 reports CUDA 13.0. PyTorch wheels bring their own CUDA runtime; a separate CUDA toolkit install is not required for training.

## Code and the virtualenv

Put the git checkout on the root disk, for example `/home/ubuntu/vaaku-whisper`. Do not clone into `/home/ubuntu/vaaku`. A stop deletes that mount.

The box already has a virtualenv at `/home/ubuntu/vaakku-venv`. Install this repo into that environment before preparation. Dataset caches stay on the instance store through `/etc/profile.d/vaaku.sh`.

## What has not been started

`scripts/prepare_data.py`, `scripts/train.py`, `scripts/evaluate.py`, and `scripts/publish_model.py` are in the repo. None of them has been run on the instance. GitHub Actions does not launch them, and `setup_aws_box.sh` does not start them.

The first data command, after the archive sizes look right, is:

```bash
python scripts/prepare_data.py --config configs/data_fleurs.yaml
```

That unpacks FLEURS and writes manifests. It is not the training run. Training is a separate command, `python scripts/train.py --config configs/train.yaml`, and it has not been started.
