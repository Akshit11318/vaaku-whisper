# Data location

Raw audio, prepared manifests, and dataset caches do not belong in git.

On the training instance the 233 GiB instance-store disk is mounted at `/home/ubuntu/vaaku`. Put datasets here:

| Path | Use |
|---|---|
| `/home/ubuntu/vaaku/data/raw` | Downloaded datasets |
| `/home/ubuntu/vaaku/data/manifests` | Normalized JSONL manifests |
| `/home/ubuntu/vaaku/data/prepared` | Train, validation, and test splits |
| `/home/ubuntu/vaaku/artifacts` | Checkpoints, evaluation reports, logs |
| `/home/ubuntu/vaaku/cache/huggingface` | Hugging Face hub and dataset cache |
| `/home/ubuntu/vaaku/tmp` | Temporary files |

The root volume is about 64 GiB. Training caches and checkpoints will fill it. The shell profile `/etc/profile.d/vaaku.sh` points `HF_HOME`, `HF_DATASETS_CACHE`, `PIP_CACHE_DIR`, and `TMPDIR` at the large disk.

That instance-store disk is wiped when the instance is **stopped** or **terminated**. A reboot keeps it. Copy anything you need to keep onto EBS or S3 before you stop the instance.

Source and license checks for IndicVoices, Kathbath, FLEURS, Vistaar, and Common Voice happen before any of those datasets are enabled for training. Nothing here is downloaded automatically.
