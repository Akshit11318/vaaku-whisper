# Vaakku — Open Indic Speech

Multilingual speech-to-text for Tamil (`ta`), Telugu (`te`), Hindi (`hi`), Kannada (`kn`), and Malayalam (`ml`).

- Model family: `vaakku-whisper-small`
- Demo: Vaakku Studio
- Code: https://github.com/Akshit11318/vaaku-whisper
- Model repository: https://huggingface.co/rudra11/vaakku-whisper-small

This repository is built in stages. The training instance is prepared by hand. GitHub Actions will only run tests and image builds. Nothing here launches EC2, trains on push, or uploads a model on push.

## Stages

1. **GPU box.** Mount the large disk, install the NVIDIA driver, and point dataset caches away from the 64 GiB root disk. Scripts: `scripts/setup_aws_box.sh`, `scripts/verify_gpu.sh`. See [docs/aws_training.md](docs/aws_training.md).
2. **Project skeleton.** Configuration, logging, and packaging.
3. **Data pipeline.** Adapters, normalization, speaker-disjoint splits, and a data-quality report.
4. **LoRA training.** `openai/whisper-small` with PEFT, plus a quick pipeline run and a full run.
5. **Evaluation.** WER and CER overall and per language.
6. **Publishing.** An explicit script you run after evaluation.
7. **Inference API and Vaakku Studio.**
8. **Docker and CI.** No continuous deployment.

Stage 1 is what you can run on the instance today. Later stages land in this same repository; pull them before running training.

## Box layout

Datasets and checkpoints go on `/home/ubuntu/vaaku`, the instance-store mount. Details are in [data/README.md](data/README.md).

```bash
bash scripts/setup_aws_box.sh
source /etc/profile.d/vaaku.sh
bash scripts/verify_gpu.sh
```
