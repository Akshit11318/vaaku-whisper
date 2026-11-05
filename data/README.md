# Data location

Audio does not go in git. Experiment 1 has not downloaded anything yet. The dataset choice and the licenses are in the [README](../README.md).

On the training instance, put files on the instance-store mount:

| Path | Use |
|---|---|
| `/home/ubuntu/vaaku/data/raw/fleurs` | `google/fleurs` for `hi_in`, `ta_in`, `te_in`, `kn_in`, `ml_in` |
| `/home/ubuntu/vaaku/data/raw/common-voice` | Scripted 27.0 validated clips for Hindi, Tamil, Telugu, and Malayalam |
| `/home/ubuntu/vaaku/data/raw/kathbath` | Kathbath for Tamil, Telugu, Hindi, Kannada, and Malayalam only |
| `/home/ubuntu/vaaku/data/manifests` | Normalized JSONL, once a prepare script exists |
| `/home/ubuntu/vaaku/data/prepared` | Train and validation manifests. FLEURS `test` stays separate |
| `/home/ubuntu/vaaku/artifacts` | Checkpoints, evaluation, logs |
| `/home/ubuntu/vaaku/cache/huggingface` | Hub and dataset cache |
| `/home/ubuntu/vaaku/tmp` | Temporary files |

Do not download IndicVoices or the Vistaar benchmark packs for this experiment. Do not download the other seven Kathbath languages. Check `df -h /home/ubuntu/vaaku` first.

`/etc/profile.d/vaaku.sh` points `HF_HOME`, `HF_DATASETS_CACHE`, `PIP_CACHE_DIR`, and `TMPDIR` at this disk so the 64 GiB root volume is not the cache.

This disk is wiped when the instance stops or is terminated. A reboot keeps it.

Common Voice files stay on this disk. The Common Voice terms forbid re-hosting them and forbid trying to identify speakers. FLEURS is CC-BY-4.0, so a future model card has to attribute it. Kathbath's card says the packaging is CC0 and the text comes from the IndicCorp crawl.
