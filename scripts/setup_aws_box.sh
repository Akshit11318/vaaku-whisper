#!/usr/bin/env bash
# Prepare an Ubuntu GPU box for Vaakku training.
# Mount the instance-store disk at /home/ubuntu/vaaku and install a kernel-matched NVIDIA driver.
# This does not train, download datasets, or publish a model.
set -euo pipefail

MOUNT="${VAAKKU_MOUNT:-/home/ubuntu/vaaku}"
DRIVER_BRANCH="${VAAKKU_NVIDIA_BRANCH:-580-server}"
LABEL="vaaku-data"

log() {
  printf '[setup] %s\n' "$*"
}

die() {
  printf '[setup] ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "$(id -u)" -eq 0 ]]; then
  die "Run this as the ubuntu user, not as root. The script uses sudo where needed."
fi

wait_for_apt() {
  local _i
  for _i in $(seq 1 90); do
    if pgrep -x apt >/dev/null || pgrep -x apt-get >/dev/null || pgrep -x dpkg >/dev/null; then
      log "apt is busy; waiting"
      sleep 10
    else
      return 0
    fi
  done
  die "apt is still running after 15 minutes"
}

find_instance_store() {
  lsblk -d -n -o NAME,MODEL | awk '$0 ~ /Instance Storage/ {print $1; exit}'
}

install_packages() {
  local kernel module_pkg
  kernel="$(uname -r)"
  module_pkg="linux-modules-nvidia-${DRIVER_BRANCH}-${kernel}"
  log "Installing system packages and ${module_pkg}"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    "${module_pkg}" \
    "nvidia-headless-no-dkms-${DRIVER_BRANCH}" \
    "nvidia-utils-${DRIVER_BRANCH}" \
    pciutils \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    pkg-config \
    ffmpeg \
    libsndfile1 \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    tmux \
    htop \
    unzip \
    jq
}

prepare_disk() {
  local disk fstype current_label uuid
  if findmnt "${MOUNT}" >/dev/null; then
    log "${MOUNT} is already mounted"
    return 0
  fi

  disk="$(find_instance_store || true)"
  if [[ -z "${disk}" ]]; then
    die "No EC2 instance-store disk found. Attach one, or set VAAKKU_MOUNT to an existing mount."
  fi
  disk="/dev/${disk}"

  fstype="$(lsblk -no FSTYPE "${disk}" | awk 'NF {print; exit}')"
  current_label="$(lsblk -no LABEL "${disk}" | awk 'NF {print; exit}')"

  if [[ -n "${fstype}" && "${current_label}" != "${LABEL}" ]]; then
    die "${disk} already has filesystem ${fstype} and label '${current_label:-none}'. Refusing to format it."
  fi

  if [[ -z "${fstype}" ]]; then
    log "Formatting empty instance store ${disk} as ext4 (${LABEL})"
    sudo mkfs.ext4 -F -L "${LABEL}" "${disk}"
  fi

  sudo mkdir -p "${MOUNT}"
  sudo mount -o defaults,noatime "LABEL=${LABEL}" "${MOUNT}"

  uuid="$(sudo blkid -s UUID -o value "${disk}")"
  if ! grep -q "${MOUNT}" /etc/fstab; then
    echo "UUID=${uuid} ${MOUNT} ext4 defaults,noatime,nofail,x-systemd.device-timeout=10 0 2" | sudo tee -a /etc/fstab >/dev/null
    log "Added ${MOUNT} to /etc/fstab"
  fi
}

prepare_dirs() {
  sudo mkdir -p \
    "${MOUNT}/data/raw" \
    "${MOUNT}/data/manifests" \
    "${MOUNT}/data/prepared" \
    "${MOUNT}/artifacts" \
    "${MOUNT}/cache/huggingface" \
    "${MOUNT}/cache/pip" \
    "${MOUNT}/tmp"
  sudo chown -R "$(id -u)":"$(id -g)" "${MOUNT}"
  cat > "${MOUNT}/DISK_README.txt" << EOF
This disk is mounted at ${MOUNT}.

Put datasets, Hugging Face caches, checkpoints, and training artifacts here.
Keep them off the root volume.

On a g5.xlarge this disk is NVMe instance store. A reboot keeps the data.
Stopping or terminating the instance wipes it. Copy checkpoints to EBS or S3 before you stop the instance.
EOF
}

write_env() {
  sudo tee /etc/profile.d/vaaku.sh >/dev/null << EOF
export VAAKKU_DATA_ROOT=${MOUNT}
export VAAKKU_DATA_DIR=${MOUNT}/data
export VAAKKU_ARTIFACT_DIR=${MOUNT}/artifacts
export HF_HOME=${MOUNT}/cache/huggingface
export HUGGINGFACE_HUB_CACHE=${MOUNT}/cache/huggingface/hub
export HF_DATASETS_CACHE=${MOUNT}/cache/huggingface/datasets
export PIP_CACHE_DIR=${MOUNT}/cache/pip
export TMPDIR=${MOUNT}/tmp
EOF
  sudo chmod 644 /etc/profile.d/vaaku.sh
  if ! grep -q '/etc/profile.d/vaaku.sh' "${HOME}/.bashrc"; then
    printf '\n# Vaakku data disk and caches\n[ -f /etc/profile.d/vaaku.sh ] && . /etc/profile.d/vaaku.sh\n' >> "${HOME}/.bashrc"
  fi
}

blacklist_nouveau() {
  sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null << 'EOS'
blacklist nouveau
options nouveau modeset=0
EOS
  echo nvidia | sudo tee /etc/modules-load.d/nvidia.conf >/dev/null
  sudo update-initramfs -u
}

load_driver() {
  local bound
  if nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver is already loaded"
    return 0
  fi
  if [[ -d /sys/bus/pci/drivers/nouveau ]]; then
    for bound in /sys/bus/pci/drivers/nouveau/0000:*; do
      [[ -e "${bound}" ]] || continue
      basename "${bound}" | sudo tee /sys/bus/pci/drivers/nouveau/unbind >/dev/null || true
    done
  fi
  sudo rmmod nvidia_drm 2>/dev/null || true
  sudo rmmod nvidia_modeset 2>/dev/null || true
  sudo rmmod nvidia_uvm 2>/dev/null || true
  sudo rmmod nvidia 2>/dev/null || true
  sudo rmmod nouveau 2>/dev/null || true
  sudo modprobe nvidia || true
  if nvidia-smi >/dev/null 2>&1; then
    sudo nvidia-smi -pm 1 || true
    log "NVIDIA driver loaded"
    return 0
  fi
  log "Driver packages are installed, but nvidia-smi is not working yet. Reboot, then run scripts/verify_gpu.sh."
  return 1
}

wait_for_apt
blacklist_nouveau
install_packages
prepare_disk
prepare_dirs
write_env
load_driver || true
log "Data root: ${MOUNT}"
log "Next: source /etc/profile.d/vaaku.sh && scripts/verify_gpu.sh"
