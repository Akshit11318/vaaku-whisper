from pathlib import Path

import pytest

from vaakku.config import load_train_config
from vaakku.logging import reset_logging_state


@pytest.mark.integration
def test_smoke_training_saves_a_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from vaakku.training.loop import run_training

    reset_logging_state()
    config = load_train_config(Path(__file__).resolve().parents[2] / "configs" / "train_smoke.yaml")
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path),
            "logging": config.logging.model_copy(update={"dir": str(tmp_path / "logs")}),
        }
    )
    output = run_training(config)
    assert (output / "run.json").is_file()
    assert (output / "best" / "adapter" / "adapter_config.json").is_file()
    assert (output / "metrics.jsonl").is_file()
    purpose = Path(output / "run.json").read_text(encoding="utf-8")
    assert "pipeline_validation" in purpose
