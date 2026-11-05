from pathlib import Path

import pytest
from tests.conftest import write_tone

from vaakku.config import load_data_config, load_train_config
from vaakku.data.adapters import iter_fleurs
from vaakku.data.normalize import canonicalize_language, normalize_transcript
from vaakku.data.pipeline import prepare_dataset
from vaakku.data.schema import NormalizedSample
from vaakku.data.split import assign_splits
from vaakku.errors import ConfigError, DataError
from vaakku.evaluation.metrics import score_predictions
from vaakku.evaluation.text import cer, normalize_for_eval, wer
from vaakku.logging import reset_logging_state


def test_transcript_normalization_keeps_tamil() -> None:
    assert normalize_transcript("  வணக்கம்\u200b  உலகம் ") == "வணக்கம் உலகம்"
    assert canonicalize_language("hi_in") == "hi"
    assert canonicalize_language("Kannada") == "kn"


def test_eval_normalization_and_error_rates() -> None:
    assert normalize_for_eval("Hello, World!") == "hello world"
    assert wer("a b c", "a x c") == pytest.approx(1 / 3)
    assert cer("abc", "ab") == pytest.approx(1 / 3)


def test_macro_average_does_not_hide_a_language() -> None:
    scored = score_predictions(
        [
            {"reference": "hello", "hypothesis": "hello", "language": "hi"},
            {"reference": "one two three", "hypothesis": "no no no", "language": "ta"},
        ]
    )
    assert scored["macro_wer"] == pytest.approx(0.5)
    assert scored["micro_wer"] == pytest.approx(0.75)
    assert [row["language"] for row in scored["per_language"]] == ["hi", "ta"]


def test_speaker_disjoint_split_has_no_leakage() -> None:
    samples = []
    for speaker, count in (("a", 10), ("b", 10), ("c", 10)):
        for index in range(count):
            samples.append(
                NormalizedSample(
                    audio_path=f"{speaker}-{index}.wav",
                    transcription="hello",
                    language="hi",
                    speaker_id=speaker,
                    dataset_name="demo",
                    sample_id=f"{speaker}-{index}",
                    duration_s=1.0,
                )
            )
    buckets, method = assign_splits(
        samples, {"train": 0.8, "validation": 0.1, "test": 0.1}, 3, True
    )
    assert method == "speaker_disjoint"
    assert all(len(rows) > 0 for rows in buckets.values())
    ids = [row.sample_id for rows in buckets.values() for row in rows]
    assert len(ids) == len(set(ids))


def test_official_fleurs_split_is_preserved(tmp_path: Path) -> None:
    audio = tmp_path / "hi_in" / "audio" / "extracted" / "dev"
    write_tone(audio / "clip.wav", seconds=1.0)
    tsv = tmp_path / "hi_in" / "dev.tsv"
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text("1\tclip.wav\traw\tनमस्ते\tchars\t16000\tFEMALE\n", encoding="utf-8")
    rows = list(
        iter_fleurs(
            type("S", (), {"path": str(tmp_path), "name": "fleurs", "allow_download": False})()
        )
    )
    assert rows[0]["split"] == "validation"
    assert rows[0]["language"] == "hi"
    assert rows[0]["audio_path"].endswith("clip.wav")


def test_vistaar_is_refused() -> None:
    from vaakku.config.models import DataSourceConfig

    source = DataSourceConfig(name="vistaar", kind="vistaar", enabled=False)
    with pytest.raises(DataError):
        from vaakku.data.adapters import iter_source_rows

        list(iter_source_rows(source))


def test_prepare_logs_rejections_and_writes_test_split(tmp_path: Path) -> None:
    reset_logging_state()
    good = tmp_path / "good.wav"
    other = tmp_path / "other.wav"
    short = tmp_path / "short.wav"
    write_tone(good, seconds=1.0)
    write_tone(other, seconds=1.2, frequency=520)
    write_tone(short, seconds=0.1)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        "\n".join(
            [
                f'{{"audio_path": "{good}", "transcription": "hello", "language": "hi", "speaker_id": "s1", "sample_id": "a"}}',
                f'{{"audio_path": "{short}", "transcription": "hello", "language": "hi", "speaker_id": "s2", "sample_id": "b"}}',
                f'{{"audio_path": "{good}", "transcription": "  ", "language": "ta", "sample_id": "c"}}',
                f'{{"audio_path": "{other}", "transcription": "hello", "language": "hi", "speaker_id": "s3", "sample_id": "d", "split": "test"}}',
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        f"""
output_dir: {tmp_path / "out"}
languages: [hi, ta]
sources:
  - name: demo
    kind: manifest
    enabled: true
    path: {manifest}
logging:
  dir: {tmp_path / "logs"}
""",
        encoding="utf-8",
    )
    report = prepare_dataset(load_data_config(config_path))
    assert report.rejected == 2
    assert report.rejection_reasons["too_short"] == 1
    assert report.rejection_reasons["empty_transcript"] == 1
    test_rows = (tmp_path / "out" / "test.jsonl").read_text(encoding="utf-8")
    assert '"sample_id": "d"' in test_rows
    assert (
        "hello"
        not in (tmp_path / "out" / "quality_report.json")
        .read_text(encoding="utf-8")
        .split("rejections")[1]
    )


def test_configs_load() -> None:
    root = Path(__file__).resolve().parents[2]
    quick = load_train_config(root / "configs" / "train_quick.yaml")
    full = load_train_config(root / "configs" / "train.yaml")
    assert quick.purpose == "pipeline_validation"
    assert full.purpose == "experiment"
    assert load_data_config(root / "configs" / "data.yaml").sources[0].enabled is False


def test_bad_language_config(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("run_name: x\npurpose: experiment\nlanguages: [fr]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_train_config(path)
