import pytest


def test_short_mel_is_padded_to_the_whisper_length() -> None:
    torch = pytest.importorskip("torch")
    from vaakku.training.loop import _fit_mel_length

    class _Conv:
        def __init__(self, stride: int) -> None:
            self.stride = (stride,)

    class _Encoder:
        class config:
            max_source_positions = 1500

        conv1 = _Conv(1)
        conv2 = _Conv(2)

    class _Model:
        def get_encoder(self):
            return _Encoder()

    features = torch.zeros(2, 80, 1008)
    fitted = _fit_mel_length(_Model(), features)
    assert fitted.shape == (2, 80, 3000)
    assert fitted[..., 1008:].eq(0).all()


def test_long_labels_are_clipped_to_the_decoder_limit() -> None:
    torch = pytest.importorskip("torch")
    from vaakku.training.features import truncate_label_ids
    from vaakku.training.loop import _fit_label_length

    class _Tokenizer:
        model_max_length = 448
        eos_token_id = 2

        def __call__(self, text: str):
            class _Out:
                input_ids = list(range(517))

            return _Out()

    clipped = truncate_label_ids(_Tokenizer(), "unused")
    assert len(clipped) == 448
    assert clipped[-1] == 2

    class _Model:
        class config:
            max_target_positions = 448

    labels = torch.arange(517).view(1, 517)
    fitted = _fit_label_length(_Model(), labels)
    assert fitted.shape == (1, 448)
