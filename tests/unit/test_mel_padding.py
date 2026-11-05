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
