from io import BytesIO

from PIL import Image


def test_load_local_image_returns_rgb(tmp_path, image_loader_module):
    image_path = tmp_path / "sample.png"
    Image.new("RGBA", (8, 6), (255, 0, 0, 128)).save(image_path)

    loader = image_loader_module.ImageLoader(str(image_path), source=True)
    result = loader.load_image()

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (8, 6)


def test_load_url_image_uses_timeout_and_returns_rgb(monkeypatch, image_loader_module):
    buffer = BytesIO()
    Image.new("RGBA", (5, 4), (0, 255, 0, 128)).save(buffer, format="PNG")

    calls = {}

    class FakeResponse:
        content = buffer.getvalue()

        def raise_for_status(self):
            calls["raise_for_status"] = True

    def fake_get(url, timeout):
        calls["url"] = url
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(image_loader_module.requests, "get", fake_get)

    url = "https://example.test/image.png"
    loader = image_loader_module.ImageLoader(url, source=False)
    result = loader.load_image()

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (5, 4)
    assert calls == {
        "url": url,
        "timeout": 10,
        "raise_for_status": True,
    }


def test_load_image_returns_none_when_open_fails(
    monkeypatch,
    image_loader_module,
):
    def broken_open(*args, **kwargs):
        raise OSError("broken image")

    monkeypatch.setattr(image_loader_module.Image, "open", broken_open)

    loader = image_loader_module.ImageLoader("missing.jpg", source=True)

    assert loader.load_image() is None


def test_load_url_returns_none_on_http_error(
    monkeypatch,
    image_loader_module,
):
    class FakeResponse:
        content = b""

        def raise_for_status(self):
            raise RuntimeError("HTTP 500")

    monkeypatch.setattr(
        image_loader_module.requests,
        "get",
        lambda url, timeout: FakeResponse(),
    )

    loader = image_loader_module.ImageLoader(
        "https://example.test/fail.jpg",
        source=False,
    )

    assert loader.load_image() is None
