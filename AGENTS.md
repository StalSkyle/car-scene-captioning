# Project Instructions

This repository implements an image-processing and ML captioning pipeline through the `autocaption` package.

When modifying the project, preserve the existing package boundaries and pipeline architecture.

## 1. Keep `main.py` orchestration-only

`main.py` may:

* construct pipeline components;
* pass data between components;
* collect results;
* expose or call `run_pipeline()`.

Do not load or initialize ML models directly in `main.py`.

In particular, do not add direct imports of:

* `torch`
* `torchvision`
* `transformers`
* `ultralytics`

Do not call model-loading APIs such as `YOLO(...)`, `torch.load(...)`, or `from_pretrained(...)` from `main.py`.

## 2. Use the public `autocaption` API from `main.py`

Project components used by `main.py` must be imported through the package root:

```python
from autocaption import CarDetector, ObjectExtractor
```

Do not bypass the public API:

```python
from autocaption.image_processor import CarDetector
from autocaption.confidence import validate_confidence
```

If `main.py` needs a new reusable project component, expose that component through `autocaption/__init__.py`.

## 3. Keep the public API synchronized with `__all__`

Any symbol imported in the form:

```python
from autocaption import SomeComponent
```

must:

1. be imported in `autocaption/__init__.py`;
2. be present in `autocaption.__all__`.

When adding a new public component, update both.

## 4. Keep shared pipeline code inside `autocaption`

Reusable helpers used by modules inside `autocaption` must live inside the `autocaption` package.

Do not create project-root helper modules such as:

```text
confidence.py
validation.py
model_utils.py
```

and then import them from `autocaption`.

Instead use package-local modules, for example:

```text
autocaption/confidence.py
autocaption/validation.py
autocaption/model_utils.py
```

Use relative imports inside the package when appropriate.

## 5. Initialize heavy components outside image-processing loops

Heavy pipeline components and ML models must be initialized once and reused for multiple images.

Do not instantiate components such as:

* `ImageRotator`
* `CarDetector`
* `ObjectExtractor`
* `SceneExtractor`
* `PhotoDescriber`
* `PhotoDescriberWithQuestion`

inside a loop over input images.

Instantiate them before the loop and reuse the instances.

## 6. Route new image-loading logic through `ImageLoader`

New code that loads images from:

* local paths;
* URLs;
* streams or other external sources

must be implemented through `ImageLoader` or an extension of it.

Do not introduce new direct `requests.get(...)` or `Image.open(...)` calls in unrelated pipeline modules.

The repository contains legacy direct `Image.open(...)` calls in existing code; do not copy or expand that pattern.

## 7. Store local model weights under `MODELS/`

New local model files and weights must be loaded from the existing `MODELS/` directory.

Do not introduce additional model directories or arbitrary local weight paths.

Remote model identifiers such as Hugging Face model IDs are not local paths and are not subject to this rule.

## Verification

After changing the project, run:

```bash
pytest -q
python3 check.py
```

The implementation is not complete if:

* tests fail; or
* `check.py` reports new convention violations.
