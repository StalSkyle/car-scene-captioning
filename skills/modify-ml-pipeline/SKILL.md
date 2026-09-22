# Modify ML Pipeline

Use this skill when adding or changing behaviour that affects one or more stages of the image-processing or ML pipeline.

Examples include:

* adding configuration parameters;
* changing detector or classifier behaviour;
* adding a new pipeline component;
* supporting a new image input form;
* adding or changing model inference behaviour.

## Procedure

### 1. Identify affected components

Before editing code, determine which existing component owns each part of the requested behaviour.

Prefer extending an existing component over implementing the same responsibility in `main.py`.

### 2. Preserve the package boundary

Reusable pipeline logic belongs inside `autocaption/`.

Do not create shared helper modules in the project root when they are used by `autocaption` modules.

Example:

```text
GOOD:
autocaption/confidence.py

BAD:
confidence.py
```

### 3. Preserve the public API

If `main.py` needs a new helper, class, or configuration object:

1. implement it inside `autocaption`;
2. export it from `autocaption/__init__.py`;
3. add it to `autocaption.__all__`;
4. import it in `main.py` using:

```python
from autocaption import SomeComponent
```

Do not import internal package modules directly from `main.py`.

### 4. Keep `main.py` as the orchestrator

`main.py` should connect existing pipeline components and pass values between them.

Do not:

* implement model inference in `main.py`;
* import ML frameworks directly into `main.py`;
* load model weights from `main.py`;
* duplicate logic already owned by an `autocaption` component.

### 5. Reuse expensive objects

Models and heavy pipeline components must be initialized before the loop that processes input images.

Create them once and reuse them for all images in a `run_pipeline()` execution.

### 6. Put image loading behind `ImageLoader`

When adding support for a new image source, extend `ImageLoader` rather than introducing new direct image-loading operations elsewhere.

Do not add new `Image.open(...)` or `requests.get(...)` calls outside the image-loading component unless there is a documented architectural reason.

### 7. Keep local model paths under `MODELS/`

Any new local weight or model path must point into `MODELS/`.

Do not create alternative directories for model files.

### 8. Add tests

Add tests for:

* the new behaviour;
* invalid input where applicable;
* backward compatibility with the existing API where applicable.

Tests should avoid loading real large ML models when mocking or fakes can test the behaviour.

### 9. Run project checks

Before completing the task, run:

```bash
pytest -q
python3 check.py
```

Do not consider the task complete while either command fails.
