"""Run optimization or independent prediction on your Modal account."""

from pathlib import Path

import modal

CODE = Path(__file__).resolve().parent if modal.is_local() else Path("/opt/carryover")
app = modal.App("structural-carryover")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgomp1")
    .pip_install_from_requirements(str(CODE / "requirements-linux-cuda12.lock"))
    .run_commands(
        'python -c "from pathlib import Path; from boltz.main import download_boltz2; '
        "p=Path('/root/.boltz'); p.mkdir(exist_ok=True); download_boltz2(p)\""
    )
    .add_local_dir(
        str(CODE),
        "/opt/carryover",
        copy=True,
        ignore=[
            "**/__pycache__/**",
            "**/*.pyc",
            ".pytest_cache/**",
            ".ruff_cache/**",
            "dist/**",
            ".venv/**",
            "runs/**",
            "run-*/**",
            ".git/**",
        ],
    )
    .run_commands("cd /opt/carryover && uv pip install --system --no-deps .")
    .env(
        {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
            "PYTHONUNBUFFERED": "1",
        }
    )
)


@app.function(
    image=image, gpu="H200", cpu=4, memory=32768, timeout=2400, max_containers=1, retries=0
)
def execute(config_data: dict, input_files: dict, carryover: bool, smoke: bool, mode: str, binder: str):
    import tempfile
    from dataclasses import replace

    from structural_carryover.transport import unpack
    parent = Path(tempfile.mkdtemp())
    config = unpack(config_data, input_files, parent/'input')
    config = replace(config, carryover=carryover,
                     phase_steps=(1, 1, 1) if smoke else config.phase_steps)
    folder = parent/'run'
    if mode == "predict":
        from structural_carryover.predict import predict
        result = predict(config, binder, folder)
    else:
        from structural_carryover.runner import run
        result = run(config, folder)
    files = {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    return {"result": result, "files": files}


@app.local_entrypoint()
def main(config: str, output_dir: str = "run", carryover: str = "on",
         smoke: bool = False, mode: str = "optimize", binder: str = ""):
    import json
    import sys

    sys.path.insert(0, str(CODE / "src"))
    from structural_carryover.transport import pack
    from structural_carryover.predict import read_binder

    if carryover not in {"on", "off"} or mode not in {"optimize", "predict"}:
        raise ValueError("choose carryover on/off and mode optimize/predict")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config_data, input_files = pack(Path(config))
    binder_sequence = read_binder(Path(binder)) if mode == "predict" else ""
    payload = execute.remote(config_data, input_files, carryover == "on", smoke, mode, binder_sequence)
    output.mkdir(parents=True)
    for name, data in payload["files"].items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(json.dumps(payload["result"], indent=2))
    print(f"Saved outputs to {output.resolve()}")
