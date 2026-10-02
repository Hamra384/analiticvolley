"""Crea en CVAT local el proyecto de evaluación y una tarea por clip con la pre-anotación cargada (S2, #5).

Credenciales por variables de entorno (nunca en el repo): CVAT_URL (default http://localhost:8080),
CVAT_USER, CVAT_PASSWORD.

Uso (cvat-sdk efímero, no es dependencia del proyecto):
  uv run --with cvat-sdk python -m tools.cvat_setup [--clips A1 ...]
"""

from __future__ import annotations

import argparse
import os

from cvat_sdk import make_client
from cvat_sdk.api_client import models
from cvat_sdk.core.proxies.tasks import ResourceType

from volley_cv.config import PROJECT_ROOT, load_clips, load_settings

PROJECT = "MVP1 - evaluación"
LABELS = [
    {
        "name": "player",
        "type": "rectangle",
        "attributes": [
            {
                "name": "team",
                "mutable": False,
                "input_type": "select",
                "values": ["?", "A", "B"],
                "default_value": "?",
            },
            {"name": "jersey", "mutable": False, "input_type": "text", "values": [""], "default_value": ""},
            {
                "name": "libero",
                "mutable": False,
                "input_type": "checkbox",
                "values": ["false"],
                "default_value": "false",
            },
        ],
    },
    {"name": "ball", "type": "rectangle", "attributes": []},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*")
    args = ap.parse_args()
    url = os.environ.get("CVAT_URL", "http://localhost:8080")
    user, password = os.environ["CVAT_USER"], os.environ["CVAT_PASSWORD"]
    root = load_settings().data_dir / "annotations" / "cvat"
    catalog = load_clips(PROJECT_ROOT / "configs" / "eval" / "clips.yaml")

    with make_client(url, credentials=(user, password)) as client:
        projects = [p for p in client.projects.list() if p.name == PROJECT]
        project = (
            projects[0]
            if projects
            else client.projects.create(models.ProjectWriteRequest(name=PROJECT, labels=LABELS))
        )
        existing = {t.name for t in client.tasks.list() if t.project_id == project.id}
        for clip in catalog.clips:
            if args.clips and clip.id not in args.clips:
                continue
            if clip.id in existing:
                print(f"{clip.id}: ya existe, se saltea")
                continue
            d = root / clip.id
            task = client.tasks.create_from_data(
                spec=models.TaskWriteRequest(name=clip.id, project_id=project.id),
                resource_type=ResourceType.LOCAL,
                resources=[str(d / "images.zip")],
                data_params={"image_quality": 90, "sorting_method": "natural"},
                annotation_path=str(d / "mot.zip"),
                annotation_format="MOT 1.1",
            )
            print(f"{clip.id}: tarea {task.id} creada -> {url}/tasks/{task.id}", flush=True)


if __name__ == "__main__":
    main()
