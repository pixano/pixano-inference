# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Generate docs/openapi.json from the FastAPI app (the /v1 contract).

Run ``python scripts/gen_openapi.py`` to regenerate. CI runs it with ``--check`` to ensure
the committed schema is in sync (so the Pixano frontend can regenerate TypeScript types).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    """Write (or check) docs/openapi.json against the current app schema."""
    from pixano_inference.ray.app import create_ray_serve_app
    from pixano_inference.ray.config import RayServeConfig

    app, _ = create_ray_serve_app(RayServeConfig(num_gpus=0))
    spec = json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"

    target = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"

    if "--check" in sys.argv:
        current = target.read_text() if target.exists() else ""
        if current != spec:
            print(f"{target} is out of date. Run: python scripts/gen_openapi.py", file=sys.stderr)
            return 1
        print("openapi.json is up to date.")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(spec)
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
