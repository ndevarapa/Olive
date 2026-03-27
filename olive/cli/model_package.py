# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

from olive.cli.base import BaseOliveCLICommand, add_logging_options, add_telemetry_options
from olive.common.utils import hardlink_copy_dir
from olive.telemetry import action

logger = logging.getLogger(__name__)


@action
class ModelPackageCommand(BaseOliveCLICommand):
    """Merge multiple single-target context binary outputs into a multi-target package with manifest.json."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "model-package",
            help="Merge multiple context binary outputs into a multi-target package with manifest.json",
        )

        sub_parser.add_argument(
            "-s",
            "--source",
            type=str,
            action="append",
            required=True,
            help=("Source context binary output directory. Can be specified multiple times. "),
        )

        sub_parser.add_argument(
            "-o",
            "--output_path",
            type=str,
            required=True,
            help="Output directory for the merged multi-target package.",
        )

        sub_parser.add_argument(
            "--model_name",
            type=str,
            default=None,
            help="Model name for the manifest. If not set, derived from the output directory name.",
        )

        add_logging_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=ModelPackageCommand)

    def run(self):
        sources = self._parse_sources()
        output_dir = Path(self.args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = self.args.model_name or output_dir.name
        manifest = {"name": model_name, "components": []}

        for target_name, source_path in sources:
            # Read model_config.json from source
            model_config = self._read_model_config(source_path)
            model_attrs = model_config.get("config", {}).get("model_attributes") or {}

            # Copy source directory to output/{target_name}/
            target_dir = output_dir / target_name
            hardlink_copy_dir(source_path, target_dir)

            constraints = {}
            for key in ("ep", "device", "architecture", "precision", "sdk_version"):
                if model_attrs.get(key) is not None:
                    constraints[key] = model_attrs[key]

            entry = {
                "variant_name": target_name,
                "file": {model_config["model_path"]},
                "constraints": constraints,
            }

            manifest["components"].append(entry)

        # Write manifest.json
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Merged {len(sources)} targets into {output_dir}")

    def _parse_sources(self) -> list[tuple[str, Path]]:
        sources = []
        for source in self.args.source:
            path = Path(source)
            if not path.is_dir():
                raise ValueError(f"Source path does not exist or is not a directory: {path}")

            # Validate model_config.json exists
            if not (path / "model_config.json").exists():
                raise ValueError(
                    f"No model_config.json found in {path}. "
                    "Source must be an Olive output directory with model_config.json."
                )

            sources.append((path.name, path))

        if len(sources) < 2:
            raise ValueError("At least two --source directories are required to merge.")

        return sources

    @staticmethod
    def _read_model_config(source_path: Path) -> dict:
        """Read and return model_config.json from a source directory."""
        config_path = source_path / "model_config.json"
        with open(config_path) as f:
            return json.load(f)
