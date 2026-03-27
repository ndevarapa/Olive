# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
from pathlib import Path
from typing import Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.handler.multi_target import MultiTargetModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class ModelPackager(Pass):
    """Generate an ORT model package with manifest.json and per-component metadata.json.

    This pass takes a MultiTargetModelHandler (produced by EPContextBinaryGenerator with
    a list of provider_options) and generates a model package following the ORT spec:

    - manifest.json at package root with component_models and model_variants
    - metadata.json per component model directory with variant descriptors

    Variant constraints include:
    - ep (required): execution provider name
    - device (optional): target device type (cpu, gpu, npu)
    - architecture (optional): hardware architecture hint
    - ep_compatibility_info (optional): EP-specific compatibility string
    """

    _accepts_composite_model = True
    _accepts_multi_target_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "model_name": PassConfigParam(
                type_=str,
                default_value=None,
                description="Model name for the manifest. If not set, derived from the output directory name.",
            ),
        }

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self,
        model: MultiTargetModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> MultiTargetModelHandler:
        assert isinstance(model, MultiTargetModelHandler), "ModelPackager requires a MultiTargetModelHandler as input."

        output_dir = Path(output_model_path).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = config.model_name or output_dir.name

        # Build model_variants dict and copy files into component directory
        component_dir = output_dir / model_name
        component_dir.mkdir(parents=True, exist_ok=True)

        model_variants = {}
        for target_name, target_model in model.get_target_models():
            target_attrs = target_model.model_attributes or {}

            self._copy_target_model(target_name, target_model, component_dir)

            file_path = self._get_relative_model_path(target_name, target_model)

            constraints = {"ep": self.accelerator_spec.execution_provider}
            device = target_attrs.get("device")
            if device:
                constraints["device"] = device
            architecture = target_attrs.get("architecture")
            if architecture:
                constraints["architecture"] = architecture
            ep_compat = target_attrs.get("ep_compatibility_info")
            if ep_compat:
                constraints["ep_compatibility_info"] = ep_compat

            model_variants[target_name] = {"file": file_path, "constraints": constraints}

        # Write metadata.json in the component directory
        metadata = {"name": model_name, "model_variants": model_variants}
        metadata_path = component_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Generated metadata at %s", metadata_path)

        # Write manifest.json at package root
        manifest = {
            "name": model_name,
            "component_models": {
                model_name: {"model_variants": model_variants},
            },
        }
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Generated manifest at %s", manifest_path)

        # Update model_attributes
        new_model_attributes = model.model_attributes or {}
        new_model_attributes = {**new_model_attributes, "manifest_path": str(manifest_path)}
        new_model_attributes.pop("additional_files", None)

        return MultiTargetModelHandler(
            [target_model for _, target_model in model.get_target_models()],
            [target_name for target_name, _ in model.get_target_models()],
            model_path=output_dir,
            model_attributes=new_model_attributes,
        )

    @staticmethod
    def _copy_target_model(
        target_name: str,
        target_model: Union[ONNXModelHandler, CompositeModelHandler],
        output_dir: Path,
    ) -> None:
        dest_dir = output_dir / target_name
        if dest_dir.exists():
            return

        if isinstance(target_model, CompositeModelHandler):
            src_dir = Path(target_model.model_path)
        else:
            src_dir = Path(target_model.model_path).parent

        if src_dir.is_dir():
            shutil.copytree(str(src_dir), str(dest_dir))
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(target_model.model_path), str(dest_dir))

    @staticmethod
    def _get_relative_model_path(
        target_name: str,
        target_model: Union[ONNXModelHandler, CompositeModelHandler],
    ) -> str:
        if isinstance(target_model, ONNXModelHandler):
            return f"{target_name}/{Path(target_model.model_path).name}"
        # For CompositeModelHandler or other types, use the directory
        return f"{target_name}/"
