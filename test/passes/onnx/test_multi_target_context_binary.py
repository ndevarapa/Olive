# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.handler.multi_target import MultiTargetModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.model_packager import ModelPackager


def _make_onnx_handler(tmp_path, name="model", model_attributes=None):
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{name}.onnx"
    model_file.write_text("dummy")
    return ONNXModelHandler(model_path=str(model_file), model_attributes=model_attributes)


def _make_multi_target(tmp_path, target_configs):
    targets = []
    names = []
    for name, attrs in target_configs:
        handler = _make_onnx_handler(tmp_path, name=name, model_attributes=attrs)
        targets.append(handler)
        names.append(name)
    return MultiTargetModelHandler(targets, names, model_path=tmp_path, model_attributes={})


# ===========================================================================
# ModelPackager tests
# ===========================================================================


class TestModelPackager:
    def _create_packager(self, ep="QNNExecutionProvider", device="NPU", config=None):
        accelerator_spec = AcceleratorSpec(accelerator_type=device, execution_provider=ep)
        return create_pass_from_dict(
            ModelPackager,
            config or {},
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

    def test_packager_generates_manifest(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [
                ("soc_60", {"architecture": "60", "device": "NPU"}),
                ("soc_73", {"architecture": "73", "device": "NPU"}),
            ],
        )

        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        result = p.run(mt, output_path)

        assert isinstance(result, MultiTargetModelHandler)

        # manifest.json at package root
        manifest_path = tmp_path / "output" / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["name"] == "output"
        assert "output" in manifest["component_models"]
        variants = manifest["component_models"]["output"]["model_variants"]
        assert "soc_60" in variants
        assert "soc_73" in variants
        assert variants["soc_60"]["constraints"]["architecture"] == "60"
        assert variants["soc_73"]["constraints"]["architecture"] == "73"
        assert variants["soc_60"]["constraints"]["ep"] == "QNNExecutionProvider"

        # metadata.json in component directory
        metadata_path = tmp_path / "output" / "output" / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["name"] == "output"
        assert metadata["model_variants"] == variants

    def test_packager_ep_compatibility_info(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [
                ("soc_60", {"architecture": "60", "ep_compatibility_info": "device=npu;soc=60"}),
                ("soc_73", {"architecture": "73", "ep_compatibility_info": "device=npu;soc=73"}),
            ],
        )

        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        with open(tmp_path / "output" / "manifest.json") as f:
            manifest = json.load(f)

        variants = manifest["component_models"]["output"]["model_variants"]
        assert variants["soc_60"]["constraints"]["ep_compatibility_info"] == "device=npu;soc=60"
        assert variants["soc_73"]["constraints"]["ep_compatibility_info"] == "device=npu;soc=73"

    def test_packager_custom_model_name(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [("soc_60", {}), ("soc_73", {})],
        )

        p = self._create_packager(config={"model_name": "my_model"})
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        with open(tmp_path / "output" / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["name"] == "my_model"
        assert "my_model" in manifest["component_models"]

        # metadata.json under my_model/
        with open(tmp_path / "output" / "my_model" / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["name"] == "my_model"

    def test_packager_rejects_non_multi_target(self, tmp_path):
        handler = _make_onnx_handler(tmp_path, "single")
        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        with pytest.raises(AssertionError, match="requires a MultiTargetModelHandler"):
            p.run(handler, output_path)

    def test_packager_copies_files(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [("soc_60", {"architecture": "60"}), ("soc_73", {"architecture": "73"})],
        )

        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # Files are under output/<model_name>/<variant>/
        assert (tmp_path / "output" / "output" / "soc_60").is_dir()
        assert (tmp_path / "output" / "output" / "soc_73").is_dir()

    def test_packager_default_model_name_from_dir(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [("t1", {"architecture": "a"}), ("t2", {"architecture": "b"})],
        )

        p = self._create_packager()
        output_path = str(tmp_path / "my_package.onnx")
        p.run(mt, output_path)

        with open(tmp_path / "my_package" / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["name"] == "my_package"

    def test_packager_device_only_when_present(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [("t1", {"architecture": "a", "device": "GPU"}), ("t2", {"architecture": "b"})],
        )

        p = self._create_packager(device="NPU")
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        with open(tmp_path / "output" / "manifest.json") as f:
            manifest = json.load(f)

        variants = manifest["component_models"]["output"]["model_variants"]
        assert variants["t1"]["constraints"]["device"] == "GPU"
        assert "device" not in variants["t2"]["constraints"]

    def test_packager_optional_fields_omitted_when_absent(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [("t1", {}), ("t2", {})],
        )

        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        with open(tmp_path / "output" / "manifest.json") as f:
            manifest = json.load(f)

        variants = manifest["component_models"]["output"]["model_variants"]
        for v in variants.values():
            assert "device" not in v["constraints"]
            assert "architecture" not in v["constraints"]
            assert "ep_compatibility_info" not in v["constraints"]

    def test_packager_manifest_path_in_result_attributes(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [("t1", {"architecture": "a"}), ("t2", {"architecture": "b"})],
        )

        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        result = p.run(mt, output_path)

        assert "manifest_path" in result.model_attributes
        assert Path(result.model_attributes["manifest_path"]).name == "manifest.json"

    def test_packager_copy_skips_existing_dest(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [("t1", {"architecture": "a"}), ("t2", {"architecture": "b"})],
        )

        p = self._create_packager(config={"model_name": "mdl"})
        output_path = str(tmp_path / "output.onnx")
        output_dir = tmp_path / "output"
        component_dir = output_dir / "mdl"
        component_dir.mkdir(parents=True)

        # Pre-create dest with a marker file
        (component_dir / "t1").mkdir()
        (component_dir / "t1" / "marker.txt").write_text("pre-existing")

        p.run(mt, output_path)

        assert (component_dir / "t1" / "marker.txt").read_text() == "pre-existing"

    def test_packager_with_composite_model_handler(self, tmp_path):
        from olive.model import CompositeModelHandler

        comp_dir_1 = tmp_path / "comp1"
        comp_dir_1.mkdir()
        (comp_dir_1 / "model.onnx").write_text("dummy")

        comp_dir_2 = tmp_path / "comp2"
        comp_dir_2.mkdir()
        (comp_dir_2 / "model.onnx").write_text("dummy")

        sub1 = ONNXModelHandler(model_path=str(comp_dir_1 / "model.onnx"))
        sub2 = ONNXModelHandler(model_path=str(comp_dir_2 / "model.onnx"))

        comp1 = CompositeModelHandler(
            model_components=[sub1],
            model_component_names=["part1"],
            model_path=str(comp_dir_1),
            model_attributes={"architecture": "60"},
        )
        comp2 = CompositeModelHandler(
            model_components=[sub2],
            model_component_names=["part1"],
            model_path=str(comp_dir_2),
            model_attributes={"architecture": "73"},
        )

        mt = MultiTargetModelHandler([comp1, comp2], ["soc_60", "soc_73"], model_path=tmp_path)

        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        result = p.run(mt, output_path)

        with open(tmp_path / "output" / "manifest.json") as f:
            manifest = json.load(f)

        variants = manifest["component_models"]["output"]["model_variants"]
        assert variants["soc_60"]["file"] == "soc_60/"
        assert variants["soc_73"]["file"] == "soc_73/"

        # Files under component dir
        assert (tmp_path / "output" / "output" / "soc_60" / "model.onnx").exists()
        assert (tmp_path / "output" / "output" / "soc_73" / "model.onnx").exists()

        assert isinstance(result, MultiTargetModelHandler)

    def test_packager_onnx_model_uses_filename_in_file_field(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [("soc_60", {"architecture": "60"})],
        )
        h2 = _make_onnx_handler(tmp_path, name="soc_73", model_attributes={"architecture": "73"})
        mt = MultiTargetModelHandler(
            [next(t for _, t in mt.get_target_models()), h2],
            ["soc_60", "soc_73"],
            model_path=tmp_path,
        )

        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        with open(tmp_path / "output" / "manifest.json") as f:
            manifest = json.load(f)

        variants = manifest["component_models"]["output"]["model_variants"]
        assert variants["soc_60"]["file"] == "soc_60/soc_60.onnx"
        assert variants["soc_73"]["file"] == "soc_73/soc_73.onnx"


# ===========================================================================
# Pass.run() multi-target auto-dispatch tests
# ===========================================================================


class TestPassRunMultiTarget:
    def test_pass_run_iterates_targets(self, tmp_path):
        """A pass that does NOT accept multi-target should iterate over each target independently."""
        from olive.passes.onnx.float16_conversion import OnnxFloatToFloat16

        h1 = _make_onnx_handler(tmp_path, "t1", model_attributes={"architecture": "60"})
        h2 = _make_onnx_handler(tmp_path, "t2", model_attributes={"architecture": "73"})
        mt = MultiTargetModelHandler([h1, h2], ["t1", "t2"], model_path=tmp_path)

        accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider="QNNExecutionProvider")

        # Mock _run_for_config to just return a new handler (avoid real ONNX ops)
        with patch.object(OnnxFloatToFloat16, "_run_for_config") as mock_run:

            def side_effect(model, config, output_model_path):
                out_file = Path(output_model_path)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text("dummy")
                return ONNXModelHandler(model_path=str(out_file), model_attributes=model.model_attributes)

            mock_run.side_effect = side_effect

            p = create_pass_from_dict(OnnxFloatToFloat16, {}, disable_search=True, accelerator_spec=accelerator_spec)
            output_path = str(tmp_path / "output.onnx")
            result = p.run(mt, output_path)

        # Result should still be MultiTargetModelHandler
        assert isinstance(result, MultiTargetModelHandler)
        assert result.target_names == ["t1", "t2"]
        # _run_for_config was called twice, once per target
        assert mock_run.call_count == 2
