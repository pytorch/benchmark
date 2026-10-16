import importlib


def test_scribe_uploaders_accept_torchvision_version():
    for module_name in ("scripts.upload_scribe", "scripts.upload_scribe_v2"):
        module = importlib.import_module(module_name)
        message = module.PytorchBenchmarkUploader().format_message(
            {"time": 1, "torchvision_version": "0.18.0"}
        )

        assert message["normal"]["torchvision_version"] == "0.18.0"
