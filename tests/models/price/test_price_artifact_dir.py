from types import SimpleNamespace

from models.price.pyfunc import artifact_dir


def test_artifact_dir_keeps_a_path_that_exists(tmp_path):
    folder = tmp_path / "artifacts" / "model_dir"
    folder.mkdir(parents=True)
    context = SimpleNamespace(artifacts={"model_dir": str(folder)})
    assert artifact_dir(context, "model_dir") == folder


def test_artifact_dir_repairs_a_windows_style_relative_path(tmp_path):
    # A model logged on Windows stores "artifacts\\model_dir"; on Linux that is one bogus name.
    folder = tmp_path / "artifacts" / "model_dir"
    folder.mkdir(parents=True)
    windows_style = str(tmp_path) + "/artifacts\\model_dir"
    context = SimpleNamespace(artifacts={"model_dir": windows_style})
    assert artifact_dir(context, "model_dir").resolve() == folder.resolve()
