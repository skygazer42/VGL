import email
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_release_artifacts(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("release-dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output_dir)],
        cwd=REPO_ROOT,
        check=True,
    )
    wheel_path = next(output_dir.glob("*.whl"))
    sdist_path = next(output_dir.glob("*.tar.gz"))
    return wheel_path, sdist_path


def _wheel_metadata(wheel_path: Path):
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith("METADATA"))
        return email.message_from_bytes(archive.read(metadata_name))


@pytest.fixture(scope="module")
def built_release_artifacts(tmp_path_factory):
    return _build_release_artifacts(tmp_path_factory)


def test_release_metadata_exposes_public_package_information(built_release_artifacts):
    wheel_path, _ = built_release_artifacts
    metadata = _wheel_metadata(wheel_path)
    project_urls = metadata.get_all("Project-URL", [])

    assert metadata["Name"] == "sky-vgl"
    assert metadata["Requires-Python"] == ">=3.10"
    assert metadata["Author-email"]
    assert metadata["Keywords"]
    assert metadata.get_all("Classifier")
    assert "Homepage, https://github.com/skygazer42/sky-vgl" in project_urls
    assert "Repository, https://github.com/skygazer42/sky-vgl" in project_urls
    assert "Documentation, https://github.com/skygazer42/sky-vgl#readme" in project_urls
    assert "Issues, https://github.com/skygazer42/sky-vgl/issues" in project_urls
    assert "Changelog, https://github.com/skygazer42/sky-vgl/releases" in project_urls
    assert set(metadata.get_all("Provides-Extra", [])) >= {
        "dev",
        "scipy",
        "networkx",
        "tensorboard",
        "dgl",
        "pyg",
        "full",
    }


def test_release_artifacts_exclude_internal_repo_only_content(built_release_artifacts):
    wheel_path, sdist_path = built_release_artifacts

    with zipfile.ZipFile(wheel_path) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdist_path) as archive:
        sdist_names = archive.getnames()

    assert not any("/.factory/" in name for name in wheel_names)
    assert not any("/.factory/" in name for name in sdist_names)
    assert not any("/docs/plans/" in name for name in wheel_names)
    assert not any("/docs/plans/" in name for name in sdist_names)
    assert not any("__pycache__" in name for name in wheel_names)
    assert not any("__pycache__" in name for name in sdist_names)
    assert not any(name.startswith("examples/") for name in wheel_names)
    assert any("/examples/homo/node_classification.py" in name for name in sdist_names)
    assert any(name.endswith("/README.md") for name in sdist_names)
    assert any(name.endswith("/LICENSE") for name in sdist_names)


def test_release_workflows_exist_for_ci_and_pypi_publish():
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    publish_path = REPO_ROOT / ".github" / "workflows" / "publish.yml"

    assert ci_path.exists()
    assert publish_path.exists()

    ci_text = ci_path.read_text(encoding="utf-8")
    publish_text = publish_path.read_text(encoding="utf-8")

    assert "python -m pytest -q" in ci_text
    assert "python -m build" in ci_text
    assert "python -m twine check" in ci_text
    assert "tags:" in publish_text
    assert "v*" in publish_text
    assert "testpypi" in publish_text.lower()
    assert "pypi" in publish_text.lower()
    assert "id-token: write" in publish_text


def test_release_readme_documents_public_install_paths():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (REPO_ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
    releasing = (REPO_ROOT / "docs" / "releasing.md").read_text(encoding="utf-8")

    assert "version-0.1.3" in readme
    assert "Version 0.1.3" in readme
    assert 'pip install sky-vgl' in readme
    assert 'pip install "sky-vgl[full]"' in readme
    assert 'pip install "sky-vgl[networkx]"' in readme
    assert 'pip install "sky-vgl[pyg]"' in readme
    assert 'pip install "sky-vgl[dgl]"' in readme
    assert "git clone https://github.com/skygazer42/sky-vgl.git" in readme
    assert "cd sky-vgl" in readme
    assert "pip install sky-vgl" in quickstart
    assert 'pip install "sky-vgl[full]"' in quickstart
    assert "sky-vgl project name" in releasing
    assert "sky-vgl` works in a clean environment" in releasing
