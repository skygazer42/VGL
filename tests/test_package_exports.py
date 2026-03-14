from gnn import __version__


def test_package_exposes_version_string():
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"
