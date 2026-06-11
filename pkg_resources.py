from packaging.version import parse as _parse

# Minimal shim providing parse_version used by tensorflow_hub
def parse_version(version):
    return _parse(version)

# Provide a fallback attribute for compatibility
__all__ = ["parse_version"]
