import pytest

from pyro_dataset.ingest import ValidationResult, validate_sequence_folder


@pytest.fixture
def valid_folder(tmp_path):
    """Create a minimal valid sequence folder."""
    name = "sdis83_brison_200_2024-01-15T10-30-00"
    folder = tmp_path / name
    (folder / "images").mkdir(parents=True)
    (folder / "labels").mkdir(parents=True)
    # 3 images with correct naming
    for ts in ["T10-30-00", "T10-31-00", "T10-32-00"]:
        (folder / "images" / f"sdis83_brison_200_2024-01-15{ts}.jpg").touch()
    # 3 label files, 2 non-empty
    (folder / "labels" / "sdis83_brison_200_2024-01-15T10-30-00.txt").write_text(
        "0 0.5 0.5 0.1 0.1"
    )
    (folder / "labels" / "sdis83_brison_200_2024-01-15T10-31-00.txt").write_text(
        "0 0.4 0.4 0.1 0.1"
    )
    (folder / "labels" / "sdis83_brison_200_2024-01-15T10-32-00.txt").touch()  # empty
    return folder


def test_valid_folder(valid_folder):
    result = validate_sequence_folder(valid_folder)
    assert result.is_valid


@pytest.mark.parametrize(
    "name",
    [
        "sdis83_brison_200_2024-01-15T10-30-00",       # hyphen separators
        "pyronear-biobio_florida_0_2025-05-30T18-42-38",  # azimuth 0
    ],
)
def test_valid_folder_names(valid_folder, tmp_path, name):
    """Rename the valid_folder fixture to check name patterns."""
    from pyro_dataset.ingest import _FOLDER_RE
    assert _FOLDER_RE.match(name), f"Expected '{name}' to match"


@pytest.mark.parametrize(
    "name",
    [
        "sdis83_brison_200",                        # missing timestamp
        "sdis83_brison_200_2024-01-15",             # incomplete timestamp (no time)
        "sdis83 brison_200_2024-01-15T10-30-00",    # space in name
        "sdis83_brison_200_2024-1-15T10-30-00",     # month not zero-padded
        "adf_avinyonet_999_2023_05_23T17_18_31",    # underscore date → invalid
        "awf-axis_armstronglookout1_2023_06_01T10_35_04",  # missing azimuth → invalid
    ],
)
def test_invalid_folder_names(name):
    from pyro_dataset.ingest import _FOLDER_RE
    assert not _FOLDER_RE.match(name), f"Expected '{name}' to not match"


def test_invalid_azimuth_range(tmp_path):
    name = "sdis83_brison_500_2024-01-15T10-30-00"
    folder = tmp_path / name
    (folder / "images").mkdir(parents=True)
    (folder / "labels").mkdir(parents=True)
    for ts in ["T10-30-00", "T10-31-00", "T10-32-00"]:
        (folder / "images" / f"sdis83_brison_500_2024-01-15{ts}.jpg").touch()
    (folder / "labels" / "sdis83_brison_500_2024-01-15T10-30-00.txt").write_text(
        "0 0.5 0.5 0.1 0.1"
    )
    (folder / "labels" / "sdis83_brison_500_2024-01-15T10-31-00.txt").write_text(
        "0 0.4 0.4 0.1 0.1"
    )
    (folder / "labels" / "sdis83_brison_500_2024-01-15T10-32-00.txt").touch()
    from pyro_dataset.ingest import validate_sequence_folder
    result = validate_sequence_folder(folder)
    assert result.has_naming_issues
    assert any("azimuth" in issue for issue in result.naming_issues)


def test_missing_images_dir(valid_folder):
    import shutil
    shutil.rmtree(valid_folder / "images")
    result = validate_sequence_folder(valid_folder)
    assert result.has_structural_issues
    assert any("images" in e for e in result.structural_issues)


def test_missing_labels_dir(valid_folder):
    import shutil
    shutil.rmtree(valid_folder / "labels")
    result = validate_sequence_folder(valid_folder)
    assert result.has_structural_issues
    assert any("labels" in e for e in result.structural_issues)


def test_bad_image_filename(valid_folder):
    (valid_folder / "images" / "bad_name.jpg").touch()
    result = validate_sequence_folder(valid_folder)
    assert result.has_naming_issues
    assert any("image" in e for e in result.naming_issues)


def test_zero_labels(valid_folder):
    for f in (valid_folder / "labels").iterdir():
        f.unlink()
    result = validate_sequence_folder(valid_folder)
    assert result.has_structural_issues
    assert any("label" in e for e in result.structural_issues)


def test_one_label_many_images(valid_folder):
    # valid_folder has 3 images — 1 label should be rejected
    for f in (valid_folder / "labels").iterdir():
        f.unlink()
    (valid_folder / "labels" / "sdis83_brison_200_2024-01-15T10-30-00.txt").write_text(
        "0 0.5 0.5 0.1 0.1"
    )
    result = validate_sequence_folder(valid_folder)
    assert result.has_structural_issues
    assert any("label" in e for e in result.structural_issues)


def test_one_label_one_image_ok(tmp_path):
    # 1 image + 1 label should be accepted
    name = "sdis83_brison_200_2024-01-15T10-30-00"
    folder = tmp_path / name
    (folder / "images").mkdir(parents=True)
    (folder / "labels").mkdir(parents=True)
    (folder / "images" / f"{name}.jpg").touch()
    (folder / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1")
    result = validate_sequence_folder(folder)
    assert not result.has_structural_issues


def test_empty_labels_not_counted(valid_folder):
    for f in (valid_folder / "labels").iterdir():
        f.unlink()
    (valid_folder / "labels" / "sdis83_brison_200_2024-01-15T10-30-00.txt").write_text(
        "0 0.5 0.5 0.1 0.1"
    )
    (valid_folder / "labels" / "sdis83_brison_200_2024-01-15T10-31-00.txt").touch()
    result = validate_sequence_folder(valid_folder)
    assert result.has_structural_issues


def test_naming_issue_is_not_structural(valid_folder):
    """A naming warning alone should not count as a structural error."""
    (valid_folder / "images" / "bad_name.jpg").touch()
    result = validate_sequence_folder(valid_folder)
    assert result.has_naming_issues
    assert not result.has_structural_issues
