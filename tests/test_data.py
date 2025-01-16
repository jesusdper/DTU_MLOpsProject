from pathlib import Path
from src.object_detection.data import (
    convert_voc_to_yolo,
    preprocess_data,
)
import os
import json
from unittest.mock import patch, mock_open
import pytest  # type: ignore
from PIL import Image

# def test_download_voc_dataset():


def test_convert_voc_to_yolo_valid_xml():
    """
    Test that the function returns the correct values for a valid XML file.
    """
    # Create and example annotation string
    xml_content = """
    <annotation>
        <object>
            <name>dog</name>
            <bndbox>
                <xmin>50</xmin>
                <ymin>100</ymin>
                <xmax>150</xmax>
                <ymax>200</ymax>
            </bndbox>
        </object>
    </annotation>
    """

    # Mock opening to avoid creating a new file
    with (
        patch("builtins.open", mock_open(read_data=xml_content)),
        patch("pathlib.Path.exists", return_value=True),
    ):
        with patch.object(Path, "read_text", return_value=xml_content):
            # Call the function with img size that gives easier numbers for the test
            result = convert_voc_to_yolo(Path("mocked_file.xml"), 400, 400)

    assert len(result) == 1, "The number of annotations is not correct"

    class_id, x_center, y_center, width, height = result[0]
    assert [
        len(result[i]) == 5 for i in range(len(result))
    ], "The number of elements in the annotation is not correct"

    CLASS_TO_ID = json.loads(os.environ["CLASS_TO_ID"])
    assert class_id == CLASS_TO_ID["dog"]

    # (50 + 150) / 2 / 400 = 0.25
    assert x_center == 0.25, "x_center is not correct"
    # (100 + 200) / 2 / 400 = 0.375
    assert y_center == 0.375, "y_center is not correct"
    # (150 - 50) / 400 = 0.25
    assert width == 0.25, "width is not correct"
    # (200 - 100) / 400 = 0.25
    assert height == 0.25, "height is not correct"


@pytest.fixture
def tmp_raw(tmp_path):
    """
    Create a temporary raw directory with a mock VOC dataset structure.
    """
    raw_dir = tmp_path / "raw"
    voc_dir = raw_dir / "VOCdevkit/VOC2012"
    images_dir = voc_dir / "JPEGImages"
    annotations_dir = voc_dir / "Annotations"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Add mock images and annotations
    for i in range(170):
        # Create a small mock image
        img = Image.new(
            "RGB", (256, 256), color=(i % 256, (i * 2) % 256, (i * 3) % 256)
        )
        img.save(images_dir / f"image{i}.jpg")

        # Create dummy annotation file with only one class
        annotation_content = f"""
        <annotation>
            <object>
                <name>dog</name>
                <bndbox>
                    <xmin>{10 + i % 20}</xmin>
                    <ymin>{15 + i % 20}</ymin>
                    <xmax>{100 + i % 20}</xmax>
                    <ymax>{120 + i % 20}</ymax>
                </bndbox>
            </object>
        </annotation>
        """

        with open(annotations_dir / f"image{i}.xml", "w") as f:
            f.write(annotation_content)

    return raw_dir


@pytest.fixture
def tmp_processed(tmp_path):
    """
    Create a temporary processed directory.
    """
    processed_dir = tmp_path / "processed"
    os.makedirs(processed_dir, exist_ok=True)

    return processed_dir


def test_preprocess_data(tmp_raw: Path, tmp_processed: Path):
    """
    Test that the function correctly creates the preprocess data directories.
    """
    splits = {"train": 100, "val": 50, "test": 20}

    # Run preprocessing function
    preprocess_data(tmp_raw, tmp_processed, splits, (256, 256))

    for split_name in splits.keys():
        split_images_dir = tmp_processed / split_name / "images"
        split_labels_dir = tmp_processed / split_name / "labels"

        # Check split directories exist
        assert split_images_dir.exists(), f"{split_images_dir} does not exist"
        assert split_labels_dir.exists(), f"{split_labels_dir} does not exist"

        split_images = list(split_images_dir.glob("*.jpg"))
        split_labels = list(split_labels_dir.glob("*.txt"))
        print(f"{split_name} images: {len(split_images)}")
        print(f"{split_name} labels: {len(split_labels)}")

        # Check number of images and labels in directories
        assert (
            len(split_images) == splits[split_name]
        ), f"Mismatch in {split_name} images"
        assert (
            len(split_labels) == splits[split_name]
        ), f"Mismatch in {split_name} labels"
