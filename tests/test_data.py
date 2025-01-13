from tests import _PATH_DATA
import xml.etree.ElementTree as ET
from pathlib import Path
from src.object_detection.data import download_voc_dataset, convert_voc_to_yolo
import os
import json
from unittest.mock import patch, mock_open

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

    assert len(result) == 1

    class_id, x_center, y_center, width, height = result[0]
    assert [len(result[i]) == 5 for i in range(len(result))]

    CLASS_TO_ID = json.loads(os.environ["CLASS_TO_ID"])
    assert class_id == CLASS_TO_ID["dog"]

    # (50 + 150) / 2 / 400 = 0.25
    assert x_center == 0.25
    # (100 + 200) / 2 / 400 = 0.375
    assert y_center == 0.375
    # (150 - 50) / 400 = 0.25
    assert width == 0.25
    # (200 - 100) / 400 = 0.25
    assert height == 0.25


def test_preprocess_data(tmp_path: Path):
    """
    Test data preprocessing is correctly carried out.
    """

    raw_dir = tmp_path / "raw"
    preprocessed_dir = tmp_path / "processed"
    splits = {"train": 100, "val": 50, "test": 20}
