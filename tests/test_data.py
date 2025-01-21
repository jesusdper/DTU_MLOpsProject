from pathlib import Path
from src.object_detection.data import (
    download_voc_dataset,
    convert_voc_to_yolo,
    preprocess_data,
)
import src.object_detection.data as data
import os
import numpy as np
import json
from unittest.mock import patch, mock_open, MagicMock
import pytest  # type: ignore
from PIL import Image
from xml.etree.ElementTree import Element, SubElement, ElementTree
import tarfile
import logging

def test_download_voc_dataset(mocker, tmp_path):
    """
    Test downloading the VOC dataset is correctly working.
    """
    mock_urlretrieve = mocker.patch("urllib.request.urlretrieve")
    mock_tarfile = mocker.patch("tarfile.open")

    # Dataset has not been downloaded yet
    download_voc_dataset(tmp_path)
    mock_urlretrieve.assert_called_once()
    mock_tarfile.assert_called_once()

    # Dataset already downloaded
    mock_urlretrieve.reset_mock()
    mock_tarfile.reset_mock()
    (tmp_path / "VOC2012.tar").touch()
    download_voc_dataset(tmp_path)
    mock_urlretrieve.assert_not_called()
    mock_tarfile.assert_called_once()


def test_download_voc_logs(caplog, tmp_path):
    """
    Test the logger is correctly logging the download status.
    """
    with caplog.at_level(logging.INFO):
        data.download_voc_dataset(tmp_path)
        assert "Downloading PASCAL VOC 2012 dataset..." in caplog.text or "PASCAL VOC 2012 dataset already downloaded." in caplog.text
        assert "Extracting dataset..." in caplog.text
        assert "Dataset extracted." in caplog.text


def test_download_voc_extract(tmp_path):
    """
    Test the dataset is correctly extracted.
    """
    with patch("urllib.request.urlretrieve") as mock_urlretrieve, patch("tarfile.open") as mock_tar:
        mock_tar.return_value.__enter__.return_value = MagicMock(spec=tarfile.TarFile)
        download_voc_dataset(tmp_path)
        mock_urlretrieve.assert_called_once()
        mock_tar.assert_called_once()


def test_convert_voc_to_yolo():
    """
    Test the conversion from VOC to YOLO format is working correctly.
    """
    xml_content = """
    <annotation>
        <object>
            <name>dog</name>
            <bndbox>
                <xmin>50</xmin><ymin>100</ymin><xmax>150</xmax><ymax>200</ymax>
            </bndbox>
        </object>
    </annotation>
    """
    with patch("builtins.open", mock_open(read_data=xml_content)), patch("pathlib.Path.exists", return_value=True):
        result = convert_voc_to_yolo(Path("mocked_file.xml"), 400, 400)

    class_id, x_center, y_center, width, height = result[0]
    CLASS_TO_ID = json.loads(os.environ["CLASS_TO_ID"])
    assert class_id == CLASS_TO_ID["dog"]
    assert x_center == 0.25
    assert y_center == 0.375
    assert width == 0.25
    assert height == 0.25


def test_convert_voc_to_yolo_valid_data(tmp_path):
    """
    Test the conversion from VOC to YOLO format is working correctly.
    """
    annotation = Element("annotation")
    obj = SubElement(annotation, "object")
    SubElement(obj, "name").text = "dog"
    bndbox = SubElement(obj, "bndbox")
    for tag, value in [("xmin", "50"), ("ymin", "50"), ("xmax", "150"), ("ymax", "200")]:
        SubElement(bndbox, tag).text = value

    xml_path = tmp_path / "sample.xml"
    ElementTree(annotation).write(xml_path)

    result = convert_voc_to_yolo(xml_path, 300, 300)
    expected = [[10, 0.333, 0.417, 0.333, 0.5]]

    for res_row, exp_row in zip(result, expected):
        for res_val, exp_val in zip(res_row, exp_row):
            if isinstance(exp_val, float):
                assert np.round(res_val, 2) == np.round(exp_val, 2)
            else:
                assert res_val == exp_val

    
@pytest.fixture
def tmp_raw(tmp_path):
    """
    Fixture for creating a raw dataset directory.
    """
    raw_dir = tmp_path / "raw"
    voc_dir = raw_dir / "VOCdevkit/VOC2012"
    images_dir = voc_dir / "JPEGImages"
    annotations_dir = voc_dir / "Annotations"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    for i in range(170):
        img = Image.new("RGB", (256, 256), color=(i % 256, (i * 2) % 256, (i * 3) % 256))
        img.save(images_dir / f"image{i}.jpg")

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
    Fixture for creating a processed dataset directory.
    """
    processed_dir = tmp_path / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    return processed_dir


def test_preprocess_data(tmp_raw, tmp_processed):
    """
    Test the data preprocessing is working correctly.
    """
    splits = {"train": 100, "val": 50, "test": 20}
    preprocess_data(tmp_raw, tmp_processed, splits, (256, 256))

    for split_name, count in splits.items():
        split_images_dir = tmp_processed / split_name / "images"
        split_labels_dir = tmp_processed / split_name / "labels"

        assert split_images_dir.exists()
        assert split_labels_dir.exists()
        assert len(list(split_images_dir.glob("*.jpg"))) == count
        assert len(list(split_labels_dir.glob("*.txt"))) == count


def test_preprocess_data_missing_files(tmp_raw, tmp_processed):
    """
    Test the data preprocessing is working correctly with missing files.
    """
    splits = {"train": 100, "val": 50, "test": 20}

    images = list((tmp_raw / "VOCdevkit/VOC2012/JPEGImages").iterdir())
    annotations = list((tmp_raw / "VOCdevkit/VOC2012/Annotations").iterdir())
    os.remove(images[0])
    os.remove(annotations[1])

    preprocess_data(tmp_raw, tmp_processed, splits, (256, 256))

    for split_name, count in splits.items():
        split_images_dir = tmp_processed / split_name / "images"
        split_labels_dir = tmp_processed / split_name / "labels"

        assert len(list(split_images_dir.glob("*.jpg"))) <= count
        assert len(list(split_labels_dir.glob("*.txt"))) <= count

def test_load_data():
    """
    Test the load_data function is working correctly.
    """
    data.load_data()
    