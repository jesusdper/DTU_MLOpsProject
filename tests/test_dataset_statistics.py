from src.object_detection.dataset_statistics import count_files_in_dir, dataset_statistics
from pathlib import Path

def test_count_files_in_dir(tmp_path):
    """
    Test the files are correctly counted with the function.
    """
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    (dir_path / "file1.txt").touch()
    (dir_path / "file2.txt").touch()

    assert count_files_in_dir(dir_path) == 2

def test_datset_statistics(tmp_path, capsys):
    """
    Test the dataset statistics are correctly computed.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train").mkdir()
    (data_dir / "train" / "images").mkdir()
    (data_dir / "train" / "labels").mkdir()
    (data_dir / "val").mkdir()
    (data_dir / "val" / "images").mkdir()
    (data_dir / "val" / "labels").mkdir()
    (data_dir / "test").mkdir()
    (data_dir / "test" / "images").mkdir()

    dataset_statistics(datadir=data_dir)

    captured = capsys.readouterr()
    assert "Statistics for Train split" in captured.out
    assert "Statistics for Val split" in captured.out
    assert "Statistics for Test split" in captured.out
    assert "Number of images: 0" in captured.out
    assert "Number of labels: 0" in captured.out
    assert "Number of images: 0" in captured.out
    assert "Number of labels: 0" in captured.out
    assert "Number of images: 0" in captured.out
    assert "Dataset statistics visualization saved as 'dataset_statistics.png'" in captured.out
    assert Path("./dataset_statistics.png").exists()
    assert Path("./dataset_statistics.png").is_file()
