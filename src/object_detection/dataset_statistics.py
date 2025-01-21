from pathlib import Path
import typer  # type: ignore
import matplotlib.pyplot as plt


def count_files_in_dir(directory: Path) -> int:
    """Count the number of files in a directory."""
    return len([f for f in directory.iterdir() if f.is_file()])

def dataset_statistics(datadir: str = "../../data/processed") -> None:
    """Compute dataset statistics for the PASCAL VOC dataset."""
    data_path = Path(datadir)
    splits = ["train", "val", "test"]

    statistics = {}
    for split in splits:
        split_path = data_path / split
        if split == "test":
            images_dir = split_path / "images"
            labels_dir = None
        else:
            images_dir = split_path / "images"
            labels_dir = split_path / "labels"

        images_count = count_files_in_dir(images_dir)
        labels_count = count_files_in_dir(labels_dir) if labels_dir else 0

        statistics[split] = {
            "images": images_count,
            "labels": labels_count,
        }

        print(f"\nStatistics for {split.capitalize()} split:")
        print(f"  Number of images: {images_count}")
        if labels_dir:
            print(f"  Number of labels: {labels_count}")

    # Visualization: Bar chart for images and labels count
    bar_labels = []
    images_count = []
    labels_count = []

    for split, stats in statistics.items():
        bar_labels.append(split.capitalize())
        images_count.append(stats["images"])
        labels_count.append(stats["labels"])

    x = range(len(bar_labels))
    width = 0.4

    plt.bar(x, images_count, width, label="Images", color="blue")
    plt.bar([p + width for p in x], labels_count, width, label="Labels", color="orange")

    plt.xlabel("Dataset Split")
    plt.ylabel("Count")
    plt.title("Dataset Split Statistics")
    plt.xticks([p + width / 2 for p in x], bar_labels)
    plt.legend()
    plt.tight_layout()

    plt.savefig("dataset_statistics.png")
    print("\nDataset statistics visualization saved as 'dataset_statistics.png'")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)
