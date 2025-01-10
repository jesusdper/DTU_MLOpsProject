import torch
from pathlib import Path
import typer


def train_model(processed_dir: Path, output_dir: Path):
    """
    Train the model using preprocessed data.

    Args:
        processed_dir (Path): Path to the directory containing preprocessed data.
        output_dir (Path): Path to the directory where outputs will be saved.
    """
    typer.echo(f"Loading preprocessed data from {processed_dir}...")

    # Load training data
    train_images_path = processed_dir / "train_images.pt"
    train_target_path = processed_dir / "train_target.pt"

    if not train_images_path.exists() or not train_target_path.exists():
        typer.echo(f"Error: Missing processed files: {train_images_path} or {train_target_path}")
        return

    train_images = torch.load(train_images_path)
    train_target = torch.load(train_target_path)

    # Convert lists to tensors (if they're not already)
    train_images = torch.stack(train_images) if isinstance(train_images, list) else train_images
    train_target = torch.stack(train_target) if isinstance(train_target, list) else train_target

    typer.echo(f"Loaded training data: images ({train_images.shape}), targets ({train_target.shape})")

    # Placeholder for model training logic
    typer.echo("Training the model...")

    # Example output: Save a dummy model file
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "trained_model.pt"
    torch.save({"dummy_model": True}, model_path)

    typer.echo(f"Model saved to {model_path}")


if __name__ == "__main__":
    # Hardcoded paths for convenience
    processed_dir = Path(r"C:\Users\jdiaz\Desktop\DTU_MLOpsProject\data\processed")
    output_dir = Path(r"C:\Users\jdiaz\Desktop\DTU_MLOpsProject\models")

    train_model(processed_dir, output_dir)
