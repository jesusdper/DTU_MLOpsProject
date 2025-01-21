from src.object_detection.utils import collate_fn
import torch  # type: ignore

def test_collate_fn():
    """
    Test the collate function is correctly working.
    """
    image1 = torch.rand((3, 224, 224))
    target1 = {"boxes": torch.tensor([[10, 10, 100, 100]]), "labels": torch.tensor([1])}
    sample1 = (image1, target1)

    image2 = torch.rand((3, 224, 224))
    target2 = {"boxes": torch.tensor([[20, 20, 120, 120]]), "labels": torch.tensor([2])}
    sample2 = (image2, target2)

    batch = [sample1, sample2]
    images, targets = collate_fn(batch)

    assert images.shape == (2, 3, 224, 224)
    assert len(targets) == 2
    assert torch.equal(targets[0]["boxes"], torch.tensor([[10, 10, 100, 100]]))
    assert torch.equal(targets[1]["boxes"], torch.tensor([[20, 20, 120, 120]]))
    assert torch.equal(targets[0]["labels"], torch.tensor([1]))
    assert torch.equal(targets[1]["labels"], torch.tensor([2]))
