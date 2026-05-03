"""Quick smoke test for the Gradient Reversal Layer.

Runs in <5 seconds on CPU or GPU. Verifies:
1. Forward pass is identity (output == input).
2. Backward pass reverses and scales the gradient by -lambda.

Run from repo root:
    python -m src.training.smoke_test_grl
"""

import torch

from src.models.grl import GradientReversalLayer, gradient_reverse


def test_grl_forward_identity() -> None:
    """Forward must be identity: output equals input."""
    x = torch.randn(4, 8)
    y = gradient_reverse(x, lambda_=0.5)
    assert torch.allclose(x, y), "GRL forward should be identity"
    print("[PASS] forward is identity")


def test_grl_backward_negates_gradient() -> None:
    """Backward must produce gradient = -lambda * upstream_grad."""
    lambda_ = 0.7
    x = torch.randn(4, 8, requires_grad=True)
    y = gradient_reverse(x, lambda_=lambda_)
    # imagine an upstream loss that gives gradient = ones to y
    y.sum().backward()
    expected_grad = -lambda_ * torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad), (
        f"GRL backward gradient mismatch: got {x.grad[0, 0]}, expected {expected_grad[0, 0]}"
    )
    print(f"[PASS] backward returns -{lambda_} * upstream_grad")


def test_grl_module_set_lambda() -> None:
    """The nn.Module wrapper allows updating lambda dynamically."""
    layer = GradientReversalLayer(lambda_=0.0)
    x = torch.randn(2, 4, requires_grad=True)

    # initially lambda=0 -> gradient should be zero
    y = layer(x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.zeros_like(x)), (
        "With lambda=0, gradient should be zero"
    )
    print("[PASS] lambda=0 produces zero gradient")

    # update lambda and re-run
    x.grad.zero_()
    layer.set_lambda(1.5)
    y = layer(x)
    y.sum().backward()
    expected = -1.5 * torch.ones_like(x)
    assert torch.allclose(x.grad, expected), "Updated lambda not applied"
    print("[PASS] set_lambda(1.5) applied dynamically")


def test_grl_on_cuda_if_available() -> None:
    """Sanity check on GPU."""
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available, skipping GPU test")
        return
    x = torch.randn(4, 8, requires_grad=True, device="cuda")
    y = gradient_reverse(x, lambda_=1.0)
    y.sum().backward()
    expected = -torch.ones_like(x)
    assert torch.allclose(x.grad, expected), "GPU backward mismatch"
    print("[PASS] works on CUDA")


if __name__ == "__main__":
    print("Running GRL smoke tests...\n")
    test_grl_forward_identity()
    test_grl_backward_negates_gradient()
    test_grl_module_set_lambda()
    test_grl_on_cuda_if_available()
    print("\nAll GRL smoke tests passed.")