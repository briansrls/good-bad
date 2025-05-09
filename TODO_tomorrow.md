# Debugging and Cleanup Plan for Tomorrow

## Debugging Plan (Focus on `train_diagnostic_v2.py`):

The goal is to find out why `train_diagnostic_v2.py` (using your `build_split` and `MLP(1)`) fails when `minimal_pytorch_test.py` (with identical core logic) succeeds. We'll assume `data/generator.py` is still in its "clean" state (`LABEL_NOISE=0.0`, `CONTR_RATE=0.0`) and `models/gb_model.py` has the single-layer `MLP`.

1.  **Verify Data Tensors (`X_data`, `y_data` in `train_diagnostic_v2.py`):**
    *   **Action**: In `train_diagnostic_v2.py`, immediately after `y_data = y_data_full[:,0].unsqueeze(1)`, print:
        *   `X_data.dtype`, `X_data.requires_grad`
        *   `y_data.dtype`, `y_data.requires_grad`
    *   **Expected**: `X_data.dtype` should be `torch.float32`. `y_data.dtype` should be `torch.float32`. `X_data.requires_grad` should be `False`. `y_data.requires_grad` should be `False`.
    *   **Purpose**: Rule out incorrect data types or `requires_grad` flags.

2.  **Inspect Model Parameters and Gradients (Interactive Debugger):**
    *   **Action**: Set a breakpoint in `train_diagnostic_v2.py` *inside the training loop*, right after `loss.backward()`.
    *   **Inspect**:
        *   `model.final_layer.weight.requires_grad` (should be True)
        *   `model.final_layer.bias.requires_grad` (should be True)
        *   `model.final_layer.weight.grad`: Is it `None`? Is it all zeros? Magnitudes?
        *   `model.final_layer.bias.grad`: Same questions.
        *   `loss.item()`
        *   `outputs.requires_grad` (should be True)
        *   `loss.requires_grad` (should be True)
    *   **Step over `optimizer.step()`**:
        *   Inspect `model.final_layer.weight.data` and `model.final_layer.bias.data` *before and after* `optimizer.step()`. Are they changing sensibly?
    *   **Purpose**: Determine if gradients are computed correctly and if the optimizer applies updates as expected.

3.  **Compare `MLP` Instantiation and Usage**: 
    *   **Action**: Double-check that `MLP(out_dim=1)` in `train_diagnostic_v2.py` truly uses the simple `nn.Linear(10,1)` + `Sigmoid` structure.
    *   **Purpose**: Ensure the correct simple model is used.

4.  **If Gradients are None or Zero**:
    *   **Action**: Work backward. If `weight.grad` is `None`, trace the computation graph connectivity: `loss` to `outputs`, `outputs` to `model.final_layer.weight`.
    *   **Purpose**: Find breaks in the computation graph.

## Cleanup Plan (After resolving diagnostic issues):

1.  **`data/generator.py`**:
    *   Restore `LABEL_NOISE` (e.g., to `0.15`).
    *   Restore `CONTR_RATE` (e.g., to `0.20`).
    *   Remove `import builtins` and revert to `min/max` if original "max not defined" error is resolved. Otherwise, keep `builtins`.
    *   Remove explicit `dtype=torch.float32` from `_encode`'s `torch.tensor()` call.

2.  **`models/gb_model.py`**:
    *   Restore `MLP` to intended architecture (e.g., with hidden layers, BatchNorm).
    *   Ensure `make_scalar()` and `make_gb()` return `MLP(out_dim=2)`.
    *   Restore `gb_loss` function to its desired operational state (e.g., with BCE components, margin, and/or confidence terms as decided).
    *   Remove temporary diagnostic modifications from `MLP.forward` (like logit printing).

3.  **`train.py`**:
    *   Uncomment `scalar_model` training loop and optimizer.
    *   Restore `console.log` to print both `L_scalar` and `L_gb`.
    *   Remove all diagnostic weight/gradient/output printing.

4.  **`experiments.cfg`**:
    *   Set `lr` to a sensible starting value (e.g., `0.001`).
    *   Set `gb_loss_lambda_margin` and `gb_loss_lambda_confidence` to desired experimental values.

5.  **Delete/Keep Diagnostic Scripts**:
    *   `minimal_pytorch_test.py` (keep as sanity check).
    *   `train_diagnostic_v2.py` (delete after use or keep if modified to be a working simple baseline).
    *   `visualize_targets.py` (keep, it's useful).

**Main Goal for Tomorrow's Debugging**: Get `train_diagnostic_v2.py` (simplest MLP(1), clean data) to show a loss near zero. 