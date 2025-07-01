# InfoNCE Loss Enable/Disable Feature

## Summary

I've successfully added an option to enable and disable the InfoNCE (contrastive) loss in your EDG contrastive learning framework. This allows you to switch between full contrastive learning mode and supervised learning mode.

## Changes Made

### 1. Configuration Files

- **Modified `config_contrastive.yaml`**: Added `enable_infonce: true` parameter
- **Created `config_no_infonce.yaml`**: Example configuration with InfoNCE disabled

### 2. Model Architecture (`models/contrastive_model.py`)

- Added `enable_infonce` parameter to `ContrastiveModel.__init__()`
- Modified `compute_losses()` method to conditionally compute InfoNCE loss
- Updated `forward_backbone()` to only register gradient hooks when InfoNCE is enabled
- When disabled, contrastive loss returns `torch.tensor(0.0)` with gradients

### 3. Training Script (`train.py`)

- Updated model instantiation to pass `enable_infonce` parameter from config
- Added logging to show InfoNCE status at training start

### 4. Documentation

- Updated `README.md` with InfoNCE configuration documentation
- Added examples showing both modes of operation

### 5. Test Files

- **`test_infonce_config.py`**: Automated test to verify functionality
- **`example_infonce_usage.py`**: Example usage demonstrating both configurations

## Usage

### Enable InfoNCE (Contrastive Learning)
```yaml
enable_infonce: true  # Default behavior
```
- Loss = InfoNCE Loss + Classification Loss
- Full contrastive learning with gradient features
- Projection head and gradient hooks are active

### Disable InfoNCE (Supervised Learning)
```yaml
enable_infonce: false
```
- Loss = Classification Loss only
- Standard supervised learning
- Projection head still exists but InfoNCE loss = 0
- No gradient hooks registered

## Benefits

1. **Ablation Studies**: Compare contrastive vs supervised learning on your dataset
2. **Debugging**: Isolate classification performance without contrastive complexity
3. **Computational Efficiency**: Reduce memory usage by disabling gradient hooks
4. **Flexibility**: Easy switching between training modes without code changes

## Verified Functionality

✅ InfoNCE loss is correctly computed when enabled
✅ InfoNCE loss returns 0.0 when disabled  
✅ Classification loss works in both modes
✅ Gradient hooks only registered when needed
✅ Configuration properly passed through training pipeline
✅ Logging shows current InfoNCE status

## Example Configurations

### Full Contrastive Learning
```bash
python train.py  # Uses config_contrastive.yaml (enable_infonce: true)
```

### Supervised Learning Only
```bash
# Edit config to use config_no_infonce.yaml or set enable_infonce: false
python train.py
```

The feature is backward compatible - existing configurations will default to `enable_infonce: true` if not specified.
