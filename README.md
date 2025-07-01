# EDG: Enhanced Discriminability via Gradients and Texture-Preserving Augmentation

## Architecture Overview

```
x → AugNet(x) → x1, x2 → Backbone → f1, f2 → hook → g1, g2
                                  ↓
InfoNCE(g1,g2) + 0.5×CE(cls(f1),label) + 0.5×CE(cls(f2),label)
```

## File Structure

```
src/
├── models/
│   └── contrastive_model.py       # Main contrastive learning model
├── network/
│   └── augnet.py                  # Learnable augmentation network
├── utils/
│   ├── loss.py                    # MomentHoMDivLoss
│   └── contrastive_loss.py        # InfoNCE loss implementation
├── train.py                       # Main training script
├── inference.py                   # Inference functions
├── visualize_augnet.py            # AugNet visualization tool
└── generate_augnet_visualization.py  # Script to generate visualizations from checkpoints
```

## Usage

### 1. Environment Setup

Make sure to install the required dependencies:

```bash
pip install torch torchvision timm tqdm pyyaml
```

### 2. Configuration File

Edit `config_contrastive.yaml` according to your dataset:

```yaml
# Key parameters
num_classes: 10          # Number of classes in your dataset
backbone: "resnet50"     # Backbone network
dataset_name: "cotton80" # Dataset name

# Training parameters
augnet_lr: 1e-4         # AugNet learning rate
model_lr: 1e-3          # Main model learning rate
lambda_moment: 1.0      # Moment loss weight
gamma_div: 1.0          # Divergence loss weight
enable_infonce: true    # Enable/disable InfoNCE contrastive loss
```

#### InfoNCE Loss Configuration

The `enable_infonce` parameter allows you to control whether the InfoNCE contrastive loss is computed:

- **`enable_infonce: true`** (default): Full contrastive learning with InfoNCE loss + classification loss
- **`enable_infonce: false`**: Only classification loss is used (standard supervised learning)

Example configurations are provided:
- `config_contrastive.yaml`: Full contrastive learning (InfoNCE enabled)
- `config_no_infonce.yaml`: Classification-only training (InfoNCE disabled)

### 3. Train the Model

```bash
cd src
python train_contrastive.py
```

The training process includes two separate backpropagations:

1. AugNet is trained with MomentHoMDivLoss.
2. The main model is trained with InfoNCE + classification loss.

### 4. Inference

```bash
# Predict class
python inference.py --checkpoint checkpoints/checkpoint_epoch_50.pth \
                   --config config_contrastive.yaml \
                   --image path/to/image.jpg \
                   --mode predict

# Extract features
python inference.py --checkpoint checkpoints/checkpoint_epoch_50.pth \
                   --config config_contrastive.yaml \
                   --image path/to/image.jpg \
                   --mode features

# Generate augmented samples
python inference.py --checkpoint checkpoints/checkpoint_epoch_50.pth \
                   --config config_contrastive.yaml \
                   --image path/to/image.jpg \
                   --mode augment \
                   --num_augs 10
```

## AugNet Visualization

After training, the model can automatically generate visualizations showing the effect of the AugNet on several sample images. This helps in understanding how the learned augmentation transforms the input images.

### Generating Visualizations

To generate visualizations from a trained model checkpoint:

```bash
# Using the batch script (Windows)
visualize_augnet.bat checkpoints/checkpoint_epoch_100.pth --num-examples 4

# Or directly with Python
python src/generate_augnet_visualization.py --checkpoint checkpoints/checkpoint_epoch_100.pth --output augnet_visualization.png
```

The visualization will show the original images alongside their AugNet transformations, helping to understand the learned augmentation space.

## Key Features

### 1. Modular Design

- **ContrastiveModel**: Main contrastive learning model.
- **ContrastiveTrainer**: Handles decoupled training logic.
- **InfoNCELoss**: Contrastive loss for gradient features.
- **GradientHook**: Hook mechanism to capture gradient features.

### 2. Learnable Augmentation

AugNet uses TinyViT blocks to achieve learnable data augmentation, trained with MomentHoMDivLoss:

- Higher-order moment matching (HoM): matches skewness and kurtosis.
- KL divergence (Div): maintains distribution consistency.

### 3. Dual Optimization

```python
# Step 1: Train AugNet
augnet_loss = moment_loss(x1, x2)
augnet_loss.backward()
augnet_optimizer.step()

# Step 2: Train main model
main_loss = infonce_loss + classification_loss
main_loss.backward()
model_optimizer.step()
```

### 4. Gradient Feature Contrastive Learning

InfoNCE loss is applied to gradient features instead of direct features, providing richer representation learning.

### 5. Flexible Loss Configuration

The model supports flexible loss configuration through the `enable_infonce` parameter:

- **Contrastive Learning Mode**: Both InfoNCE contrastive loss and classification loss
- **Supervised Learning Mode**: Only classification loss (InfoNCE disabled)

This allows you to compare the effectiveness of contrastive learning versus standard supervised learning on your dataset.

## Dataset Integration

Currently supports the following datasets (implement the corresponding dataset class as needed):

- Cotton80
- IP102
- SoyLocal

To add a new dataset:

1. Implement your dataset class in `dataset/`
2. Update the `load_existing_dataset` function in `utils/data_utils.py`
3. Set `dataset_name` in the config file

## Training Monitoring

The training process logs the following metrics:

- AugNet Loss (MomentHoMDivLoss)
- Contrastive Loss (InfoNCE)
- Classification Loss (CrossEntropy)
- Validation Accuracy

Logs are saved in `training.log`, and checkpoints are saved in the `checkpoints/` directory.

## Notes

1. **Memory Usage**: Gradient hooks increase memory usage; it is recommended to use a smaller batch size.
2. **Convergence Speed**: AugNet and the main model may converge at different rates; adjust learning rates as needed.
3. **Dataset Adaptation**: Ensure your dataset class returns (image, label) format.

## Troubleshooting

1. **CUDA Out of Memory**: Reduce batch_size or image_size.
2. **Gradient Hook Error**: Make sure to clear the hook cache at the start of each epoch.
3. **Data Loading Error**: Check dataset paths and formats.

## References

This implementation combines:

- SimCLR contrastive learning framework
- The concept of learnable data augmentation
- Distribution matching of higher-order statistics
- Representation learning with gradient features
