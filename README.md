# Enhancing-Tactile-Visual-Fusion-for-Real-World-Robotic-Perception
# Transformer Image Reconstruction Training Testbed

A modular framework for training and comparing different transformer architectures on masked image reconstruction tasks.

## Features

- **Multiple Transformer Architectures**: ViT, Swin Transformer, DETR, MAE Up, Conventional Autoencoder
- **Memory Management**: Optimized for laptop training with memory monitoring
- **Comprehensive Evaluation**: PSNR, SSIM, MAE, MSE metrics
- **Visual Comparisons**: Automated generation of prediction visualizations !TODO!
- **Modular Design**: Easy to add new models and configurations

## Quick Start

1. **Install Dependencies**
**Please ensure you are using python 3.10.** 
We recommend creating a virtual environment, although this is not required.
Run the following in your root directory
```bash
pip install -r requirements.txt
```

2. **Update Configuration**
Update `configs/base_config.yaml` to include your image paths and desired training parameters.

Masked images must have the same name as their unmasked counterpart.
To create masked images, see utils/mask.py.

3. **Run Training**
```bash
# Train all models
python main.py

# Train specific models
python main.py --models vit swin

# Use custom config
python main.py --config configs/my_config.yaml
```

## Configuration

The `configs/base_config.yaml` contains all training parameters:
- Dataset paths and preprocessing settings
- Training hyperparameters
- Memory management options
- Model selection (defaults to all)
- Evaluation settings

## Output

Results are saved to the specified output directory:
- Individual model folders with checkpoints and logs
- Prediction visualizations
- Training history plots  
- Metrics comparison charts
- HTML summary report

## Memory Management

The framework includes several memory optimization features:
- Mixed precision training
- Memory usage monitoring
- Gradient accumulation
- Sequential model training
- Automatic cleanup between models

## Adding New Models

To add a new transformer architecture:

1. Create `models/your_model.py` inheriting from `BaseTransformerModel`
2. Implement the `build_model()` method
3. Add model to `MODEL_REGISTRY` in `main.py`
4. Create `configs/your_model_config.yaml` (optional)

Example:
```python
from .base_model import BaseTransformerModel

class YourModel(BaseTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        
    def build_model(self):
        # Implement your model architecture
        pass
```

## Project Structure

```
training_testbed/
├── configs/                 # Configuration files
│   ├── base_config.yaml    # Main configuration
│   ├── vit_config.yaml     # ViT-specific settings
│   ├── swin_config.yaml    # Swin-specific settings
│   ├── detr_config.yaml    # DETR-specific settings
│   ├── mae_up_config.yaml  # MAE-UP-specific settings
│   └── conv_ae_config.yaml # ConvAE-specific settings
├── datasets/               # Data loading utilities
│   └── data_loader.py      # Dataset creation and preprocessing
├── models/                 # Model implementations
│   ├── base_model.py       # Abstract base class
│   ├── vit_model.py        # Vision Transformer
│   ├── swin_model.py       # Swin Transformer
│   ├── detr_model.py       # DETR-style model
│   ├── mae_up_model.py     # Masked Autoencoder with UNet-style decoder
│   └── conv_ae_model.py    # Convolutional Autoencoder
├── trainers/               # Training logic
│   └── trainer.py          # Model trainer with callbacks
├── utils/                  # Utility functions
│   ├── metrics.py          # Evaluation metrics
│   ├── visualization.py    # Plotting and visualization
│   └── memory_utils.py     # Memory management
├── main.py                 # Main training script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Model Architectures

### Vision Transformer (ViT)
- Standard ViT implementation for image reconstruction
- Patch-based processing with positional embeddings
- Multi-head self-attention with MLP blocks

### Swin Transformer
- Hierarchical vision transformer with shifted windows
- Window-based attention for computational efficiency
- Multi-stage architecture with patch merging

### DETR-style Model
- CNN backbone with transformer decoder
- Object queries for patch reconstruction
- Cross-attention between queries and image features

### MAE-UP Model
- Masked Autoencoder with UNet-style Progressive decoder
- Self-supervised pretraining with random mask patches
- Progressive upsampling decoder for detailed reconstruction
- Skip connections between encoder and decoder stages

### Convolutional Autoencoder
- Traditional convolutional architecture baseline
- Encoder: Series of Conv2D + BatchNorm + ReLU blocks
- Decoder: Transposed convolutions for upsampling
- Bottleneck representation for compression

## Metrics

The framework evaluates models using:
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better) 
- **MAE**: Mean Absolute Error (lower is better)
- **MSE**: Mean Squared Error (lower is better)

## Memory Optimization

For laptop training, the framework includes:
- **Mixed Precision**: FP16 training to reduce memory usage !TODO!
- **Gradient Accumulation**: Simulate larger batch sizes
- **Memory Monitoring**: Track usage and trigger cleanup
- **Sequential Training**: Train models one at a time
- **Dynamic Memory Growth**: GPU memory allocation as needed

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size` in config
- Enable `enable_mixed_precision: true`
- Increase `gradient_accumulation_steps`
- Reduce `img_size` or model dimensions

### Slow Training
- Increase `batch_size` if memory allows
- Reduce `num_layers` or `embed_dim`
- Use fewer `epochs` for initial testing

## License
???