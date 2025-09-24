# Enhancing-Tactile-Visual-Fusion-for-Real-World-Robotic-Perception
# Transformer Image Reconstruction Training Testbed

A modular framework for training and comparing different transformer architectures on masked image reconstruction tasks.

## Features

- **Multiple Transformer Architectures**: ViT, Swin Transformer, DETR
- **Memory Management**: Optimized for laptop training with memory monitoring
- **Comprehensive Evaluation**: PSNR, SSIM, MAE, MSE metrics
- **Visual Comparisons**: Automated generation of prediction visualizations
- **Modular Design**: Easy to add new models and configurations

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Update Configuration**
Edit `configs/base_config.yaml` with your dataset paths and training parameters.

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
- Model selection
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
│   └── detr_config.yaml    # DETR-specific settings
├── datasets/               # Data loading utilities
│   └── data_loader.py      # Dataset creation and preprocessing
├── models/                 # Model implementations
│   ├── base_model.py       # Abstract base class
│   ├── vit_model.py        # Vision Transformer
│   ├── swin_model.py       # Swin Transformer
│   └── detr_model.py       # DETR-style model
├── trainers/               # Training logic
│   └── trainer.py          # Model trainer with callbacks
├── utils/                  # Utility functions
│   ├── metrics.py          # Evaluation metrics
│   ├── visualization.py    # Plotting and visualization
│   └── memory_utils.py     # Memory management
├── main.py                 # Main training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
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

## Metrics

The framework evaluates models using:
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better) 
- **MAE**: Mean Absolute Error (lower is better)
- **MSE**: Mean Squared Error (lower is better)

## Memory Optimization

For laptop training, the framework includes:
- **Mixed Precision**: FP16 training to reduce memory usage
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

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)

## License
???