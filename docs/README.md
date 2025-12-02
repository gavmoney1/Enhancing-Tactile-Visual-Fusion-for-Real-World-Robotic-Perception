# Enhancing-Tactile-Visual-Fusion-for-Real-World-Robotic-Perception
# Transformer Image Reconstruction Training Testbed

A modular framework for training and comparing different transformer architectures on masked image reconstruction tasks.

## Features

- **Multiple Transformer Architectures**: ViT, Swin Transformer, DETR, MAE Up, Conventional Autoencoder
- **Memory Management**: Optimized for laptop training with memory monitoring
- **Comprehensive Evaluation**: PSNR, SSIM, MAE, MSE metrics
- **Visual Comparisons**: Automated generation of prediction visualizations !TODO!
- **Modular Design**: Easy to add new models and configurations

## How to install software

0. Prerequisites
To deploy this project in your own environment, you will need:

- **Operating system**: Linux (recommended) or Windows with Python support
- **Python**: 3.10 (required)
- **Git**: for cloning the repository
- **Hardware**:
  - CPU-only is supported, but **GPU + CUDA** is strongly recommended for training performance
1. **Clone the repository**
This can be done in many ways, but most easily by going to your command terminal and inputting:

-git clone https://github.com/gavmoney1/Enhancing-Tactile-Visual-Fusion-for-Real-World-Robotic-Perception.git
cd Enhancing-Tactile-Visual-Fusion-for-Real-World-Robotic-Perception

3. **Install Dependencies**
We recommend creating a virtual environment, although this is not required.
Run the following in your root directory
```bash
pip install -r requirements.txt
```
3. **Configure your dataset**
Download your desired dataset, and move it into the repository
Make sure that your dataset is a folder of images. Your dataset's parent folder should not have subfolders.
Next, configure utils/mask.py to point to your dataset, and run it. This creates masked images for your training

3. **Update Configuration**
Update `configs/base_config.yaml` to include your image paths for both your original and masked datasets, and desired training parameters.
-data.orig_root: path to your original images
-data.mask_root: path to your masked images

Masked images must have the same name as their unmasked counterpart (and are created to have the same name in utils/mask.py)

4. **Run Training**
```bash
# Train all models
python main.py

# Train specific models
python main.py --models vit swin

# Use custom config
python main.py --config configs/my_config.yaml
```

## How to use each completed feature

1. **Unified Training Configuration System**
All training settings are controlled through YAML configuration files
under `configs/`. These include:
- dataset paths
- image preprocessing settings
- hyperparameters (batch size, epochs, learning rate)
- enabled models
- memory options
- evaluation settings

To use:
1. Open `configs/base_config.yaml`
2. Edit:
   - training parameters (epochs, lr, batch size)
   - list of models to train

3. Run training:
```bash
python main.py --config configs/base_config.yaml

main.py loads the config, which passes it into the trainer (trainers/trainer.py). Data loader reads dataset paths from config, while model builders receive specific instructions from their configs.

2. **Multi-Model Training Pipeline**

Train one or more architectures through the same interface:
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


Produces:
trained model checkpoints
experiment folders for each model
training logs, reconstructions, metrics

data comes from datasets/data_loader.py
models come from models/*.py
training loop executed in trainers/trainer.py
metrics computed via utils/metrics.py

3. **Evaluation & Metrics**

The framework evaluates models using:
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better) 
- **MAE**: Mean Absolute Error (lower is better)
- **MSE**: Mean Squared Error (lower is better)

trainers/trainer.py collects predictions and ground truths, which calls metric functions in utils/metrics.py
Results found in each model’s output directory and in experiment summary HTML report

4. **Visualization & Experiment Report**
Feature Description
After training, the framework generates:
-reconstruction image grids
-comparison charts
-training history plots
-an HTML summary report with all results

HTML summary report is found in experiments/ folder.
Explore each model's folder to see their visualizations.


5. **Memory Optimization**

For laptop training, the framework includes:
- **Mixed Precision**: FP16 training to reduce memory usage
- **Gradient Accumulation**: Simulate larger batch sizes
- **Memory Monitoring**: Track usage and trigger cleanup
- **Sequential Training**: Train models one at a time
- **Dynamic Memory Growth**: GPU memory allocation as needed

These options can be enabled in configs/base_config.yaml:
-enable_mixed_precision: true
-gradient_accumulation_steps: 4
-sequential_training: true



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
