import os
import tensorflow as tf
import numpy as np
import cv2
from typing import Optional, Dict, Any
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from utils.metrics import MetricsCalculator

class InferenceEngine:
    """
    Inference engine for pretrained image reconstruction models (.keras).
    Loads a trained model, runs predictions on images or folders,
    computes metrics, and creates visual comparisons.
    """

    def __init__(self, model_path: str, config_path: str, img_size: int = 224, enable_visualization: bool = False):
        """
        Initialize the inference engine.
        """
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.img_size = img_size
        self.enable_visualization = enable_visualization

        # Get model name
        base_name = os.path.basename(self.model_path)
        self.model_name = base_name.replace("_model.keras", "")

        # Directories from config file
        self.input_target_folder = self.config["data"]["orig_root"]
        self.input_mask_folder = self.config["data"]["mask_root"]
        base_out = self.config["data"]["out_dir"]
        self.output_folder = os.path.join(base_out, self.model_name)
        os.makedirs(self.output_folder, exist_ok=True)

        # Configure GPU + load model
        self._configure_gpu()
        self.model = self._load_model()

        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()

    # Initialization and setup
    def _configure_gpu(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Configured {len(gpus)} GPU(s) for memory growth")
            except RuntimeError as e:
                print(f"WARNING: GPU memory configuration error: {e}")
        else:
            print("No GPU found, using CPU")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load cnofiguration from YAML file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"Loaded configuration from: {config_path}")
        return config

    def _load_model(self):
        """Load the trained model with custom objects if needed."""
        model_name = os.path.basename(os.path.dirname(self.model_path))
        
        # Import custom layers/objects based on model type
        custom_objects = {}
        
        if 'vit' in model_name.lower():
            from models.vit_model import PatchEmbedding, TransformerBlock
            custom_objects = {
                'PatchEmbedding': PatchEmbedding,
                'TransformerBlock': TransformerBlock
            }
            print(f"Registered custom layers for ViT model: {list(custom_objects.keys())}")
        elif 'conv' in model_name.lower() or 'autoencoder' in model_name.lower():
            from models.conv_autoencoder_model import EncoderBlock, DecoderBlock
            custom_objects = {
                'EncoderBlock': EncoderBlock,
                'DecoderBlock': DecoderBlock
            }
            print(f"Registered custom layers for Conv Autoencoder: {list(custom_objects.keys())}")
        else:
            print(f"WARNING: No custom layers registered for {model_name}. Attempting to load normally...")
        
        try:
            if custom_objects:
                model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects=custom_objects,
                    compile=False
                )
            else:
                model = tf.keras.models.load_model(
                    self.model_path,
                    compile=False
                )
            print(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Prepare image for model input
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        return img

    def _postprocess_image(self, pred: np.ndarray) -> np.ndarray:
        """
        Convert prediction to uint8 image for saving
        """
        return np.clip(pred * 255, 0, 255).astype(np.uint8)

    # Visualizations
    def _create_comparison_grid(self, masked_dir, predicted_dir, target_dir, output_dir, model_name, num_samples=5):
        """
        Create a comparison grid showing matched masked, predicted, and target images.
        There should be five images for each (15 total)
        """
        print(f"Creating comparison grid for {model_name}...")

        # Collect filenames
        masked_files = {os.path.basename(f): os.path.join(masked_dir, f) for f in os.listdir(masked_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        predicted_files = {os.path.basename(f): os.path.join(predicted_dir, f) for f in os.listdir(predicted_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

        # Match filenames across masked and predicted
        matching = sorted(set(masked_files.keys()) & set(predicted_files.keys()))
        if not matching:
            print("WARNING: No matching filenames between masked and predicted folders.")
            return

        matching = matching[:num_samples]
        masked_imgs, predicted_imgs, target_imgs = [], [], []

        for fname in matching:
            masked_path = masked_files[fname]
            predicted_path = predicted_files[fname]

            # Derive possible target path
            base_name = os.path.basename(fname)
            video_id = "_".join(base_name.split("_")[:-1])
            simple_name = base_name.split("_")[-1]

            # Extract video identifier from filename or subdirectory.
            # Handles:
            #     (1) matched_frames/20220503_100937/0000001.jpg -> 20220503_100937
            #     or 
            #     (2) matched_frames/20220503_100937_0000001.jpg -> 20220503_100937
    
            candidate1 = os.path.join(target_dir, video_id, simple_name)
            candidate2 = os.path.join(target_dir, base_name)
            target_path = candidate1 if os.path.exists(candidate1) else (
                candidate2 if os.path.exists(candidate2) else None
            )

            if target_path is None:
                print(f"WARNING: No target found for {fname}")
                continue

            def load_rgb(path):
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size))
                return img.astype(np.float32) / 255.0

            masked_imgs.append(load_rgb(masked_path))
            predicted_imgs.append(load_rgb(predicted_path))
            target_imgs.append(load_rgb(target_path))

        if not masked_imgs:
            print("WARNING: No valid triplets found.")
            return

        # Generate comparison grid
        plt.style.use('default')
        sns.set_palette("husl")
        num_display = len(masked_imgs)
        fig, axes = plt.subplots(3, num_display, figsize=(num_display * 2, 6))
        fig.suptitle(f"{model_name} - Reconstruction Comparison", fontsize=16)

        for i in range(num_display):
            axes[0, i].imshow(masked_imgs[i]); axes[0, i].set_title('Masked', fontsize=10); axes[0, i].axis('off')
            axes[1, i].imshow(predicted_imgs[i]); axes[1, i].set_title('Predicted', fontsize=10); axes[1, i].axis('off')
            axes[2, i].imshow(target_imgs[i]); axes[2, i].set_title('Target', fontsize=10); axes[2, i].axis('off')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{model_name}_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison grid saved to: {save_path}")

    # Main function for making predictions
    def make_predictions(self) -> Dict[str, Any]:
        input_mask_folder = self.input_mask_folder
        output_folder = self.output_folder
        os.makedirs(output_folder, exist_ok=True)

        image_files = [
            f for f in os.listdir(input_mask_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        num_images = len(image_files)
        print(f"Found {num_images} masked images in {input_mask_folder}")

        metrics_results = {}
        all_preds, all_targets = [], []

        for idx, filename in enumerate(image_files, start=1):
            input_path = os.path.join(input_mask_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                img = self._preprocess_image(input_path)
                pred = self.model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                pred = self._postprocess_image(pred)
                cv2.imwrite(output_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

                # get metrics
                all_preds.append(pred.astype(np.float32) / 255.0)
                all_targets.append(img.astype(np.float32))

                if idx % 10 == 0 or idx == num_images:
                    print(f"Processed {idx}/{num_images} images")

            except Exception as e:
                print(f"WARNING: Skipping {filename}: {e}")

        # Compute metrics
        if all_preds:
            preds_tensor = tf.convert_to_tensor(np.stack(all_preds), dtype=tf.float32)
            targets_tensor = tf.convert_to_tensor(np.stack(all_targets), dtype=tf.float32)
            metrics_results = self.metrics_calculator.calculate_metrics(
                targets_tensor, preds_tensor, "inference"
            )
            print(f"\nInference Metrics: {metrics_results}")

        # Create comparison grid matched by filename
        if self.enable_visualization:
            self._create_comparison_grid(
                masked_dir=self.input_mask_folder,
                predicted_dir=self.output_folder,
                target_dir=self.input_target_folder,
                output_dir=self.output_folder,
                model_name=self.model_name,
                num_samples=5
            )

        print(f"Predictions saved to {output_folder}")
        return metrics_results
