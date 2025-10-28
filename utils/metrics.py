import tensorflow as tf


class PSNRMetric(tf.keras.metrics.Metric):
    """Peak Signal-to-Noise Ratio (PSNR) metric"""
    def __init__(self, name="psnr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean = tf.keras.metrics.Mean(name="psnr_mean")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.image.psnr(y_true, y_pred, max_val=1.0)
        self.mean.update_state(values, sample_weight=sample_weight)

    def result(self):
        return self.mean.result()

    def reset_state(self):
        self.mean.reset_state()


class SSIMMetric(tf.keras.metrics.Metric):
    """Structural Similarity Index (SSIM) metric"""
    def __init__(self, name="ssim", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean = tf.keras.metrics.Mean(name="ssim_mean")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.image.ssim(y_true, y_pred, max_val=1.0)
        self.mean.update_state(values, sample_weight=sample_weight)

    def result(self):
        return self.mean.result()

    def reset_state(self):
        self.mean.reset_state()


class PerceptualLoss:
    """VGG16-based perceptual loss for image reconstruction tasks"""
    
    def __init__(self):
        # Load VGG16 and extract intermediate layers
        vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        vgg.trainable = False
        
        # Use multiple layers for richer feature extraction
        layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
        outputs = [vgg.get_layer(name).output for name in layer_names]
        
        self.feature_extractor = tf.keras.Model(
            inputs=vgg.input,
            outputs=outputs,
            name='vgg16_feature_extractor'
        )
        self.feature_extractor.trainable = False
        
        # Weights for different layers (earlier layers = more weight)
        self.layer_weights = [1.0, 0.75, 0.5, 0.25]
    
    def preprocess(self, x):
        """Preprocess images for VGG16 (expects [0, 1] input)"""
        # Convert from [0, 1] to [0, 255]
        x = x * 255.0
        # Apply VGG16 preprocessing (RGB to BGR and mean subtraction)
        return tf.keras.applications.vgg16.preprocess_input(x)
    
    def __call__(self, y_true, y_pred):
        """Calculate perceptual loss between true and predicted images"""
        # Preprocess images
        y_true_processed = self.preprocess(y_true)
        y_pred_processed = self.preprocess(y_pred)
        
        # Extract features
        true_features = self.feature_extractor(y_true_processed)
        pred_features = self.feature_extractor(y_pred_processed)
        
        # Calculate weighted feature loss
        total_loss = 0.0
        for true_feat, pred_feat, weight in zip(true_features, pred_features, self.layer_weights):
            # Normalize by number of elements to make loss scale-invariant
            n_elements = tf.cast(tf.size(true_feat), tf.float32)
            loss = tf.reduce_sum(tf.square(true_feat - pred_feat)) / n_elements
            total_loss += weight * loss
        
        return total_loss


class StyleLoss:
    """Gram matrix based style loss for texture consistency"""
    
    def __init__(self):
        # Use same VGG16 setup as perceptual loss
        vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        vgg.trainable = False
        
        layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3']
        outputs = [vgg.get_layer(name).output for name in layer_names]
        
        self.feature_extractor = tf.keras.Model(
            inputs=vgg.input,
            outputs=outputs
        )
        self.feature_extractor.trainable = False
    
    def preprocess(self, x):
        """Preprocess images for VGG16"""
        x = x * 255.0
        return tf.keras.applications.vgg16.preprocess_input(x)
    
    def gram_matrix(self, features):
        """Calculate gram matrix for style representation"""
        # features shape: (batch, height, width, channels)
        batch, height, width, channels = tf.shape(features)[0], tf.shape(features)[1], tf.shape(features)[2], tf.shape(features)[3]
        
        # Reshape to (batch, height*width, channels)
        features_reshaped = tf.reshape(features, [batch, height * width, channels])
        
        # Calculate gram matrix: (batch, channels, channels)
        gram = tf.matmul(features_reshaped, features_reshaped, transpose_a=True)
        
        # Normalize by number of elements
        return gram / tf.cast(height * width, tf.float32)
    
    def __call__(self, y_true, y_pred):
        """Calculate style loss between true and predicted images"""
        y_true_processed = self.preprocess(y_true)
        y_pred_processed = self.preprocess(y_pred)
        
        true_features = self.feature_extractor(y_true_processed)
        pred_features = self.feature_extractor(y_pred_processed)
        
        total_loss = 0.0
        for true_feat, pred_feat in zip(true_features, pred_features):
            true_gram = self.gram_matrix(true_feat)
            pred_gram = self.gram_matrix(pred_feat)
            
            loss = tf.reduce_mean(tf.square(true_gram - pred_gram))
            total_loss += loss
        
        return total_loss


def total_variation_loss(y_pred):
    """
    Total variation loss for smoothness.
    Encourages spatial smoothness in the reconstructed image.
    """
    # Calculate differences in horizontal and vertical directions
    pixel_dif1 = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    pixel_dif2 = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    
    # Sum of absolute differences
    sum_axis = [1, 2, 3]
    tv_loss = tf.reduce_mean(tf.abs(pixel_dif1), axis=sum_axis) + \
              tf.reduce_mean(tf.abs(pixel_dif2), axis=sum_axis)
    
    return tf.reduce_mean(tv_loss)


# Global instances (created once to avoid rebuilding VGG models)
_perceptual_loss = None
_style_loss = None


def get_perceptual_loss():
    """Get or create perceptual loss instance"""
    global _perceptual_loss
    if _perceptual_loss is None:
        _perceptual_loss = PerceptualLoss()
    return _perceptual_loss


def get_style_loss():
    """Get or create style loss instance"""
    global _style_loss
    if _style_loss is None:
        _style_loss = StyleLoss()
    return _style_loss


def combined_loss(y_true, y_pred, 
                  mae_weight=1.0,
                  ssim_weight=1.0, 
                  perceptual_weight=2.0,
                  style_weight=0.5,
                  tv_weight=0.001):
    """
    Enhanced combined loss for image inpainting/reconstruction.
    
    This loss function is specifically designed for tasks with heavy masking (90%).
    
    Args:
        y_true: Ground truth images (batch, height, width, channels)
        y_pred: Predicted images (batch, height, width, channels)
        mae_weight: Weight for MAE loss (pixel-level accuracy)
        ssim_weight: Weight for SSIM loss (structural similarity)
        perceptual_weight: Weight for perceptual loss (high for inpainting)
        style_weight: Weight for style/texture loss
        tv_weight: Weight for total variation loss (smoothness)
    
    Returns:
        Combined weighted loss value
    """
    
    # 1. MAE Loss (pixel-level accuracy)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # 2. SSIM Loss (structural similarity)
    ssim_loss = tf.reduce_mean(1.0 - tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    # 3. Perceptual Loss (feature-level similarity - CRITICAL for inpainting!)
    perceptual = get_perceptual_loss()(y_true, y_pred)
    
    # 4. Style Loss (texture consistency)
    style = get_style_loss()(y_true, y_pred)
    
    # 5. Total Variation Loss (smoothness)
    tv = total_variation_loss(y_pred)
    
    # Combine all losses
    total_loss = (mae_weight * mae + 
                  ssim_weight * ssim_loss + 
                  perceptual_weight * perceptual +
                  style_weight * style +
                  tv_weight * tv)
    
    return total_loss


def combined_loss_light(y_true, y_pred):
    """
    Lighter version with reduced perceptual weight for faster training.
    Use this if training is too slow or memory-constrained.
    """
    return combined_loss(
        y_true, y_pred,
        mae_weight=1.0,
        ssim_weight=1.0,
        perceptual_weight=1.0,  # Reduced from 2.0
        style_weight=0.25,      # Reduced from 0.5
        tv_weight=0.001
    )


class MetricsCalculator:
    """Calculate and store metrics for model comparison"""

    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate all metrics for a model (offline use, not Keras training)"""
        psnr = float(tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0)))
        ssim = float(tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)))
        mae = float(tf.reduce_mean(tf.abs(y_true - y_pred)))
        mse = float(tf.reduce_mean(tf.square(y_true - y_pred)))

        self.metrics[model_name] = {
            "PSNR": psnr,
            "SSIM": ssim,
            "MAE": mae,
            "MSE": mse
        }

        return self.metrics[model_name]

    def compare_models(self):
        """Generate comparison between models"""
        if not self.metrics:
            return "No metrics available for comparison"

        comparison = "Model Comparison Results:\n" + "="*50 + "\n"

        # Create comparison table
        models = list(self.metrics.keys())
        metrics_names = list(self.metrics[models[0]].keys())

        # Header
        comparison += f"{'Model':<15}"
        for metric in metrics_names:
            comparison += f"{metric:<12}"
        comparison += "\n" + "-"*60 + "\n"

        # Data rows
        for model in models:
            comparison += f"{model:<15}"
            for metric in metrics_names:
                value = self.metrics[model][metric]
                comparison += f"{value:<12.4f}"
            comparison += "\n"

        # Best performing model for each metric
        comparison += "\n" + "Best Performance:\n" + "-"*30 + "\n"
        for metric in metrics_names:
            if metric in ["PSNR", "SSIM"]:  # Higher is better
                best_model = max(models, key=lambda x: self.metrics[x][metric])
                best_value = self.metrics[best_model][metric]
            else:  # Lower is better for MAE, MSE
                best_model = min(models, key=lambda x: self.metrics[x][metric])
                best_value = self.metrics[best_model][metric]

            comparison += f"{metric}: {best_model} ({best_value:.4f})\n"

        return comparison