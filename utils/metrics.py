import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np


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


class LPIPSMetric(tf.keras.metrics.Metric):
    """Simplified perceptual similarity metric using VGG features"""
    def __init__(self, name="lpips", **kwargs):
        super().__init__(name=name, **kwargs)
        self.vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        self.vgg.trainable = False
        self.mean = tf.keras.metrics.Mean(name="lpips_mean")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Rescale to [0, 255] for VGG
        features_true = self.vgg(y_true * 255.0)
        features_pred = self.vgg(y_pred * 255.0)
        values = tf.reduce_mean(tf.square(features_true - features_pred), axis=list(range(1, len(features_true.shape))))
        self.mean.update_state(values, sample_weight=sample_weight)

    def result(self):
        return self.mean.result()

    def reset_state(self):
        self.mean.reset_state()


def combined_loss(y_true, y_pred, mae_weight=0.8, ssim_weight=0.2):
    """Combined MAE and SSIM loss"""
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_loss = tf.reduce_mean(1.0 - tf.image.ssim(y_true, y_pred, max_val=1.0))
    return mae_weight * mae + ssim_weight * ssim_loss


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

    def calculate_classification_metrics(self, y_true, y_pred, model_name: str):
        """Calculate classification metrics"""
        # Convert predictions to class labels
        if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
            y_pred_classes = tf.argmax(y_pred, axis=-1).numpy()
        else:
            y_pred_classes = y_pred.numpy()
        
        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            y_true_classes = tf.argmax(y_true, axis=-1).numpy()
        else:
            y_true_classes = y_true.numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Classification report
        class_report = classification_report(y_true_classes, y_pred_classes, output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        self.metrics[model_name] = metrics
        return metrics
