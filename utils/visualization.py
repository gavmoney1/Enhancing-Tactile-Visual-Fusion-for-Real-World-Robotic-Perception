import os
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Dict, Any
import seaborn as sns
from PIL import Image

class Visualizer:
    """Handle visualization of results and comparisons"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.out_dir = config['data']['out_dir']
        self.num_samples = config['evaluation'].get('num_visualization_samples', 16)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def save_sample_predictions(self, model, test_ds, model_name: str):
        """Save sample predictions for visual inspection"""
        print(f"Generating predictions for {model_name}...")
        
        predictions_dir = os.path.join(self.out_dir, model_name, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Get one batch from test dataset
        for masked_batch, target_batch in test_ds.take(1):
            pred_batch = model.predict(masked_batch, verbose=0)
            
            # Clip predictions to [0, 1]
            pred_batch = tf.clip_by_value(pred_batch, 0.0, 1.0)
            
            # Save individual images
            self._save_image_batch(masked_batch, predictions_dir, "masked")
            self._save_image_batch(pred_batch, predictions_dir, "predicted")
            self._save_image_batch(target_batch, predictions_dir, "target")
            
            # Create comparison grid
            self._create_comparison_grid(
                masked_batch, pred_batch, target_batch, 
                predictions_dir, model_name
            )
            break
    
    def _save_image_batch(self, images, output_dir: str, prefix: str):
        images = tf.clip_by_value(images, 0.0, 1.0)
        images = tf.image.convert_image_dtype(images, tf.uint8)

        num_save = min(self.num_samples, images.shape[0])

        for i in range(num_save):
            filename = f"{prefix}_{i:03d}.png"
            filepath = os.path.join(output_dir, filename)

            # Convert tensor to numpy array
            arr = images[i].numpy()
            Image.fromarray(arr).save(filepath)
    
    def _create_comparison_grid(self, masked, predicted, target, output_dir: str, model_name: str):
        """Create a comparison grid showing masked, predicted, and target images"""
        num_display = min(8, masked.shape[0])  # Display up to 8 samples
        
        fig, axes = plt.subplots(3, num_display, figsize=(num_display * 2, 6))
        fig.suptitle(f'{model_name} - Image Reconstruction Comparison', fontsize=16)
        
        for i in range(num_display):
            # Masked image
            axes[0, i].imshow(masked[i])
            axes[0, i].set_title('Masked', fontsize=10)
            axes[0, i].axis('off')
            
            # Predicted image
            axes[1, i].imshow(predicted[i])
            axes[1, i].set_title('Predicted', fontsize=10)
            axes[1, i].axis('off')
            
            # Target image
            axes[2, i].imshow(target[i])
            axes[2, i].set_title('Target', fontsize=10)
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self, histories: Dict[str, Dict], save_path: str):
        """Plot training histories for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History Comparison', fontsize=16)
        
        metrics = ['loss', 'psnr', 'ssim']
        metric_titles = ['Loss', 'PSNR', 'SSIM']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            for model_name, history in histories.items():
                if metric in history:
                    epochs = range(1, len(history[metric]) + 1)
                    ax.plot(epochs, history[metric], label=f'{model_name} (train)', linewidth=2)
                    
                    val_metric = f'val_{metric}'
                    if val_metric in history:
                        ax.plot(epochs, history[val_metric], 
                               label=f'{model_name} (val)', linestyle='--', linewidth=2)
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to: {save_path}")
    
    def create_metrics_comparison_plot(self, metrics: Dict[str, Dict], save_path: str):
        """Create bar plot comparing metrics across models"""
        models = list(metrics.keys())
        metric_names = list(metrics[models[0]].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        for idx, metric in enumerate(metric_names):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            values = [metrics[model][metric] for model in models]
            bars = ax.bar(models, values, alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(metric, fontsize=14)
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Highlight best performing model
            if metric in ['PSNR', 'SSIM']:  # Higher is better
                best_idx = values.index(max(values))
            else:  # Lower is better for MAE, MSE
                best_idx = values.index(min(values))
            
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics comparison plot saved to: {save_path}")
    
    def create_summary_report(self, results: Dict[str, Any], save_path: str):
        """Create a comprehensive HTML summary report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Testbed Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .best {{ background-color: #ffffcc; font-weight: bold; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Transformer Models Training Results</h1>
            
            <div class="summary">
                <h2>Experiment Summary</h2>
                <p><strong>Image Size:</strong> {self.config['data']['img_size']}x{self.config['data']['img_size']}</p>
                <p><strong>Mask Ratio:</strong> {self.config['data']['mask_ratio'] * 100}%</p>
                <p><strong>Batch Size:</strong> {self.config['training']['batch_size']}</p>
                <p><strong>Epochs:</strong> {self.config['training']['epochs']}</p>
                <p><strong>Models Trained:</strong> {', '.join(self.config['models_to_train'])}</p>
            </div>
        """
        
        # Add metrics table
        if 'metrics' in results:
            html_content += """
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Model</th><th>PSNR</th><th>SSIM</th><th>MAE</th><th>MSE</th></tr>
            """
            
            for model, metrics in results['metrics'].items():
                if model != "detr":
                    html_content += f"""
                    <tr>
                        <td><strong>{model}</strong></td>
                        <td>{metrics['PSNR']:.4f}</td>
                        <td>{metrics['SSIM']:.4f}</td>
                        <td>{metrics['MAE']:.4f}</td>
                        <td>{metrics['MSE']:.4f}</td>
                    </tr>
                    """
                else:
                    html_content += f"""
                    <tr>
                        <td><strong>{model}</strong></td>
                        <td><strong>{metrics['PSNR']:.4f}</strong></td>
                        <td><strong>{metrics['SSIM']:.4f}</strong></td>
                        <td><strong>{metrics['MAE']:.4f}</strong></td>
                        <td><strong>{metrics['MSE']:.4f}</strong></td>
                    </tr>
                    """
            
            html_content += "</table>"
        
        # Add training times
        if 'training_times' in results:
            html_content += """
            <h2>Training Times</h2>
            <table>
                <tr><th>Model</th><th>Training Time (seconds)</th><th>Training Time (minutes)</th></tr>
            """
            
            for model, time_seconds in results['training_times'].items():
                time_minutes = time_seconds / 60
                if model != "detr":
                    html_content += f"""
                    <tr>
                        <td><strong>{model}</strong></td>
                        <td>{time_seconds:.2f}</td>
                        <td>{time_minutes:.2f}</td>
                    </tr>
                    """
                else:
                    html_content += f"""
                    <tr>
                        <td><strong>{model}</strong></td>
                        <td><strong>{time_seconds:.2f}</strong></td>
                        <td><strong>{time_minutes:.2f}</strong></td>
                    </tr>
                    """
        # Add metrics table
        if 'metrics' in results:
            html_content += """
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Model</th><th>PSNR</th><th>SSIM</th><th>MAE</th><th>MSE</th></tr>
            """
            
            for model, metrics in results['metrics'].items():
                html_content += f"""
                <tr>
                    <td><strong>{model}</strong></td>
                    <td>{metrics['PSNR']:.4f}</td>
                    <td>{metrics['SSIM']:.4f}</td>
                    <td>{metrics['MAE']:.4f}</td>
                    <td>{metrics['MSE']:.4f}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        # Add training times
        if 'training_times' in results:
            html_content += """
            <h2>Training Times</h2>
            <table>
                <tr><th>Model</th><th>Training Time (seconds)</th><th>Training Time (minutes)</th></tr>
            """
            
            for model, time_seconds in results['training_times'].items():
                time_minutes = time_seconds / 60
                html_content += f"""
                <tr>
                    <td><strong>{model}</strong></td>
                    <td>{time_seconds:.2f}</td>
                    <td>{time_minutes:.2f}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        html_content += """
            <h2>Visual Results</h2>
            <p>Check the individual model folders for detailed predictions and comparisons.</p>
            
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Summary report saved to: {save_path}")
