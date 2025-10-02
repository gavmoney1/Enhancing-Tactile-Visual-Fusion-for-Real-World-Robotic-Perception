import os
# Suppress TF INFO and WARNING logs, has to do with tf2 vs tf1 backwards compatibility stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set Qt to offscreen (headless), only for generating plots
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured {len(gpus)} GPU(s) for memory growth")
    except RuntimeError as e:
        print(f"GPU memory configuration error: {e}")
else:
    print("No GPU found, using CPU")
import sys
import copy
import yaml
import argparse
import traceback
from typing import Dict, Any
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets.data_loader import DataLoader
from models.vit_model import ViTModel
from models.swin_model import SwinModel
from models.detr_model import DETRModel
from models.conv_autoencoder_model import ConvolutionalAutoencoderModel
from models.mae_up_model import MAEUpModel
from trainers.trainer import ModelTrainer
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer

MODEL_REGISTRY = {
    'vit': ViTModel,
    'swin': SwinModel,
    'detr': DETRModel,
    'conv_autoencoder': ConvolutionalAutoencoderModel,
    'mae_up': MAEUpModel
}

class TrainingTestbed:
    """Main training testbed"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.results = {
            'metrics': {},
            'training_times': {},
            'histories': {}
        }
        
        self.data_loader = DataLoader(self.config)
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(self.config)
        os.makedirs(self.config['data']['out_dir'], exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loaded configuration from: {config_path}")
        return config
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete training experiment"""
        print("="*60)
        print("TRANSFORMER IMAGE RECONSTRUCTION TRAINING TESTBED")
        print("="*60)
        
        try:
            print("\n1. Loading datasets...")
            train_ds, val_ds, test_ds = self.data_loader.create_datasets()
            
            models_to_train = self.config['models_to_train']
            print(f"\n2. Models to train: {models_to_train}")

            trained_models = {}
            for model_name in models_to_train:
                if model_name not in MODEL_REGISTRY:
                    print(f"Warning: Unknown model '{model_name}', skipping...")
                    continue
                
                print(f"\n{'='*40}")
                print(f"Training {model_name.upper()}")
                print(f"{'='*40}")
                
                try:
                    result = self._train_single_model(
                        model_name, train_ds, val_ds, test_ds
                    )
                    trained_models[model_name] = result
                    
                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")
                    print(traceback.format_exc())
                    continue
            
            print(f"\n{'='*40}")
            print("GENERATING RESULTS AND COMPARISONS")
            print(f"{'='*40}")
            
            self._generate_final_results(trained_models)
            
            return self.results
            
        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            print(traceback.format_exc())
            return {}
    
    def _train_single_model(self, model_name: str, train_ds, val_ds, test_ds) -> Dict[str, Any]:
        """Train a single model"""
        # Make deep copy of base config
        model_config = copy.deepcopy(self.config)
        
        # Load model-specific config if it exists
        model_config_path = os.path.join('configs', f'{model_name}_config.yaml')
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                model_specific = yaml.safe_load(f)
                # Merge configs while preserving base config structure
                for key, value in model_specific.items():
                    if key in model_config and isinstance(model_config[key], dict):
                        model_config[key].update(value)
                    else:
                        model_config[key] = value
    
        trainer = ModelTrainer(model_config, model_name)
        
        # Build model
        model_class = MODEL_REGISTRY[model_name]
        model_builder = model_class(model_config)
        trainer.build_and_compile_model(model_builder)
        
        # Train
        training_result = trainer.train(train_ds, val_ds)
        
        # Evaluate
        eval_results = trainer.evaluate(test_ds)
        
        # Generate metrics
        print(f"Calculating detailed metrics for {model_name}...")
        self._calculate_detailed_metrics(trainer.model, test_ds, model_name)
        
        # Generate visualizations
        if self.config['evaluation']['save_predictions']:
            self.visualizer.save_sample_predictions(trainer.model, test_ds, model_name)
        
        # Store results
        self.results['training_times'][model_name] = training_result['training_time']
        self.results['histories'][model_name] = training_result['history']
        
        return {
            'trainer': trainer,
            'model': trainer.model,
            'training_result': training_result,
            'eval_results': eval_results
        }
    
    def _calculate_detailed_metrics(self, model, test_ds, model_name: str):
        """Calculate detailed metrics on test set"""
        all_targets = []
        all_predictions = []
        
        # Collect predictions
        for masked_batch, target_batch in test_ds:
            pred_batch = model.predict(masked_batch, verbose=0)
            
            all_targets.append(target_batch)
            all_predictions.append(pred_batch)
        
        # Concatenate all batches
        targets = tf.concat(all_targets, axis=0)
        predictions = tf.concat(all_predictions, axis=0)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(targets, predictions, model_name)
        self.results['metrics'][model_name] = metrics
        
        print(f"{model_name} metrics: {metrics}")
    
    def _generate_final_results(self, trained_models: Dict):
        """Generate final results, comparisons, and visualizations"""
        output_dir = self.config['data']['out_dir']
        
        # Print comparison
        comparison_text = self.metrics_calculator.compare_models()
        print(f"\n{comparison_text}")
        
        # Save comparison to file
        with open(os.path.join(output_dir, "model_comparison.txt"), 'w') as f:
            f.write(comparison_text)
        
        # Generate plots
        if self.results['histories']:
            self.visualizer.plot_training_history(
                self.results['histories'],
                os.path.join(output_dir, "training_history.png")
            )
        
        if self.results['metrics']:
            self.visualizer.create_metrics_comparison_plot(
                self.results['metrics'],
                os.path.join(output_dir, "metrics_comparison.png")
            )
        
        # Generate summary report
        self.visualizer.create_summary_report(
            self.results,
            os.path.join(output_dir, "summary_report.html")
        )
        
        print(f"\nAll results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Transformer Image Reconstruction Training Testbed")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/base_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Specific models to train (overrides config file)'
    )
    
    args = parser.parse_args()
    
    # Initialize testbed
    testbed = TrainingTestbed(args.config)
    
    # Override models if specified
    if args.models:
        testbed.config['models_to_train'] = args.models
        print(f"Overriding models to train: {args.models}")
    
    # Run experiment
    results = testbed.run_experiment()
    
    if results:
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("EXPERIMENT FAILED!")
        print(f"{'='*60}")
        sys.exit(1)

if __name__ == "__main__":
    main()