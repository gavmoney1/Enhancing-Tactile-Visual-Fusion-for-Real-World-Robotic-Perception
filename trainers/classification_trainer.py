import os
import time
import tensorflow as tf
from typing import Dict, Any
from tensorflow import keras
from utils.memory_utils import MemoryManager

class ClassificationTrainer:
    """Trainer for image classification models"""
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.history = None
        
        # Memory management
        self.memory_manager = MemoryManager(config)
        
    def build_and_compile_model(self, model_builder, train_ds):
        """Build and compile the classification model"""
        print(f"Building {self.model_name} for classification...")
        
        # Get sample batch to determine input shape and num_classes
        for batch_images, batch_labels in train_ds.take(1):
            input_shape = batch_images.shape[1:]
            # Handle both one-hot and sparse labels
            if len(batch_labels.shape) > 1 and batch_labels.shape[-1] > 1:
                num_classes = batch_labels.shape[-1]
            else:
                num_classes = self.config['classification']['num_classes']
            break
        
        print(f"Input shape: {input_shape}, Num classes: {num_classes}")
        
        # Build model
        self.model = model_builder.build_classifier(input_shape, num_classes)
        
        # Compile
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['optimization']['learning_rate']
        )
        
        # Choose loss based on label format
        if len(batch_labels.shape) > 1 and batch_labels.shape[-1] > 1:
            loss = keras.losses.CategoricalCrossentropy(
                label_smoothing=self.config['classification'].get('label_smoothing', 0.0)
            )
            metrics = ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=min(5, num_classes), name='top5_accuracy')]
        else:
            loss = keras.losses.SparseCategoricalCrossentropy()
            metrics = ['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=min(5, num_classes), name='top5_accuracy')]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print(f"Model compiled successfully")
        self.model.summary()
        
    def train(self, train_ds, val_ds) -> Dict[str, Any]:
        """Train the classification model"""
        print(f"\nTraining {self.model_name} for classification...")
        
        callbacks = self._setup_callbacks()
        
        # Check memory before training
        self.memory_manager.check_memory_usage()
        
        start_time = time.time()
        
        try:
            self.history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config['training']['epochs'],
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            print(f"Training completed in {training_time:.2f}s")
            
            # Clean up memory
            self.memory_manager.cleanup_memory()
            
            return {
                'history': self.history.history,
                'training_time': training_time
            }
            
        except Exception as e:
            print(f"Training failed for {self.model_name}: {str(e)}")
            self.memory_manager.cleanup_memory()
            raise e
    
    def evaluate(self, test_ds) -> Dict[str, float]:
        """Evaluate the model on test set"""
        print(f"\nEvaluating {self.model_name} on test set...")
        
        results = self.model.evaluate(test_ds, verbose=1, return_dict=True)
        
        print(f"Test results: {results}")
        
        # Clean up memory after evaluation
        self.memory_manager.cleanup_memory()
        
        return results
    
    def _setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_dir = os.path.join(
            self.config['data']['out_dir'], 
            'checkpoints', 
            f'{self.model_name}_classifier'
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config['training'].get('early_stopping', {}).get('enabled', False):
            monitor = self.config['training']['early_stopping'].get('monitor', 'val_loss')
            # Override monitor for classification if it's still set to val_loss
            if monitor == 'val_loss':
                monitor = 'val_accuracy'
            
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=self.config['training']['early_stopping'].get('patience', 10),
                mode='max' if 'accuracy' in monitor else 'min',
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Reduce LR on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks
