import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers.experimental import AdamW
from typing import Dict, Any
from utils.metrics import combined_loss, PSNRMetric, SSIMMetric
from utils.memory_utils import MemoryManager

@tf.keras.utils.register_keras_serializable(package="Custom")
class CombinedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Warmup followed by main schedule using tf.where for safety"""
    def __init__(self, warmup_schedule, main_schedule, warmup_steps):
        super().__init__()
        self.warmup_steps = float(warmup_steps)
        self.warmup_schedule = warmup_schedule
        self.main_schedule = main_schedule

    @tf.function
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        return tf.where(
            step < warmup_steps,
            self.warmup_schedule(step),
            self.main_schedule(step - warmup_steps)
        )

    def get_config(self):
        return {
            "warmup_steps": int(self.warmup_steps),
            "warmup_schedule": tf.keras.optimizers.schedules.serialize(self.warmup_schedule),
            "main_schedule": tf.keras.optimizers.schedules.serialize(self.main_schedule),
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            warmup_schedule=tf.keras.optimizers.schedules.deserialize(config["warmup_schedule"]),
            main_schedule=tf.keras.optimizers.schedules.deserialize(config["main_schedule"]),
            warmup_steps=config["warmup_steps"],
        )



class ModelTrainer:
    """Handles training of individual models"""
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        self.config = config
        self.model_name = model_name
        self.out_dir = os.path.join(config['data']['out_dir'], model_name)
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Memory management
        self.memory_manager = MemoryManager(config)
        
        # Training parameters
        self.epochs = int(config['training']['epochs'])
        self.learning_rate = float(config['optimization']['learning_rate'])
        self.weight_decay = float(config['optimization']['weight_decay'])
        self.warmup_steps = int(config['optimization']['warmup_steps'])
        self.model = None
        self.history = None
        
    def build_and_compile_model(self, model_builder, train_ds):
        """Build and compile the model"""
        print(f"\nBuilding {self.model_name} model...")
        
        self.model = model_builder.build_model()
        
        # Create learning rate schedule with warmup
        lr_schedule = self._create_lr_schedule(train_ds)
        
        # Configure optimizer
        optimizer = AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.weight_decay
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[PSNRMetric(name='psnr'), SSIMMetric(name='ssim')]
        )
        
        # Print model summary
        self.model.summary()
        
        # Estimate memory usage
        batch_size = self.config['training']['batch_size']
        img_size = self.config['data']['img_size']
        estimated_memory = self.memory_manager.estimate_model_memory(
            self.model, batch_size, img_size
        )
        print(f"Estimated memory usage: {estimated_memory:.2f} GB")
        
        return self.model
    
    def _create_lr_schedule(self, train_ds):
        """Create learning rate schedule with warmup"""
        warmup_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.0,
            decay_steps=int(self.warmup_steps),
            end_learning_rate=self.learning_rate,
            power=1.0
        )

        # approximate steps = epochs × steps_per_epoch
        steps_per_epoch = len(train_ds)
        main_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=int(self.epochs * steps_per_epoch),
            alpha=0.1
        )

        return CombinedSchedule(warmup_schedule, main_schedule, self.warmup_steps)
    
    def train(self, train_ds, val_ds) -> Dict[str, Any]:
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_and_compile_model first.")
        
        print(f"\nStarting training for {self.model_name}...")
        start_time = time.time()
        
        # Callbacks
        callbacks = self._create_callbacks()
        
        # Check memory before training
        self.memory_manager.check_memory_usage()
        
        # Train model
        try:
            self.history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Save final model
            self.model.save(os.path.join(self.out_dir, "final_model.keras"))
            
            # Clean up memory
            self.memory_manager.cleanup_memory()
            
            return {
                'history': self.history.history,
                'training_time': training_time,
                'model_path': os.path.join(self.out_dir, "best_model.keras")
            }
            
        except Exception as e:
            print(f"Training failed for {self.model_name}: {str(e)}")
            self.memory_manager.cleanup_memory()
            raise e
    
    def _create_callbacks(self):
        """Create training callbacks"""
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.out_dir, "best_model.keras"),
                monitor="val_ssim",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_ssim",
                mode="max",
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(self.out_dir, "training_log.csv")
            )
        ]
        
        # Memory monitoring callback
        class MemoryCallback(tf.keras.callbacks.Callback):
            def __init__(self, memory_manager):
                super().__init__()
                self.memory_manager = memory_manager
            
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0:  # Check every 5 epochs
                    self.memory_manager.check_memory_usage()
        
        callbacks.append(MemoryCallback(self.memory_manager))
        
        return callbacks
    
    def evaluate(self, test_ds) -> Dict[str, float]:
        """Evaluate the trained model"""
        if self.model is None:
            # Load best model
            model_path = os.path.join(self.out_dir, "best_model.keras")
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        'CombinedSchedule': CombinedSchedule,
                        'combined_loss': combined_loss,
                        'psnr_metric': PSNRMetric,
                        'ssim_metric': SSIMMetric
                    }
                )
            else:
                raise ValueError("No trained model found")
        
        print(f"Evaluating {self.model_name}...")
        results = self.model.evaluate(test_ds, return_dict=True, verbose=1)
        
        # Save evaluation results
        eval_path = os.path.join(self.out_dir, "evaluation_results.txt")
        with open(eval_path, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value:.6f}\n")
        
        return results
