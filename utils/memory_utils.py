import tensorflow as tf
import psutil
import gc

class MemoryManager:
    """Manage memory usage during training"""
    
    def __init__(self, config):
        self.max_memory_gb = config['memory']['max_memory_gb']
        self.enable_mixed_precision = config['memory']['enable_mixed_precision']
        self.gradient_accumulation_steps = config['memory']['gradient_accumulation_steps'] # UNUSED
        
        # Enable mixed precision if requested - disable for now due to compatibility issues
        # if self.enable_mixed_precision:
        #     tf.keras.mixed_precision.set_global_policy('mixed_float16')
        #     print("Mixed precision training enabled")
        print("Mixed precision disabled for debugging")
    
    def check_memory_usage(self):
        """Check current memory usage"""
        memory_info = psutil.virtual_memory()
        memory_used_gb = (memory_info.total - memory_info.available) / (1024**3)
        memory_percent = memory_info.percent
        
        print(f"Memory usage: {memory_used_gb:.2f} GB ({memory_percent:.1f}%)")
        
        if memory_used_gb > self.max_memory_gb * 0.9:
            print("Warning: High memory usage detected")
            self.cleanup_memory()
    
    def cleanup_memory(self):
        """Clean up memory"""
        gc.collect()
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
        print("Memory cleanup performed")
    
    @staticmethod
    def estimate_model_memory(model, batch_size, img_size):
        """Estimate memory usage of a model"""
        # Rough estimation based on parameters and batch size
        num_params = model.count_params()
        input_size = batch_size * img_size * img_size * 3 * 4  # 4 bytes per float32
        param_size = num_params * 4  # 4 bytes per parameter
        
        # Rough estimate including gradients and intermediate activations
        estimated_gb = (input_size + param_size * 3) / (1024**3)
        
        return estimated_gb
