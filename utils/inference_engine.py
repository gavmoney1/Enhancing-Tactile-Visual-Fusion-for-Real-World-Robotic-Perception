import tensorflow as tf
import os

class InferenceEngine:
    """
    Inference engine for pretrained models. 
    Handles model loading and image reconstruction.
    """

    def __init__(self, model_class, weights_path, config):
        """"
        Initializes engine with a given model and pretrained weights

        Args:
            model_class (object): class of model (ex: MAEUpModel)
            weights_path (str): Path to trained model (.h5)
            config (dict): Model configuration (same used during training)
        """
        self.model_class = model_class
        self.config = config
        self.weights_path = weights_path
        self.model = None

        self._load_model()

    def _load_model(self):
        """
        Load model architecture with pretrained weights
        """
        print("Building model...")
        base_model = self.model_class(self.config)
        self.model = base_model.build_model()

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        print(f"Loading weights from {self.weights_path}")
        self.model.load_weights(self.weights_path)
        print("Model weights loaded successfully!")

    def _preprocess_image(self, image_path):
        """
        Prepare an image for model input
        """
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (self.config['data']['img_size'], self.config['data']['img_size']))
        img = tf.expand_dims(img, axis=0)  # [1, H, W, 3]
        return img
    
    def _postprocess_image(self, tensor):
        """
        Convert tensor [H, W, 3] to uint8 image for saving
        """
        tensor = tf.clip_by_value(tensor, 0.0, 1.0)
        tensor = tf.image.convert_image_dtype(tensor, tf.uint8)
        return tensor
    
    def predict_image(self, image_path):
        """
        Basic function to predict one image
        Might have to be change depending on the structure of 
        the imported models
        """
        img = self._preprocess_image(image_path)
        predictions = self.model(img, training=False)
        return predictions[0]
    
    def make_predictions(self, input_root, output_root):
        """
        Run predictions on all images in a folder.

        Args:
            input_root (str): Root directory containing input images
            output_root (str): Directory to save predictions
        """
        if not os.path.exists(input_root):
            raise FileNotFoundError(f"Input folder not found: {input_root}")
        
        # Walk through input directory to gather image files, 
        # handles images stored in subdirectories
        images = []
        for root, _, files in os.walk(input_root):
            for fname in files:
                if not fname.lower().endswith((".jpg", ".jpeg",".png")):
                    continue

        if not images:
            raise ValueError(f"No images found in {input_root}")
        
        total_images = len(images)
        print(f"Running inferences on {total_images} images...")

        # Loop through each image and run a prediction
        for i, img_path in enumerate(images, start=1):
            print(f"[{i}/{total_images}] Processing: {os.path.basename(img_path)}")
            img_prediction = self.predict_image(img_path)

        # Save predicted images'as images in output_root
        relative_path = os.path.relpath(img_path, input_root)
        save_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(output_root), exist_ok=True)
        img_out = self._postprocess_image(img_prediction)
        tf.io.write_file(save_path, tf.image.encode_jpeg(img_out))

        print(f"Inference completed for {total_images} images.")
        print(f"Predictions saved under: {output_root}")
        


