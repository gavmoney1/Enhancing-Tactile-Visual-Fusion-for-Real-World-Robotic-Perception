from inference_engine import InferenceEngine

engine = InferenceEngine(
    model_path="/bighome/gcmoney/Enhancing-Tactile-Visual-Fusion-for-Real-World-Robotic-Perception/demo_test/conv_autoencoder/best_model.keras",
    config_path="configs/base_config.yaml",
    img_size=224,
    enable_visualization=True
)

results = engine.make_predictions()
print("Final metrics:", results)
