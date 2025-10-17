from inference_engine import InferenceEngine

engine = InferenceEngine(
    model_path="models/vit_model.keras",
    config_path="configs/base_config.yaml",
    img_size=224,
    enable_visualization=True
)

results = engine.make_predictions()
print("Final metrics:", results)
