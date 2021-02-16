cd unet_pipeline
python Train.py ./experiments/albunet512F10/01_train_config_part0.yaml
python Train.py ./experiments/albunet512F10/02_train_config_part1.yaml
python Inference.py ./experiments/albunet512F10/03_inference_config.yaml
python TripletSubmit.py ./experiments/albunet512F10/04_submit.yaml

