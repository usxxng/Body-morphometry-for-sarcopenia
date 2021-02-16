cd unet_pipeline
python Train.py ./experiments/albunet1024fold10albJW_2/01_train_config_part0.yaml
python Train.py ./experiments/albunet1024fold10albJW_2/02_train_config_part1.yaml
python Train.py ./experiments/albunet1024fold10albJW_2/03_train_config_part2.yaml
python Inference.py ./experiments/albunet1024fold10albJW_2/04_inference_config.yaml
python TripletSubmit.py ./experiments/albunet1024fold10albJW_2/05_submit.yaml






