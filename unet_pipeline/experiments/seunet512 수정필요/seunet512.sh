cd unet_pipeline
python Train.py ./experiments/seunet512/01_train_config_part0.yaml
python Train.py ./experiments/seunet512/02_train_config_part1.yaml
python Train.py ./experiments/seunet512/03_train_config_part2.yaml
python Inference.py ./experiments/seunet512/04_inference_config.yaml
python TripletSubmit.py ./experiments/seunet512/05_submit.yaml

