# Dog Breed Classification

This project uses deep learning to classify dog breeds from images.

## Data

Dataset: [Dog Breed Image Dataset](https://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset) from Kaggle
Training data: [Google Drive Link](https://drive.google.com/file/d/1j_Z_y5zXJWZd5Z6X5Z6X5Z6X5Z6X5Z6/view?usp=sharing)

## Usage

### Train
image_name: `dogbreed-classifier`
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs -it {image_name} python src/train.py
```

### Eval
image_name: `dogbreed-classifier`
Replace `$checkpoint_filename` with the actual checkpoint file name.
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs -it {image_name} python src/eval.py 'logs/dogbreed_classification/checkpoints/$checkpoint_filename'
```

### Infer
image_name: `dogbreed-classifier`
Replace `$checkpoint_filename` with the actual checkpoint file name.
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs -v $(pwd)/samples:/app/samples -v $(pwd)/predictions:/app/predictions -it {image_name} python src/infer.py --input_folder samples --output_folder predictions --ckpt_path 'logs/dogbreed_classification/checkpoints/$checkpoint_filename'
```

## Project Structure

- `data/`: Contains the dataset
- `logs/`: Stores training logs and checkpoints
- `samples/`: Input folder for inference
- `predictions/`: Output folder for inference results
- `src/`: Source code
  - `train.py`: Training script
  - `eval.py`: Evaluation script
  - `infer.py`: Inference script
  - `models/`: Contains model architectures
  - `utils/`: Utility functions
- `Dockerfile`: Docker configuration for the project
- `uv.lock`: Lock file for uv
- `pyproject.toml`: Configuration for the project

