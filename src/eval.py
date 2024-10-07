import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from datamodules.dogbreed_dataset import DogBreedDataModule
from models.dogbreed_classifier import DogBreedClassifier
from utils.logging_utils import setup_logger


def load_model_from_checkpoint(checkpoint_path: str) -> DogBreedClassifier:
    return DogBreedClassifier.load_from_checkpoint(checkpoint_path)


def evaluate(data_module: DogBreedDataModule, model: DogBreedClassifier) -> None:
    trainer = L.Trainer(
        accelerator="auto",
        logger=TensorBoardLogger(save_dir="logs", name="dogbreed_evaluation"),
    )
    results = trainer.validate(model, datamodule=data_module)

    print("\nValidation Metrics:")
    for k, v in results[0].items():
        print(f"{k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DogBreed Classifier")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    args = parser.parse_args()

    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    log_dir = base_dir / "logs"

    # Set up logger
    setup_logger(log_dir / "eval_log.log")

    # Initialize DataModule
    data_module = DogBreedDataModule(data_dir=data_dir, batch_size=32, num_workers=0)
    data_module.setup(stage="validate")

    # Load model from checkpoint
    model = load_model_from_checkpoint(args.checkpoint)

    # Run evaluation
    evaluate(data_module, model)


if __name__ == "__main__":
    main()
