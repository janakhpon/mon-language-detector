import argparse
import fasttext
import time
from pathlib import Path

from .utils import get_logger, PROJECT_ROOT

logger = get_logger(__name__)

def validate_data_files(train_file: Path, valid_file: Path):
    """Ensure training data exists before starting."""
    if not train_file.exists() or not valid_file.exists():
        raise FileNotFoundError(f"Training data files missing. Run pipeline first.\nTrain: {train_file}\nValid: {valid_file}")

def train_model(
    train_file: Path,
    valid_file: Path,
    model_output: Path,
    quantized_output: Path,
    lr: float,
    epoch: int,
    word_ngrams: int,
    minn: int,
    maxn: int,
    dim: int,
    thread: int
):
    """Train the fastText language identification model."""
    logger.info("Initializing fastText model training...")
    validate_data_files(train_file, valid_file)
    
    start_time = time.time()
    
    try:
        # Hyperparameters for maximum reliability on sub-word orthographic patterns
        model = fasttext.train_supervised(
            input=str(train_file),
            lr=lr,
            epoch=epoch,
            wordNgrams=word_ngrams,
            minn=minn,
            maxn=maxn,
            dim=dim,
            loss='softmax',
            thread=thread
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
        
    duration = time.time() - start_time
    logger.info(f"Training completed successfully in {duration:.2f} seconds.")
    
    logger.info("Evaluating on validation set...")
    result = model.test(str(valid_file))
    
    logger.info(f"Validation Examples: {result[0]}")
    logger.info(f"Precision@1: {result[1]:.4f}")
    logger.info(f"Recall@1: {result[2]:.4f}")
    
    labels = model.get_labels()
    logger.info(f"Detected classes: {labels}")
    
    model_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save_model(str(model_output))
        size_mb = model_output.stat().st_size / (1024*1024)
        logger.info(f"Full model saved to {model_output} ({size_mb:.2f} MB)")
    except IOError as e:
        logger.error(f"Failed to save model to {model_output}: {e}")
        raise
    
    logger.info("Quantizing model for production deployment...")
    try:
        model.quantize(input=str(train_file), qnorm=True, retrain=True, cutoff=100000)
        model.save_model(str(quantized_output))
        
        q_result = model.test(str(valid_file))
        q_size_mb = quantized_output.stat().st_size / (1024*1024)
        logger.info(f"Compressed model saved to {quantized_output} ({q_size_mb:.2f} MB)")
        logger.info(f"Quantized Precision@1: {q_result[1]:.4f}")
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        # Not raising, since full model succeeded

def main():
    parser = argparse.ArgumentParser(description="Train fastText language detector")
    parser.add_argument("--train-file", type=Path, default=PROJECT_ROOT / "data" / "train.txt")
    parser.add_argument("--valid-file", type=Path, default=PROJECT_ROOT / "data" / "valid.txt")
    parser.add_argument("--model-output", type=Path, default=PROJECT_ROOT / "data" / "langid_mon_mya_eng.bin")
    parser.add_argument("--quantized-output", type=Path, default=PROJECT_ROOT / "data" / "langid_mon_mya_eng_compressed.ftz")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--word-ngrams", type=int, default=1)
    parser.add_argument("--minn", type=int, default=2)
    parser.add_argument("--maxn", type=int, default=5)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--thread", type=int, default=4)
    
    args = parser.parse_args()
    
    train_model(
        train_file=args.train_file,
        valid_file=args.valid_file,
        model_output=args.model_output,
        quantized_output=args.quantized_output,
        lr=args.lr,
        epoch=args.epoch,
        word_ngrams=args.word_ngrams,
        minn=args.minn,
        maxn=args.maxn,
        dim=args.dim,
        thread=args.thread
    )

if __name__ == "__main__":
    main()
