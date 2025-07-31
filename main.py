"""
main.py

Entry point: train the model pipeline and evaluate on the test set.
"""

import logging

from src.train import train_pipeline
from src.evaluate import evaluate_model

def main() -> None:
    """
    Train the multi-output regression pipeline and evaluate its performance.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline")
    model, X_test, y_test = train_pipeline()
    logger.info("Evaluating model on test set")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s"
    )
    main()
