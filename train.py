from pathlib import Path
import tensorflow as tf
from dataset_utils import create_dataset, NUM_CLASSES, CLASS_NAMES, CLASS_COLORS
from model import build_unet

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
VALID_DIR = BASE_DIR / "valid"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    """Train the U-Net model and save the trained weights."""
    image_size = (256, 256)
    batch_size = 8
    epochs = 30

    print("Loading training dataset...")
    train_dataset = create_dataset(TRAIN_DIR, image_size=image_size, batch_size=batch_size, shuffle=True)
    print("Loading validation dataset...")
    valid_dataset = create_dataset(VALID_DIR, image_size=image_size, batch_size=batch_size, shuffle=False)

    print("Building the model...")
    model = build_unet(input_shape=(*image_size, 3), num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "crop_area_unet.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("Starting training...")
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    print("Saving final model weights...")
    model.save(MODEL_DIR / "crop_area_unet.keras")
    model.save(MODEL_DIR / "crop_area_unet.h5")
    print("Training complete. Model files saved in models/.")


if __name__ == "__main__":
    main()
