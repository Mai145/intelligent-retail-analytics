import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

# --- CONFIGURATION ---
CSV_FILENAME = "dataset.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # 10 Epochs per fold for a solid baseline
N_SPLITS = 10


def build_model():
    """Builds and compiles the MobileNetV2 based emotion classification model."""
    input_layer = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Load pre-trained MobileNetV2 without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_layer)

    # Freeze the base model layers (feature extraction only)
    base_model.trainable = False

    # Add custom classification head for 8 emotions
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(8, activation='softmax', name='emotion_prediction')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_evaluate_kfold():
    """Executes 10-Fold Cross Validation on the dataset."""
    print("--- STARTING 10-FOLD CROSS VALIDATION ---")

    # 1. Load the dataset
    if not os.path.exists(CSV_FILENAME):
        print(f"CRITICAL ERROR: {CSV_FILENAME} not found. Please run main.py first.")
        return

    df = pd.read_csv(CSV_FILENAME)
    print(f"Total samples loaded from CSV: {len(df)}")

    # Ensure dataframe columns are treated as strings (Required by ImageDataGenerator)
    df['image_path'] = df['image_path'].astype(str)
    df['emotion'] = df['emotion'].astype(str)

    # 2. Define K-Fold strategy
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Lists to store results for the final academic report
    fold_accuracies = []
    fold_losses = []

    # Image Generator for pure normalization (No augmentation for pure accuracy test)
    datagen = ImageDataGenerator(rescale=1. / 255)

    fold_no = 1
    best_accuracy = 0.0

    # 3. K-Fold Training Loop
    for train_index, val_index in kfold.split(df):
        print("\n" + "=" * 50)
        print(f"   TRAINING FOLD {fold_no} / {N_SPLITS}   ")
        print("=" * 50)

        # Split the dataframe into train and validation for the current fold
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]

        # Create data generators
        train_generator = datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col="image_path",
            y_col="emotion",
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=True
        )

        val_generator = datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col="image_path",
            y_col="emotion",
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False  # Do not shuffle validation data for accurate evaluation
        )

        # Initialize a fresh model for this fold
        model = build_model()

        # Train the model
        model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            verbose=1  # Show progress bar
        )

        # Evaluate the model on the unseen validation fold
        print(f"\nEvaluating Fold {fold_no}...")
        eval_results = model.evaluate(val_generator, verbose=0)
        loss = eval_results[0]
        accuracy = eval_results[1]

        print(f"Fold {fold_no} Results -> Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

        fold_accuracies.append(accuracy * 100)
        fold_losses.append(loss)

        # Save the best performing model dynamically for later use
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save("best_emotion_model_kfold.keras")
            print(f"--> New best model saved! (Accuracy: {accuracy * 100:.2f}%)")

        fold_no += 1

    # 4. FINAL ACADEMIC REPORT
    print("\n" + "=" * 60)
    print("   10-FOLD CROSS VALIDATION FINAL REPORT   ")
    print("=" * 60)
    for i in range(N_SPLITS):
        print(f"Fold {i + 1} - Accuracy: %{fold_accuracies[i]:.2f} - Loss: {fold_losses[i]:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print("-" * 60)
    print(f"AVERAGE ACCURACY : %{mean_acc:.2f} (+/- %{std_acc:.2f})")
    print(f"AVERAGE LOSS     : {np.mean(fold_losses):.4f}")
    print("=" * 60)
    print("The best performing model has been saved as 'best_emotion_model_kfold.keras'.")


if __name__ == "__main__":
    # Ensure GPU memory growth is enabled to prevent sudden memory crashes
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Support Enabled: {len(physical_devices)} GPU(s) found.")
        except RuntimeError as e:
            print(e)

    train_evaluate_kfold()