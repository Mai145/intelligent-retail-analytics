import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURATION ---
MODEL_PATH = "best_emotion_model_kfold.keras"
CSV_FILENAME = "dataset.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def evaluate_and_plot():
    """Evaluates the best saved model and plots a Confusion Matrix."""
    print("--- LOADING MODEL AND DATASET ---")

    # 1. Load Model and Data
    if not tf.io.gfile.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found!")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    df = pd.read_csv(CSV_FILENAME)

    # Use the entire dataset for final evaluation
    datagen = ImageDataGenerator(rescale=1. / 255)

    eval_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="image_path",
        y_col="emotion",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False  # DO NOT SHUFFLE for correct matrix matching
    )

    # 2. Get Predictions
    print("Generating predictions on the whole dataset...")
    predictions = model.predict(eval_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = eval_generator.classes
    class_labels = list(eval_generator.class_indices.keys())

    # 3. Generate Classification Report (Corrected parameter: target_names)
    print("\n" + "=" * 40)
    print("      CLASSIFICATION REPORT      ")
    print("=" * 40)
    # Replaced 'target_size' with 'target_names'
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # 4. Create Confusion Matrix (Visual results)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix: Emotion Recognition (RAVDESS)')
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')

    # Save the plot for your final presentation
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nConfusion Matrix saved as 'confusion_matrix.png'")

    # Show the plot window
    plt.show()


if __name__ == "__main__":
    # Ensure GPU memory growth is enabled
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    evaluate_and_plot()