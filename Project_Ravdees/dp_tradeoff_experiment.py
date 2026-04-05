import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_privacy as tf_privacy
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy import compute_dp_sgd_privacy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

# --- 1. CONFIGURATION ---
CSV_FILENAME = "dataset.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # Kept low to ensure the experiment finishes overnight
N_SPLITS = 5  # 3-Fold CV: Balances academic rigor with computational time for DP
LEARNING_RATE = 0.001
L2_NORM_CLIP = 1.0  # DP Rule: Bounding the maximum gradient norm
DELTA = 1e-5  # Constant tolerance value for DP formulation

# Noise multipliers to test.
noise_multipliers = [0.1, 0.5, 1.0, 2.0, 4.0]

results_epsilon = []
results_accuracy = []

# --- 2. DATASET LOADING ---
print("Loading dataset from CSV...")
if not os.path.exists(CSV_FILENAME):
    raise FileNotFoundError(f"CRITICAL ERROR: {CSV_FILENAME} not found. Run main.py first.")

df = pd.read_csv(CSV_FILENAME)
print(f"Total samples loaded from CSV: {len(df)}")

df['image_path'] = df['image_path'].astype(str)
df['emotion'] = df['emotion'].astype(str)

datagen = ImageDataGenerator(rescale=1. / 255)


def build_model():
    """Builds a fresh MobileNetV2 model."""
    input_layer = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_layer)

    base_model.trainable = False  # Feature extraction only

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output_layer = tf.keras.layers.Dense(8, activation='softmax', name='emotion_prediction')(x)

    return tf.keras.models.Model(inputs=base_model.input, outputs=output_layer)


print("\n--- STARTING DP + K-FOLD EXPERIMENT LOOP ---")

kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# --- 3. THE EXPERIMENT LOOP (NOISE x FOLDS) ---
for noise in noise_multipliers:
    print("\n" + "=" * 60)
    print(f"   TESTING NOISE MULTIPLIER: {noise}   ")
    print("=" * 60)

    fold_accuracies = []
    fold_no = 1

    for train_index, val_index in kfold.split(df):
        print(f"\n--- Fold {fold_no} / {N_SPLITS} for Noise {noise} ---")

        # Free up GPU memory before starting a new fold
        tf.keras.backend.clear_session()

        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]

        train_generator = datagen.flow_from_dataframe(
            dataframe=train_df, x_col="image_path", y_col="emotion",
            target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True
        )

        val_generator = datagen.flow_from_dataframe(
            dataframe=val_df, x_col="image_path", y_col="emotion",
            target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
        )

        model = build_model()

        optimizer = tf_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=L2_NORM_CLIP,
            noise_multiplier=noise,
            num_microbatches=1,
            learning_rate=LEARNING_RATE
        )

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the fold
        model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, verbose=1)

        # Evaluate the fold
        val_acc = model.evaluate(val_generator, verbose=0)[1]
        print(f"Fold {fold_no} Validation Accuracy: {val_acc:.4f}")
        fold_accuracies.append(val_acc)

        fold_no += 1

    # Calculate Mean Accuracy across all folds for this specific noise
    mean_noise_acc = np.mean(fold_accuracies)

    # Calculate Privacy Budget (Epsilon)
    # Using the train_generator size from the last fold as 'N' (they are almost identical)
    N = train_generator.samples
    epsilon, _ = compute_dp_sgd_privacy(n=N, batch_size=BATCH_SIZE, noise_multiplier=noise, epochs=EPOCHS, delta=DELTA)

    print(
        f"\n[!!!] FINAL RESULT FOR NOISE {noise} -> Epsilon (ε): {epsilon:.2f} | 3-Fold Mean Accuracy: {mean_noise_acc:.4f}")

    # Store for the graph
    results_epsilon.append(epsilon)
    results_accuracy.append(mean_noise_acc)

# --- 4. PLOTTING THE TRADE-OFF CURVE ---
print("\n" + "=" * 60)
print("   EXPERIMENT FINISHED. PLOTTING THE TRADE-OFF CURVE   ")
print("=" * 60)

plt.figure(figsize=(10, 6))
plt.plot(results_epsilon, results_accuracy, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

plt.title(f'Privacy-Utility Trade-off (RAVDESS, {N_SPLITS}-Fold CV)', fontsize=14, fontweight='bold')
plt.xlabel('Privacy Budget ($\epsilon$) - Lower is More Private', fontsize=12)
plt.ylabel(f'Mean Validation Accuracy ({N_SPLITS}-Fold)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().invert_xaxis()

plt.savefig('privacy_tradeoff_curve_kfold.png', dpi=300)
print("[!] Graph successfully saved as 'privacy_tradeoff_curve_kfold.png'")
plt.show()