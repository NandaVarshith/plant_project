import tensorflow as tf
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# ======================
# CONFIG
# ======================
train_csv = "train_labels.csv"
test_csv = "test_labels.csv"

train_img_dir = "train_images/train_images"
test_img_dir = "test_images/test_images"

IMG_SIZE = (300, 300)
BATCH_SIZE = 32

# ======================
# LOAD CSV
# ======================
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

le = LabelEncoder()
train_df['label_encoded'] = le.fit_transform(train_df['label'])
test_df['label_encoded'] = le.transform(test_df['label'])

num_classes = len(le.classes_)

# ======================
# IMAGE FUNCTION
# ======================
def load_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, tf.one_hot(label, num_classes)

# ======================
# DATASETS
# ======================
train_paths = train_df['filename'].apply(lambda x: os.path.join(train_img_dir, x)).values
train_labels = train_df['label_encoded'].values

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_paths = test_df['filename'].apply(lambda x: os.path.join(test_img_dir, x)).values
test_labels = test_df['label_encoded'].values

test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_dataset = test_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ======================
# LOAD TRAINED MODEL (12 epochs)
# ======================
model = tf.keras.models.load_model("multiorgan_model_12epochs.keras")

print("Model loaded successfully.")

# ======================
# UNFREEZE TOP 30% OF BACKBONE
# ======================

# First make entire model trainable
model.trainable = True

# Freeze first 70% layers
total_layers = len(model.layers)
freeze_until = int(total_layers * 0.7)

for layer in model.layers[:freeze_until]:
    layer.trainable = False

print(f"Total layers: {total_layers}")
print(f"Frozen layers: {freeze_until}")
print(f"Trainable layers: {total_layers - freeze_until}")

# ======================
# RECOMPILE WITH LOW LR
# ======================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting fine-tuning...")

# ======================
# TRAIN 5 EPOCHS
# ======================
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5
)

# ======================
# SAVE FINAL MODEL
# ======================
model.save("multiorgan_final_finetuned.keras")

print("Fine-tuning complete.")