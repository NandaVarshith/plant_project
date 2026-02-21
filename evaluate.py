import tensorflow as tf

# Paths
test_dir = "dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Load saved model
model = tf.keras.models.load_model("leaf_species_model_finetuned.keras")

# Evaluate
loss, accuracy = model.evaluate(test_dataset)

print("Test Accuracy:", accuracy)
print("Test Loss:", loss)