import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random

class PotatoDiseaseClassifier:
    def __init__(self, model_path):
        """Initialize the classifier with a model path."""
        self.model = None
        self.model_path = model_path
        self.class_names = ['Early_Blight', 'Late_Blight', 'Healthy']
        self.img_height = 224  # Standard input size for ResNet50V2
        self.img_width = 224   # Standard input size for ResNet50V2
        self.load_model()

    def create_model(self):
        """Create a model using transfer learning with ResNet50V2"""
        try:
            # Load pre-trained ResNet50V2 model
            base_model = tf.keras.applications.ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_shape=(self.img_height, self.img_width, 3)
            )
            
            # Freeze the base model
            base_model.trainable = False
            
            # Create the model
            model = tf.keras.Sequential([
                # Input layer
                tf.keras.layers.Input(shape=(self.img_height, self.img_width, 3)),
                
                # Data preprocessing
                tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
                
                # Base model
                base_model,
                
                # Classification layers
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            # Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            raise Exception(f"Error creating model: {str(e)}")

    def load_model(self):
        """Load or create the model."""
        try:
            if os.path.exists(self.model_path):
                # Clear any existing models from memory
                tf.keras.backend.clear_session()
                # Load model with custom object scope to ensure proper loading
                with tf.keras.utils.custom_object_scope({'tf': tf}):
                    self.model = tf.keras.models.load_model(self.model_path)
                print("Model loaded successfully")
            else:
                print("Model file not found. Creating new model...")
                self.model = self.create_model()
                # Save the model
                self.model.save(self.model_path)
                print("New model created and saved successfully")
        except Exception as e:
            raise Exception(f"Error with model: {str(e)}")

    def preprocess_image(self, image):
        """Preprocess the image for model prediction."""
        try:
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image to match model's expected sizing
            image = image.resize((self.img_width, self.img_height), Image.Resampling.BILINEAR)
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)
            
            # Ensure shape is correct
            if image_array.shape != (self.img_height, self.img_width, 3):
                raise ValueError(f"Invalid image shape: {image_array.shape}")
            
            # Expand dimensions to create batch
            image_array = np.expand_dims(image_array, 0)
            
            # Preprocess input (normalizing to the range expected by ResNet50V2)
            image_array = tf.keras.applications.resnet_v2.preprocess_input(image_array)
            
            return image_array
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")

    def predict(self, image):
        """Simulate different diagnoses for demonstration"""
        try:
            # Randomly select a class and generate a high confidence
            predicted_class = random.choice(self.class_names)
            confidence = random.uniform(0.85, 0.98)
            
            # Create simulated probabilities
            probabilities = {
                'Early_Blight': 0.1,
                'Late_Blight': 0.1,
                'Healthy': 0.1
            }
            probabilities[predicted_class] = confidence
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities
            }
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")

def test_model(model_path, test_image_path):
    """Test the model with a sample image"""
    try:
        classifier = PotatoDiseaseClassifier(model_path)
        image = Image.open(test_image_path)
        result = classifier.predict(image)
        print(f"Test Result: {result}")
        return result
    except Exception as e:
        print(f"Test Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the classifier
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'potato_disease_model.h5')
    classifier = PotatoDiseaseClassifier(model_path)
