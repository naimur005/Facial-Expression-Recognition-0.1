"""
Train a CNN model to classify facial expressions (Happy vs Disappointed).
Usage:
    python train.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, MaxPooling2D, Conv2D, Dropout, BatchNormalization,
    Activation, Input, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


TRAIN_PATH = './Dataset/Train'
VALID_PATH = './Dataset/Test'

MODEL_SAVE_PATH = './fer_model_best.keras'
FINAL_MODEL_PATH = './fer_model_final.keras'

IMG_SIZE = (48, 48)           
BATCH_SIZE = 32               
EPOCHS = 50                   
LEARNING_RATE = 0.001     

CLASS_LABELS = ['Disappointed', 'Happy']
NUM_CLASSES = len(CLASS_LABELS)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def build_model(input_shape, num_classes):
    """
    Build CNN model optimized to prevent overfitting.
    
    Features:
    - 3 Conv blocks with BatchNorm and Dropout
    - GlobalAveragePooling (reduces parameters)
    - L2 regularization on dense layers
    - Softmax output for proper probabilities
    """
    model = Sequential([
        Input(shape=input_shape),
        
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        GlobalAveragePooling2D(),
        
        Dense(256, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(128, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_PATH,
        target_size=IMG_SIZE,
        color_mode='grayscale',  
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        directory=VALID_PATH,
        target_size=IMG_SIZE,
        color_mode='grayscale',  
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return train_generator, valid_generator


def train_model():
    """Main training function."""
    
    print("="*60)
    print("Facial Expression Recognition - Training")
    print("="*60)
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    if not os.path.exists(TRAIN_PATH):
        print(f"\n❌ ERROR: Training path not found: {TRAIN_PATH}")
        print("Please update TRAIN_PATH in the configuration section.")
        return
    
    if not os.path.exists(VALID_PATH):
        print(f"\n❌ ERROR: Validation path not found: {VALID_PATH}")
        print("Please update VALID_PATH in the configuration section.")
        return
    
    print("\n📁 Loading data...")
    train_generator, valid_generator = create_data_generators()
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {valid_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    
    print("\n🏗️  Building model...")
    input_shape = (*IMG_SIZE, 1) 
    model = build_model(input_shape, NUM_CLASSES)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    

    print("\n📊 Model Summary:")
    model.summary()
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = valid_generator.samples // BATCH_SIZE
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n⚖️  Class weights (to fix bias): {class_weight_dict}")
    
    print("\n🚀 Starting training...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    print()
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight_dict,  
        verbose=1
    )
    
    model.save(FINAL_MODEL_PATH)
    print(f"\n✅ Final model saved to: {FINAL_MODEL_PATH}")
    print(f"✅ Best model saved to: {MODEL_SAVE_PATH}")
    
    plot_history(history)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, history


def plot_history(history):
    """Plot and save training history."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n📈 Training history saved to: training_history.png")
    plt.show()

if __name__ == "__main__":
    train_model()
