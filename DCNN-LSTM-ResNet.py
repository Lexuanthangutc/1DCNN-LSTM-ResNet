import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, LSTM, Dense, Add, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Concatenate

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

tf.random.set_seed(42)
np.random.seed(42)
basepath = os.getcwd()

with open(basepath + '/new_data/train_Z24_08_08.p', 'rb') as f:
    input_train = pickle.load(f)
    output_train = pickle.load(f)
with open(basepath + '/new_data/valid_Z24_08_08.p', 'rb') as f:
    input_valid = pickle.load(f)
    output_valid = pickle.load(f)    
    
input_train = input_train.transpose(0,2,1)
input_valid = input_valid.transpose(0,2,1)


X_train, X_temp, y_train, y_temp = train_test_split(input_train, output_train, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_valid = np.concatenate((X_valid,X_train))
y_valid = np.concatenate((y_valid,y_train))

label=np.unique(y_train)
print('Label = ' + str(label))
num_classes = len(np.unique(y_train))
print('No. Labels: ' + str(num_classes))





def resnet_block(x, filters, kernel_size=3, stride=1, dilation_rate=1, use_projection_shortcut=False, use_layer_norm=False):
    """Standard ResNet block."""
    F1, F2, F3 = filters
    shortcut = x
    
    # First component: 1x1 convolution
    x = Conv1D(filters=F1, kernel_size=1, strides=stride, dilation_rate=dilation_rate, padding='valid')(x)
    if use_layer_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second component: 3x3 convolution
    x = Conv1D(filters=F2, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, padding='same')(x)
    if use_layer_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Third component: 1x1 convolution
    x = Conv1D(filters=F3, kernel_size=1, strides=1, dilation_rate=dilation_rate, padding='valid')(x)
    if use_layer_norm:
        x = BatchNormalization()(x)
    
    # Adjust the shortcut if needed (projection shortcut)
    if use_projection_shortcut:
        shortcut = Conv1D(filters=F3, kernel_size=1, strides=stride, dilation_rate=dilation_rate, padding='valid')(shortcut)
        if use_layer_norm:
            shortcut = BatchNormalization()(shortcut)
    
    # Add the shortcut to the main path
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def build_model(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    
    # Initial Conv Layer with LayerNormalization
    x = Conv1D(filters=64, kernel_size=7, padding="same", strides=2)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # ResNet Blocks with Dilated Convolutions
    x = resnet_block(x, [64, 64, 256], use_projection_shortcut=True)  # Regular block
    x_dilated = resnet_block(x, [64, 64, 256], use_projection_shortcut=True, dilation_rate=2)  # Dilated block
    
    # Merge the two
    x = Add()([x, x_dilated])
    
    # LSTM
    # lstm = LSTM(256, return_sequences=True, recurrent_activation='softmax', dropout=0.5)(x)
    lstm = LSTM(128, return_sequences=True, recurrent_activation='softmax')(x)
    
    
    # Double Skip Connection from Original Input and Intermediate Layer
    shortcut1 = Conv1D(filters=lstm.shape[-1], kernel_size=1, padding="same", strides=2)(input_tensor)
    shortcut1 = BatchNormalization()(shortcut1)
    shortcut2 = Conv1D(filters=lstm.shape[-1], kernel_size=1, padding="same", strides=1)(x)
    shortcut2 = BatchNormalization()(shortcut2)
    
    # Concatenate the shortcuts to the main path
    x = Concatenate(axis=-1)([lstm, shortcut1, shortcut2])
    x = Activation('relu')(x)
    
    # Global Pooling as an additional summarizer before the dense layer
    x_avg = GlobalAveragePooling1D()(x)
    x_max = GlobalMaxPooling1D()(x)
    x = Concatenate(axis=-1)([x_avg, x_max])
    
    # Dense Layers for Classification
    x = Dense(128, activation='relu')(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    # Build Model
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Run the model on GPU if available
with tf.device('/GPU:0'):
    model_1DCNN_LSTM_ResNet = build_model((5, 8000), num_classes)  # Adjusted input shape to have 3 dimensions
    model_1DCNN_LSTM_ResNet.summary()
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)
    
    # Model Checkpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint("model_1DCNN_LSTM_ResNet.h5", save_best_only=True, monitor='val_accuracy', mode='max')
    
    # Train the model
    history_1DCNN_LSTM_ResNet = model_1DCNN_LSTM_ResNet.fit(X_train, y_train, batch_size=64, epochs=100, 
                                                             validation_data=(X_valid, y_valid),
                                                             callbacks=[early_stopping,model_checkpoint])
    
    
DCNN_LSTM_resNet_acc = history_1DCNN_LSTM_ResNet.history['accuracy']
DCNN_LSTM_resNet_val = history_1DCNN_LSTM_ResNet.history['val_accuracy']
plt.figure()
plt.plot(history_1DCNN_LSTM_ResNet.history['accuracy'], label='Training')
plt.plot(history_1DCNN_LSTM_ResNet.history['val_accuracy'], label='Validation')
plt.title("Plot Accuracy Training and Validation")
plt.legend()
plt.show()

display(HTML('<hr>'))
print("--------------test set----------------")
# check test set
y_pred = model_1DCNN_LSTM_ResNet.predict(X_test,verbose = 0)
y_pred_bool = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_bool)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
report = classification_report(y_test, y_pred_bool,labels=label, output_dict=True)
df_report = pd.DataFrame(report).transpose()
display(df_report.round(3))

display(HTML('<hr>'))

print("--------------validate----------------")
# check validate set
y_pred = model_1DCNN_LSTM_ResNet.predict(X_valid,verbose = 0)
y_pred_bool = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_valid, y_pred_bool)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
report = classification_report(y_valid, y_pred_bool,labels=label, output_dict=True)
df_report = pd.DataFrame(report).transpose()
display(df_report.round(3))

display(HTML('<hr>'))
# check train set
print("----------------train----------------")
y_pred = model_1DCNN_LSTM_ResNet.predict(X_train,verbose = 0)
y_pred_bool = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_train, y_pred_bool)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
report = classification_report(y_train, y_pred_bool,labels=label, output_dict=True)
df_report = pd.DataFrame(report).transpose()
display(df_report.round(3))

# Final result.
test_loss_1DCNN_LSTM_ResNet, test_acc_1DCNN_LSTM_ResNet = model_1DCNN_LSTM_ResNet.evaluate(X_test, y_test)
print('Final model has loss of test set is: {} and accuracy is: {}'.format(test_loss_1DCNN_LSTM_ResNet,test_acc_1DCNN_LSTM_ResNet))
val_loss_1DCNN_LSTM_ResNet, val_acc_1DCNN_LSTM_ResNet = model_1DCNN_LSTM_ResNet.evaluate(X_valid, y_valid)
print('Final model has loss of validation set is: {} and accuracy is: {}'.format(val_loss_1DCNN_LSTM_ResNet,val_acc_1DCNN_LSTM_ResNet))