#There are 10k images in total, with 61 images of cracked surface
#pic dimesions are (227,227,3)

import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import yaml
import random
import mlflow
import mlflow.tensorflow
from PIL import Image
import matplotlib.image
from scipy.ndimage import rotate
import glob
import cv2

def ssim_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.double)
    y_pred = tf.cast(y_pred, dtype=tf.double)
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err

def train_data_generator(data, batch_size=64, shuffle=True):
    num_samples = len(data)
    while True:
        if shuffle:
            np.random.shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_data = data[offset:offset+batch_size]
            images = []
            for image_path in batch_data:
                image = cv2.imread(image_path)
                if np.random.random() < 0.3: 
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    hue_shift = np.random.uniform(-130, 130)
                    sat_scale = np.random.uniform(0.5, 1.5)
                    image[:,:,0] = (image[:,:,0] + hue_shift) % 180
                    image[:,:,1] = np.clip(image[:,:,1] * sat_scale, 0, 255)
                    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
                else:
                    pass
                images.append(image)
            X_batch = np.array(images)/255.0
            yield X_batch, X_batch

def load_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    batch_size= config['batch_size']

    train_data_dir= config['train_dir']
    train_data_files = glob.glob(os.path.join(train_data_dir, '*.jpg'))
    train_dataset= train_data_generator(train_data_files, batch_size= batch_size)

    test_data_dir= config['test_dir']
    test_data_files = glob.glob(os.path.join(test_data_dir, '*.jpg'))
    test_data_files.sort()

    test_dataset= []
    for fn in test_data_files:
        img= imread(fn)
        test_dataset.append(img)
    test_dataset=np.array(test_dataset)/255.0

    print(" TRAIN AND TEST DATA LOADED AND NORMALIZED !!!!!!!!")

    return  train_dataset, test_dataset

def build_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = tf.keras.Sequential()
    for layer_config in config['layers']:
        layer_type = getattr(tf.keras.layers, layer_config.pop('type'))
        model.add(layer_type(**layer_config))

    opt = keras.optimizers.Adam(learning_rate=config['autoencoder']['lr'])
    model.compile(optimizer=opt, loss=ssim_loss, metrics= config['autoencoder']['metrics'])
    
    return model

def train_model(train_dataset, test_dataset, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    autoencoder=build_model(config_path=config_path)
    batch_size= config['batch_size']
    early_stop= EarlyStopping(monitor= config['model']['early_stop']['monitor'], patience= config['model']['early_stop']['patience'])
    checkpoint= ModelCheckpoint(config['model']['checkpoint']['filepath'], monitor=config['model']['checkpoint']['monitor'], verbose= config['model']['checkpoint']['verbose'], save_best_only= config['model']['checkpoint']['save_best'], mode=config['model']['checkpoint']['mode'])
    callback_list= [early_stop,checkpoint]
    train_data_len= len(glob.glob(os.path.join(config['train_dir'], '*.jpg')))
    autoencoder.fit(train_dataset, epochs= config['model']['fit']['epochs'], steps_per_epoch= train_data_len // batch_size, verbose=config['model']['fit']['verbose'], validation_data=(test_dataset,test_dataset), callbacks= callback_list)

    return autoencoder

def calculate_accuracy_and_threshold(y, y_predicted, accuracy_type):
    accuracy_list = []
    for i in range(len(y)):
        if accuracy_type == 'mse':
            accuracy_value = mse(y[i], y_predicted[i])
        elif accuracy_type == 'ssim':
            accuracy_value = ssim(y[i], y_predicted[i], multichannel=True)
        else:
            raise ValueError("Invalid accuracy type. Allowed values: 'mse' or 'ssim'")
        accuracy_list.append(accuracy_value)

    accuracy_anomaly= accuracy_list[:66]
    n=5
    sorted_accuracy = sorted(accuracy_anomaly)
    accuracy_threhold = sorted_accuracy[-n] if accuracy_type == 'ssim' else sorted_accuracy[n]

    return accuracy_list, accuracy_threhold

def predict(test, test_predicted, accuracy_type):
    accuracy_list, accuracy_threshold = calculate_accuracy_and_threshold(test, test_predicted, accuracy_type=accuracy_type)

    if accuracy_type == 'mse':
        prediction_list = [0 if i < accuracy_threshold else 1 for i in accuracy_list]
    elif accuracy_type == 'ssim':
        prediction_list = [0 if i > accuracy_threshold else 1 for i in accuracy_list]
    else:
        raise ValueError("Invalid accuracy type. Allowed values: 'mse' or 'ssim'")
    
    return prediction_list

def reconstruct_images(y, y_predicted, metric_list, num_of_images=5, anomaly_images=True):
    black_line= np.zeros((1,227,3))

    if anomaly_images:
        image_range=(0,62)
    else:
        image_range=(69,2300)

    for i in range(num_of_images):
        num= random.randint(image_range[0],image_range[1])
        input_anomaly= y[num]
        output_anomaly=y_predicted[num]
        checkout= np.concatenate((input_anomaly, black_line,output_anomaly), axis=0)
        name= str(num) + '_' + str(metric_list[num]) + '.png'
        print("saved: ", i)
        mlflow.log_image(checkout, name)

def log_filter_and_dilation(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    filter_list = []
    dilation_list= []
    for layer in config['layers']:
        if layer['type'] == 'Conv2D':
            filter_list.append(layer['filters'])
            dilation_list.append(layer['dilation_rate'])
        else:
            pass 

    mlflow.log_param('filters', filter_list)
    mlflow.log_param('dilations', dilation_list)    

def label_test_data(num_of_anomaly_pictures=66):
    actual=[]
    for i in range(num_of_anomaly_pictures):
        actual.append(1)
    for i in range(2470):
        actual.append(0)
    return actual

def log_error_df(ssim_list, mse_list, actual, ssim_prediction_list, mse_prediction_list):
    error_df = pd.DataFrame({
        'SSIM': ssim_list,
        'MSE': mse_list,
        'Actual': actual,
        'SSIM_Prediction': ssim_prediction_list,
        'MSE_Prediction': mse_prediction_list
    })

    error_df.to_csv('Error.csv', index=False)
    mlflow.log_artifact('Error.csv')

def log_confusion_matrix(actual, ssim_prediction_list, mse_prediction_list):
    for name, prediction_list in [("SSIM", ssim_prediction_list), ("MSE", mse_prediction_list)]:
        print(f"CM based on {name}:  ", confusion_matrix(actual, prediction_list))
        cm = confusion_matrix(actual, prediction_list)
        tn, fn, fp, tp = cm.ravel()
        for metric_name, value in zip([f"{name}_tn", f"{name}_fp", f"{name}_fn", f"{name}_tp"], [tn, fn, fp, tp]):
            mlflow.log_metric(metric_name, value)

def log_classification_report(actual, prediction_list):
    report = classification_report(actual, prediction_list, output_dict=True)
    for label, metrics in report.items():
        if label == "accuracy":
            continue
        for metric_name, value in metrics.items():
            metric_name = f"{label}_{metric_name}"
            mlflow.log_metric(metric_name, value)




mlflow.set_tracking_uri('http://localhost:5001')
mlflow.set_experiment("project_two")
print("setting uri")

mlflow.tensorflow.autolog()
os.environ["CUDA_VISIBLE_DEVICES"]="4"
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
gpu_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

config_dir= '/mnt/working/poddarm/config.yml'
mlflow.log_artifact(config_dir, artifact_path= "configs")

# ae= build_model(config_dir)
# print(ae.summary())
log_filter_and_dilation(config_dir)


#CHANGE VARIABLE NAMES
train_data,test_data= load_data(config_dir)
print("data loaded")

autoencoder = train_model(train_data, test_data[70:], config_dir)
print("model trained")
y_predicted= autoencoder.predict(test_data)



#Evaluate model

actual= label_test_data()
mse_list, mse_threshold= calculate_accuracy_and_threshold(test_data, y_predicted, accuracy_type='mse')
ssim_list, ssim_threshold= calculate_accuracy_and_threshold(test_data, y_predicted,  accuracy_type='ssim')

mlflow.log_param('ssim_threshold', ssim_threshold)
mlflow.log_param('mse_threshold', mse_threshold)

reconstruct_images(test_data, y_predicted, ssim_list, 5, True)
reconstruct_images(test_data, y_predicted, ssim_list, 5, False)


print("evalutaing model...")

#predicting on basis of ssim
ssim_prediction_list=predict(test_data,y_predicted,'ssim')
mse_prediction_list=predict(test_data,y_predicted,'mse')

#creating a scatterplot of ssim_score
plt.scatter(list(range(66)), ssim_list[:66], c='r')
plt.scatter(list(range(66, 500)), ssim_list[66:500], c='b')
plt.savefig("ssim_scatter.png")
mlflow.log_artifact("ssim_scatter.png")
plt.close()

#creating a scatterplot of ssim_score
plt.scatter(list(range(66)), mse_list[:66], c='r')
plt.scatter(list(range(66, 500)), mse_list[66:500], c='b')
plt.savefig("mse_scatter.png")
mlflow.log_artifact("mse_scatter.png")
plt.close()



print("creating an evaluation file")
log_error_df(ssim_list, mse_list, actual, ssim_prediction_list, mse_prediction_list)
print("Done!")


log_confusion_matrix(actual,ssim_prediction_list,mse_prediction_list)
log_classification_report(actual,ssim_prediction_list)










# ae = build_model(config_dir)
# ae.load_weights("/mnt/working/poddarm/saved_model/weights-improvements-29-0.9531.hdf5")
