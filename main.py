from vggish_input import waveform_to_examples, wavfile_to_examples
import numpy as np
import tensorflow as tf
from keras.models import load_model
import vggish_params
from pathlib import Path
import ubicoustics
from pydub import AudioSegment
import wget
import sox
import librosa
import os
import json


def trim_output(selected_file, trim_list, whole_length, pre_decor):
    duration = librosa.get_duration(filename=selected_file)
    for index, part in enumerate(trim_list):
        start = part[0]
        end = part[1]
        start_time = (start / whole_length) * duration
        end_time = (end / whole_length) * duration
        tfm = sox.Transformer()
        tfm.trim(start_time, end_time)
        split_list = selected_file.split('/')
        if len(split_list) is not 0:
            split_list = split_list[len(split_list) - 1]
        output_file_name = pre_decor + '/' + split_list.split('.')[0] + '_out' + str(index) + '.wav'
        tfm.build_file(selected_file, output_file_name)


###########################
# Download model, if it doesn't exist
###########################
def model_prepare(model_filename):
    MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
    MODEL_PATH = "models/example_model.hdf5"
    print("=====")
    print("Checking model... ")
    print("=====")
    ubicoustics_model = Path(model_filename)
    if (not ubicoustics_model.is_file()):
        print("Downloading example_model.hdf5 [867MB]: ")
        wget.download(MODEL_URL, MODEL_PATH)


###########################
# Load Model
###########################
def model_ready(model_file_name):
    context = ubicoustics.everything
    context_mapping = ubicoustics.context_mapping
    selected_context = 'everything'
    model = load_model(model_file_name)
    context = context_mapping[selected_context]
    graph = tf.get_default_graph()

    label = dict()
    for k in range(len(context)):
        label[k] = context[k]
    return model, graph, label


def use_model(model_filename, from_file_path, from_file_name, status, model, graph, label):
    storage_path = 'storage'
    output_dir = 'output_cough'
    # selected_file = 'example.wav'
    selected_file = storage_path + '/' + from_file_name.split('.')[0] + '_transfered.wav'
    cmd = 'ffmpeg -i ' + from_file_path + ' -c:a pcm_f32le ' + selected_file
    os.system(cmd)

    ###########################
    # Read Wavfile and Make Predictions
    ###########################
    x = wavfile_to_examples(selected_file)

    with graph.as_default():
        x = x.reshape(len(x), 96, 64, 1)
        trim_list = []
        predictions = model.predict(x)

        cough_list = []
        start_flag = True
        for k in range(len(predictions)):
            prediction = predictions[k]
            m = np.argmax(prediction)
            print("Prediction: %s (%0.2f)" % (ubicoustics.to_human_labels[label[m]], prediction[m]))
            if ubicoustics.to_human_labels[label[m]] == 'Coughing':
                if start_flag:
                    start_flag = False
                    cough_list.append(k)
                else:
                    start_flag = True
                    cough_list.append(k)
                    trim_list.append(cough_list)
                    cough_list = []
            elif start_flag == False:
                start_flag = True
                cough_list.append(k)
                trim_list.append(cough_list)
                cough_list = []

    trim_output(selected_file, trim_list, len(predictions), storage_path + '/' + output_dir + '/' + status)


if __name__ == '__main__':
    model_filename = "models/example_model.hdf5"
    public_data_path = '../../data/public_dataset'
    dirs = os.listdir(public_data_path)
    count = 0
    status = 'unknown'
    absolute_path = ''
    flag = False

    print("=== start model preparation ===")
    model_prepare(model_filename)
    print("=== finish model preparation ===")
    model, graph, label = model_ready(model_filename)
    print("=== finish model ready ===")

    for dir in dirs:
        count = count + 1
        print("file process count: ", count)
        print("file: ", dir)
        # json file
        if dir.split('.')[1] == 'json':
            status = 'unknown'

            with open(public_data_path + '/' + dir, 'r') as reader:
                jf = json.loads(reader.read())

                if float(jf["cough_detected"]) <= 0.5:
                    flag = True
                    continue

                if 'status' in jf:
                    status = jf['status']

        elif dir.split('.')[1] == 'webm':
            if flag is True:
                flag = False
                continue

            use_model(model_filename, public_data_path + '/' + dir, dir, status, model, graph, label)

        elif dir.split('.')[1] == 'ogg':
            continue

        # control times
        # if count > 20:
        #     break
    print("finish")
