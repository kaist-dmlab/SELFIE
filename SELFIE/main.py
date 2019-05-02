import os, sys
from pathlib import Path
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from algorithm.default import *
from algorithm.activebias import *
from algorithm.coteaching import *
from algorithm.selfie import *
from reader import image_input_reader

def main():
    print("------------------------------------------------------------------------")
    print("This code trains Densnet(L={10,25,40}, k=12) using SELFIE in tensorflow-gpu environment.")
    print("\nDescription -----------------------------------------------------------")
    print("Please download datasets from our github before running command.")
    print("For SELFIE, the hyperparameter was set to be ""uncertainty threshold = 0.05"" and  ""history length=15.")
    print("For Training, we follow the same configuration in our paper")
    print("For Training, training_epoch = 100, batch = 128, initial_learning rate = 0.1 (decayed 50% and 75% of total number of epochs), use momentum of 0.9, warm_up=25, restart=2, ...")
    print("You can easily change the value in main.py")

    if len(sys.argv) != 8:
        print("Run Cmd: python main.py  gpu_id  data model_name  method_name  noise_type  noise_rate  log_dir")
        print("\nParamters -----------------------------------------------------------")
        print("1. gpu_id: gpu number which you want to use")
        print("2. data: {CIFAR-10, CIFAR-100, Tiny-ImageNet, ANIMAL-10N}")
        print("3. model_name: {DenseNet-10-12, DenseNet-25-12, DenseNet-40-12, VGG-19}")
        print("4. Method: {Default, ActiveBias, Coteaching, SELFIE}")
        print("5. noise_type: {pair, symmetry, none}")
        print("6. noist_rate: the rate which you want to corrupt (CIFAR-10, CIFAR-100, Tiny-ImageNet) or the true noise rate of dataset (ANIMAL-10N)")
        print("7. log_dir: log directory to save training loss/acc and test loss/acc")
        sys.exit(-1)

    # For user parameters
    gpu_id = int(sys.argv[1])
    data = sys.argv[2]
    model_name = sys.argv[3]
    method_name = sys.argv[4]
    noise_type = sys.argv[5]
    noise_rate = float(sys.argv[6])
    log_dir = sys.argv[7]

    # Data read
    datapath = str(Path(os.path.dirname((os.path.abspath(__file__))))) + "/dataset/" + data
    if os.path.exists(datapath):
        print("Dataset exists in ", datapath)
    else:
        print("Dataset doen't exist in ", datapath, ", please downloads and locates the data.")
        sys.exit(-1)

    # Parameters for data reading
    channel = 3
    if data == "CIFAR-10":
        num_train_files = 5
        num_train_images = 50000
        num_test_image = 10000
        width = 32
        height = 32
        num_classes = 10

    elif data == "CIFAR-100":
        num_train_files = 1
        num_train_images = 50000
        num_test_image = 10000
        width = 32
        height = 32
        num_classes = 100

    elif data == "Tiny-ImageNet":
        num_train_files = 4
        num_train_images = 100000
        num_test_image = 10000
        width = 64
        height = 64
        num_classes = 200

    elif data == "ANIMAL-10N":
        num_train_files = 1
        num_train_images = 50000
        num_test_image = 5000
        width = 64
        height = 64
        num_classes = 10

    input_reader = image_input_reader.ImageReader(data, datapath, num_train_files, num_train_images, num_test_image, width, height, channel, num_classes)

    # Parameters for SELFIE
    optimizer = 'momentum'
    total_epochs = 100
    batch_size = 128
    lr_boundaries = [int(np.ceil(num_train_images/batch_size)*50), int(np.ceil(num_train_images/batch_size)*75)]
    lr_values = [0.1, 0.02, 0.004]
    warm_up = 25
    threshold = 0.05
    queue_size = 15
    restart = 3

    if method_name == "Default":
        default(gpu_id, input_reader, model_name, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, noise_rate, noise_type, log_dir=log_dir)
    elif method_name == "ActiveBias":
        active_bias(gpu_id, input_reader, model_name, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, noise_rate, noise_type, warm_up, log_dir=log_dir)
    elif method_name == "Coteaching":
        coteaching(gpu_id, input_reader, model_name, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, noise_rate, noise_type, log_dir=log_dir)
    elif method_name == "SELFIE":
        selfie(gpu_id, input_reader, model_name, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, noise_rate, noise_type, warm_up, threshold, queue_size, restart=restart, log_dir=log_dir)

if __name__ == '__main__':
    print(sys.argv)
    main()
