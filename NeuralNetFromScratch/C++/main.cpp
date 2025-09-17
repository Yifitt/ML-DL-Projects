#include <iostream>
#include <chrono>
#include <ctime>
#include "layer.h"
#include "functions.h"
#include "neural_network.h"
#include "utils.h"

using namespace Eigen;

const std::string mnist_train_data_path = "data/MNIST/raw/train-images.idx3-ubyte";
const std::string mnist_train_label_path = "data/MNIST/raw/train-labels.idx1-ubyte";
const std::string mnist_test_data_path = "data/MNIST/raw/t10k-images.idx3-ubyte";
const std::string mnist_test_label_path = "data/MNIST/raw/t10k-labels.idx1-ubyte";


int main() {
    using clock = std::chrono::high_resolution_clock;
    srand(time(nullptr));

    std::vector<VectorXd> train_dataset;
    std::vector<VectorXd> label_train_dataset;

    std::vector<VectorXd> test_dataset;
    std::vector<VectorXd> label_test_dataset;

    utils::read_mnist_train_data(mnist_train_data_path, train_dataset);
    utils::read_mnist_train_label(mnist_train_label_path, label_train_dataset);

    utils::read_mnist_test_data(mnist_test_data_path, test_dataset);
    utils::read_mnist_test_label(mnist_test_label_path, label_test_dataset);


    Layer hidden_layer1(784, 64, functions::sigmoid,functions::sigmoid_derivative,"sigmoid");

    Layer output_layer1(64, 10,functions::sigmoid , functions::sigmoid_derivative,"sigmoid");


    NeuralNetwork nn1({hidden_layer1, output_layer1});

    auto trainTime1 = clock::now();
    nn1.train(train_dataset, label_train_dataset, 0.01, 1);
    auto trainTimeEnd1 = clock::now();
    std::chrono::duration<float> deltaTime1 = trainTimeEnd1  - trainTime1;
    std::cout<<"Training took "<<deltaTime1.count()<<" seconds."<<std::endl;

    nn1.test(test_dataset, label_test_dataset);

    nn1.save_model("my_model.json");



    return 0;
}