#ifndef LAYER_H
#define LAYER_H


#include <functional>
#include "json.hpp"
#include "functions.h"

using namespace Eigen;

class Layer {
private:
    MatrixXd weights;
    VectorXd biases;
    VectorXd neurons_values;
    VectorXd neurons_values_activate;
    std::function<VectorXd(const VectorXd&)> activation_function;
    std::function<VectorXd(const VectorXd&)> activation_function_derivative;

    std::string activation_name;

    VectorXd delta;

public:
    Layer(int input_size_neurons, int neurons,std::function<VectorXd(const VectorXd&)> activation_function,
        std::function<VectorXd(const VectorXd&)> activation_function_derivative,const std::string& activation_name = "unknown") :
    activation_function(activation_function),activation_function_derivative(activation_function_derivative),activation_name(activation_name)
    {
        weights = MatrixXd::Random(input_size_neurons, neurons);
        biases = VectorXd::Zero(neurons);
    }


    void forward(const VectorXd &input) {
        VectorXd Y = weights.transpose() * input + biases;
        neurons_values = Y;
        neurons_values_activate = activation_function(Y);
    }


    VectorXd get_neurons_values_activate() {
        return neurons_values_activate;
    }

    void set_delta(const VectorXd& delta) {
        this->delta = delta;
    }

    VectorXd get_delta() {
        return delta;
    }

    MatrixXd get_weights() {
        return weights;
    }

    void update_weights(const VectorXd& input, double learning_rate) {
        weights -= learning_rate * input * delta.transpose();
        biases -= learning_rate * delta;
    }

    VectorXd derivative_of_activation_function() {
        return activation_function_derivative(neurons_values);
    }
    nlohmann::json to_json() const {
        nlohmann::json j;

        // Convert weights matrix to 2D vector
        std::vector<std::vector<double>> weights_vec(weights.rows());
        for (int i = 0; i < weights.rows(); i++) {
            weights_vec[i].resize(weights.cols());
            for (int j = 0; j < weights.cols(); j++) {
                weights_vec[i][j] = weights(i, j);
            }
        }

        // Convert biases vector
        std::vector<double> biases_vec(biases.size());
        for (int i = 0; i < biases.size(); i++) {
            biases_vec[i] = biases(i);
        }

        j["weights"] = weights_vec;
        j["biases"] = biases_vec;
        j["input_size"] = weights.rows();
        j["output_size"] = weights.cols();
        j["activation"] = activation_name;

        return j;
    }
};

#endif // LAYER_H