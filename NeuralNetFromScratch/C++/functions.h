#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <Eigen/Dense>

using namespace Eigen;

namespace functions {
    inline VectorXd sigmoid(const VectorXd &v) {
        return 1.0 / (1.0 + (-v.array()).exp());
    }

    inline VectorXd sigmoid_derivative(const VectorXd &v) {
        VectorXd sig = sigmoid(v);
        return sig.array() * (1.0 - sig.array());
    }

    inline VectorXd tanh(const VectorXd &v) {
        auto e_pos = v.array().exp();
        auto e_neg = (-v.array()).exp();
        return (e_pos - e_neg) / (e_pos + e_neg);
    }
    inline VectorXd tanh_derivative(const VectorXd &v) {
        auto t = tanh(v).array();
        return (1.0 - t * t).matrix();
    }
    inline VectorXd reLU(const VectorXd &v) {
        return v.array().cwiseMax(0.0);
    }
    inline VectorXd reLU_derivative(const VectorXd &v) {
        return (v.array() > 0.0).cast<double>();
    }
    inline VectorXd linear(const VectorXd &v) {
        return v; // No activation
    }

    inline VectorXd linear_derivative(const VectorXd &v) {
        return VectorXd::Ones(v.size());
    }

    inline double error_function(const VectorXd &output, const VectorXd &target) {
        return 0.5 * (output - target).squaredNorm();
    }

   inline  VectorXd error_function_derivative(const VectorXd &output, const VectorXd &target) {
        return output - target;
    }

}

#endif // FUNCTIONS_H