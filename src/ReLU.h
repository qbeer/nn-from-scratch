//
// Created by olaralex on 2019.05.23..
//

#ifndef NEURAL_NET_RELU_H
#define NEURAL_NET_RELU_H


#include "ActivationFunction.h"

class ReLU : public ActivationFunction {
public:
    arma::Mat<double> call(arma::Mat<double> value) override;

    arma::Mat<double> derivative(arma::Mat<double> value) override;
};


#endif //NEURAL_NET_RELU_H