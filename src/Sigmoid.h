//
// Created by olaralex on 2019.05.25..
//

#ifndef NEURAL_NET_SIGMOID_H
#define NEURAL_NET_SIGMOID_H


#include "ActivationFunction.h"

class Sigmoid : public ActivationFunction {
public:
    arma::Mat<double> call(arma::Mat<double> value);

    arma::Mat<double> derivative(arma::Mat<double> value) override;
};


#endif //NEURAL_NET_SIGMOID_H
