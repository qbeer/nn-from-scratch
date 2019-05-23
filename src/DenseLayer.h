//
// Created by olaralex on 2019.05.23..
//

#ifndef NEURAL_NET_DENSELAYER_H
#define NEURAL_NET_DENSELAYER_H


#include <array>
#include <iostream>
#include <cmath>
#include "ActivationFunction.h"
#include <armadillo>

class DenseLayer {
public:
    DenseLayer(const unsigned int inputSize, const unsigned int units, ActivationFunction activationFunction);

    arma::Mat<double> feedForward(const arma::Mat<double> &inputValue);

    arma::Mat<double> backProp(arma::Mat<double> &previous_gradient);

    void applyGrads(double learningRate);

private:
    const unsigned int inputSize;
    const unsigned int units;
    arma::Mat<double> weights;
    arma::Mat<double> biases;
    arma::Mat<double> activationInputs;
    arma::Mat<double> activationDerivative;
    arma::Mat<double> intermediateGradient;
    arma::Mat<double> biasGradient;
    arma::Mat<double> weightsGradient;
    arma::Mat<double> gradient;
    arma::Mat<double> input;
    arma::Mat<double> output;
    ActivationFunction activationFunction;
};

#endif //NEURAL_NET_DENSELAYER_H
