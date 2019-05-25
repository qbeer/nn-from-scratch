#include <utility>

//
// Created by olaralex on 2019.05.23..
//

#include "DenseLayer.h"
#include "LinearActivation.h"
#include "ReLU.h"
#include "Sigmoid.h"

DenseLayer::DenseLayer(unsigned int inputSize, unsigned int units,
                       std::string activationName) : inputSize(inputSize), units(units) {
    arma::arma_rng::set_seed(100);
    this->weights = arma::randn(inputSize, units) * sqrt(2. / (double) (inputSize + units));
    this->biases = arma::randn(units) * sqrt(2. / (double) (inputSize + units));
    if (activationName == "relu") {
        this->activationFunction = new ReLU();
    } else if (activationName == "sigmoid") {
        this->activationFunction = new Sigmoid();
    } else {
        this->activationFunction = new LinearActivation();
    }
}

arma::Mat<double> DenseLayer::feedForward(const arma::Mat<double> &inputValue) {
    this->input = inputValue;
    //std::cout << "Input size : " << this->input.size() << std::endl;
    this->activationInputs = this->input * this->weights + this->biases.t();
    //std::cout << "Activation inputs size : " << this->activationInputs.size() << std::endl;
    this->output = this->activationFunction->call(this->activationInputs);
    //std::cout << "Output size : " << this->output.size() << std::endl;
    return this->output;
}

arma::Mat<double> DenseLayer::backProp(arma::Mat<double> &previous_gradient) {
    //std::cout << "Prev. gradient size : " << previous_gradient.size() << std::endl;
    this->activationDerivative = this->activationFunction->derivative(this->activationInputs);
    //std::cout << "Activation deriv. size : " << this->activationDerivative.size() << std::endl;
    this->intermediateGradient = previous_gradient % this->activationDerivative.t();
    //std::cout << "Intermediate gradient size : " << this->intermediateGradient.size() << std::endl;
    this->weightsGradient = this->intermediateGradient * this->input;
    //std::cout << "Weights gradient size : " << this->weightsGradient.size() << std::endl;
    //std::cout << "Gradient size : " << this->gradient.size() << std::endl << std::endl;
    return this->weights * this->intermediateGradient;
}

void DenseLayer::applyGrads(double learningRate) {
    this->weights -= learningRate * this->weightsGradient.t();
    this->biases -= learningRate * this->intermediateGradient;
}
