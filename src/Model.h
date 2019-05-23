//
// Created by olaralex on 2019.05.23..
//

#ifndef NEURAL_NET_MODEL_H
#define NEURAL_NET_MODEL_H

#include <vector>
#include "DenseLayer.h"
#include <armadillo>
#include <iostream>

class Model {
public:
    Model();
    void add(DenseLayer layer);
    void fit(arma::mat data, arma::mat target, double learningRate, int EPOCHS);

private:
    std::vector<DenseLayer> layers;
    arma::mat gradient;
    arma::mat feedValue;
    arma::mat fullForwardPass(const arma::mat &input);
    arma::mat fullBackwardPass(const arma::mat &input);
    void applyGradients(const double learningRate);
};


#endif //NEURAL_NET_MODEL_H
