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

    void fit(const arma::mat &data, const arma::mat &target, double learningRate, int EPOCHS);

    std::vector<double> predict(const arma::mat &data);

private:
    std::vector<DenseLayer> layers;

    arma::mat fullForwardPass(arma::mat input);

    arma::mat fullBackwardPass(arma::mat grad);

    void applyGradients(const double learningRate);
};


#endif //NEURAL_NET_MODEL_H
