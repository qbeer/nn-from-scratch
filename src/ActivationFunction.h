//
// Created by olaralex on 2019.05.23..
//

#include <vector>
#include <armadillo>

#ifndef NEURAL_NET_ACTIVATIONFUNCTION_H
#define NEURAL_NET_ACTIVATIONFUNCTION_H

/*
 * Basically linear activation, therefore no activation!
 * */
class ActivationFunction {
public:
    virtual arma::Mat<double> call(arma::Mat<double> value) { return value; };

    virtual arma::Mat<double> derivative(arma::Mat<double> value) { return value; };
};


#endif //NEURAL_NET_ACTIVATIONFUNCTION_H
