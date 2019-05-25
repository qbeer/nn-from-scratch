//
// Created by olaralex on 2019.05.25..
//

#ifndef NEURAL_NET_LINEARACTIVATION_H
#define NEURAL_NET_LINEARACTIVATION_H


#include <armadillo>
#include "ActivationFunction.h"

class LinearActivation : public ActivationFunction {
public:
    arma::Mat<double> call(arma::Mat<double> value);

    arma::Mat<double> derivative(arma::Mat<double> value);
};


#endif //NEURAL_NET_LINEARACTIVATION_H
