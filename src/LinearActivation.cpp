//
// Created by olaralex on 2019.05.25..
//

#include "LinearActivation.h"

arma::Mat<double> LinearActivation::call(arma::Mat<double> value) {
    return value;
}

arma::Mat<double> LinearActivation::derivative(arma::Mat<double> value) {
    return arma::ones(value.size());
}
