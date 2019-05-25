//
// Created by olaralex on 2019.05.23..
//

#define EPSILON 1e-12

#include "ReLU.h"

arma::Mat<double> ReLU::call(arma::Mat<double> value) {
    return arma::clamp(value, 0., arma::abs(value).max() + EPSILON);
}

arma::Mat<double> ReLU::derivative(arma::Mat<double> value) {
    return arma::clamp(arma::sign(value), 0., 1.);
}
