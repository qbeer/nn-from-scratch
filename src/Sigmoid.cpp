//
// Created by olaralex on 2019.05.25..
//

#include "Sigmoid.h"

arma::Mat<double> Sigmoid::call(arma::Mat<double> value) {
    return arma::pow(arma::ones(value.size()).t() + arma::exp(-value), -1);
}

arma::Mat<double> Sigmoid::derivative(arma::Mat<double> value) {
    return this->call(value) * (arma::ones(value.size()).t() - this->call(value));
}
