//
// Created by olaralex on 2019.05.23..
//

#include "Model.h"

Model::Model() {
}

void Model::add(DenseLayer layer) {
    this->layers.push_back(layer);
}

void Model::fit(const arma::mat &data, const arma::mat &target, double learningRate, int EPOCHS) {
    for (int i = 0; i < EPOCHS; ++i) {
        for (int row_ind = 0; row_ind < data.n_rows; ++row_ind) {
            arma::Mat<double> output = this->fullForwardPass(data.submat(row_ind, 0, row_ind, data.n_cols - 1));
            arma::mat gradient = this->fullBackwardPass(-2 * (target.at(row_ind) / 10. - output)); // percentages

            if (row_ind % 10 == 0) {
                std::cout << "Loss : " << arma::accu(arma::square(output - target.at(row_ind) / 10.)) << "\tEPOCH : "
                          << i
                          << "\tGradient size : " << arma::accu(arma::sqrt((arma::square(gradient)))) << std::endl;
            }

            this->applyGradients(learningRate);
        }
    }
}

std::vector<double> Model::predict(const arma::mat &data) {
    std::vector<double> predictions(data.n_rows);
    for (int row_ind = 0; row_ind < data.n_rows; ++row_ind) {
        arma::Mat<double> output = this->fullForwardPass(data.submat(row_ind, 0, row_ind, data.n_cols - 1)) * 10.;
        predictions.push_back(arma::floor(output).eval()(0, 0));
    }
    return predictions;
}

arma::mat Model::fullForwardPass(arma::mat input) {
    for (auto &layer : this->layers) {
        input = layer.feedForward(input);
    }
    return input;
}

arma::mat Model::fullBackwardPass(arma::mat grad) {
    auto rit = this->layers.rbegin();
    for (; rit != this->layers.rend(); ++rit) {
        grad = (*rit).backProp(grad);
    }
    return grad;

}

void Model::applyGradients(const double learningRate) {
    for (auto &layer : this->layers) {
        layer.applyGrads(learningRate);
    }
}

