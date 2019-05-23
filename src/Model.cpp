//
// Created by olaralex on 2019.05.23..
//

#include "Model.h"

Model::Model() {
}

void Model::add(DenseLayer layer) {
    this->layers.push_back(layer);
}

void Model::fit(arma::mat data, arma::mat target, double learningRate, int EPOCHS) {
    for (int i = 0; i < EPOCHS; ++i) {
        for (int row_ind = 0; row_ind < data.n_rows; ++row_ind) {
            arma::Mat<double> row = data.submat(row_ind, 0, row_ind, data.n_cols - 1);
            //std::cout << "\nStarting full forward pass\n";
            arma::Mat<double> output = this->fullForwardPass(row);
            double loss = arma::accu(arma::square(output - target.at(row_ind)));
            arma::Mat<double> lossDerivative = -2 * (output - target.at(row_ind));

            //std::cout << "\nStarting full backward pass...\n";
            arma::Mat<double> gradient = this->fullBackwardPass(lossDerivative);

            if (row_ind % 10 == 0) {
                std::cout << "Loss : " << loss << "\t EPOCH : " << i << std::endl;
            }

            this->applyGradients(learningRate);
        }
    }
}

arma::mat Model::fullForwardPass(const arma::mat &input) {
    arma::mat input_ = input;
    for (auto &layer : this->layers) {
        input_ = layer.feedForward(input_);
    }
    return input_;
}

arma::mat Model::fullBackwardPass(const arma::mat &input) {
    arma::mat grad = input;
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

