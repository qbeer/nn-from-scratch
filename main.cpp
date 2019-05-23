#include "src/DenseLayer.h"
#include "src/ReLU.h"
#include "src/DataReader.h"
#include "src/Model.h"
#include <armadillo>
#include <iostream>

int main() {
    DataReader reader("../src/data/mnist.csv");

    arma::mat data = reader.getData();
    arma::mat labels = reader.getLabels();

    Model model;
    model.add(DenseLayer(784, 512, ReLU()));
    model.add(DenseLayer(512, 256, ReLU()));
    model.add(DenseLayer(256, 64, ReLU()));
    model.add(DenseLayer(64, 1, ReLU()));

    model.fit(data, labels, 1e-6, 1);

    return 0;
}