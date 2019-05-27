#include "src/DenseLayer.h"
#include "src/ReLU.h"
#include "src/Sigmoid.h"
#include "src/DataReader.h"
#include "src/Model.h"
#include <armadillo>
#include <iostream>

int main() {
    DataReader reader("../src/data/mnist.csv");

    arma::mat data = reader.getData();
    arma::mat labels = reader.getLabels();

    Model model;
    model.add(DenseLayer(784, 500, "relu"));
    model.add(DenseLayer(500, 300, "relu"));
    model.add(DenseLayer(300, 200, "relu"));
    model.add(DenseLayer(200, 100, "relu"));
    model.add(DenseLayer(100, 1, "sigmoid"));

    model.fit(data, labels, 1e-1, 100);

    std::vector<double> predictions = model.predict(data);
    std::vector<double> actual_labels = std::vector<double>(labels.memptr(), labels.memptr() + labels.size() - 1);

    double matches = 0.;
    for(int i = 0; i < predictions.size(); ++i){
        if(predictions[i] == actual_labels[i]){
            matches++;
        }
    }

    std::cout << "Precision : " << 100.*matches / (double) predictions.size() << "%\n";

    return 0;
}