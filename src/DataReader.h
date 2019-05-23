//
// Created by olaralex on 2019.05.23..
//

#ifndef NEURAL_NET_DATAREADER_H
#define NEURAL_NET_DATAREADER_H

#include <string>
#include <armadillo>

class DataReader {
public:
    DataReader(std::string filePath);

    arma::Mat<double> getData();

    arma::Mat<double> getLabels();

private:
    void readFile();

    std::string filePath;
    arma::mat data;
    arma::mat labels;
};


#endif //NEURAL_NET_DATAREADER_H
