//
// Created by olaralex on 2019.05.23..
//

#include "DataReader.h"

DataReader::DataReader(std::string filePath) {
    this->filePath = filePath;
    this->readFile();

}

arma::Mat<double> DataReader::getData() {
    return this->data;
}

arma::Mat<double> DataReader::getLabels() {
    return this->labels;
}

void DataReader::readFile() {
    std::string delimeter = ",";
    std::ifstream csv(this->filePath);
    std::vector<std::vector<double>> datas;

    for (std::string line; std::getline(csv, line);) {

        std::vector<double> data;

        // split string by delimeter
        int start = 0U;
        int end = line.find(delimeter);
        while (end != std::string::npos) {
            data.push_back(std::stod(line.substr(start, end - start)) / 255.);
            start = end + delimeter.length();
            end = line.find(delimeter, start);
        }
        data.push_back(std::stod(line.substr(start, end)));
        datas.push_back(data);
    }

    arma::mat data_mat = arma::zeros<arma::mat>(datas.size(), datas[0].size() - 1);
    arma::mat target_mat = arma::zeros<arma::mat>(datas.size(), 1);

    for (int i = 0; i < datas.size(); i++) {
        std::vector<double>::const_iterator first = datas[i].begin() + 1;
        std::vector<double>::const_iterator last = datas[i].end();
        arma::mat r(std::vector<double>(first, last));
        data_mat.row(i) = r.t();
    }

    std::vector<double> target(datas.size());
    for (int i = 0; i < datas.size(); i++) {
        target[i] = datas[i][0] * 255.;
    }

    target_mat = arma::mat(target);

    this->data = std::move(data_mat);
    this->labels = std::move(target_mat);
}
