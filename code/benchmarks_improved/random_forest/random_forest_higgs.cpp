#include "aoclda.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <cmath>

#ifndef HIGGS_CSV_PATH
#define HIGGS_CSV_PATH "/nobackup/wsgp73/HIGGS.csv"
#endif

bool load_higgs_csv(const std::string &path,
                    std::vector<std::vector<float>> &features,
                    std::vector<int> &labels) {
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "Could not open " << path << std::endl;
        return false;
    }

    std::cout << "Loading data from " << path << ". Moment later......." << std::endl;

    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string cell;
        
        std::getline(ss, cell, ',');
        labels.push_back(std::stoi(cell));

        std::vector<float> feature_row;
        while(std::getline(ss, cell, ',')) {
            feature_row.push_back(std::stof(cell));
        }
        features.push_back(feature_row);
    }
    return true;
}

int main() {
    std::cout << "AOCL-DA Decision Forest on HIGGS Dataset\n";

    std::vector<std::vector<float>> all_features_rm; // Row-major
    std::vector<int> all_labels;
    if (!load_higgs_csv(HIGGS_CSV_PATH, all_features_rm, all_labels)) {
        return 1;
    }

    if (all_features_rm.empty()) {
        std::cerr << "ERROR: No data loaded from CSV.\n";
        return 1;
    }

    int n_total = all_labels.size();
    int n_feat = all_features_rm[0].size();
    int n_cls = *std::max_element(all_labels.begin(), all_labels.end()) + 1;

    std::cout << "Dataset loaded: " << n_total << " samples, " << n_feat << " features.\n";

    int n_train = static_cast<int>(n_total * 0.8);
    int n_test = n_total - n_train;
    std::cout << "Training set size: " << n_train << "\n";
    std::cout << "Test set size: " << n_test << "\n";

    std::cout << "Transposing data to column-major format...\n";
    std::vector<float> Xtr_cm(n_train * n_feat);
    for (int i = 0; i < n_train; ++i) {
        for (int j = 0; j < n_feat; ++j) {
            Xtr_cm[j * n_train + i] = all_features_rm[i][j];
        }
    }

    std::vector<float> Xte_cm(n_test * n_feat);
    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_feat; ++j) {
            Xte_cm[j * n_test + i] = all_features_rm[n_train + i][j];
        }
    }
    
    int* y_tr_data = all_labels.data();
    int* y_te_data = all_labels.data() + n_train;


    da_handle f = nullptr;
    if (da_handle_init_s(&f, da_handle_decision_forest) != da_status_success) {
        std::cerr << "ERROR: da_handle_init_s\n";
        return 1;
    }
    da_options_set_int(f, "logging", 1);


    if (da_forest_set_training_data_s(f,
            /* n_rows = */ n_train,
            /* n_cols = */ n_feat,
            /* n_classes = */ n_cls,
            /* X = */ Xtr_cm.data(),
            /* stride = */ n_train,
            /* y = */ y_tr_data
        ) != da_status_success)
    {
        std::cerr << "ERROR: da_forest_set_training_data_s\n";
        da_handle_destroy(&f);
        return 1;
    }

    da_options_set_int   (f, "number of trees", 32);
    da_options_set_int   (f, "maximum depth", 40);
    da_options_set_int   (f, "node minimum samples", 10);
    da_options_set_string(f, "scoring function", "entropy");
    da_options_set_string(f, "task", "classification");
    da_options_set_int   (f, "seed", 42);
    da_options_set_int   (f, "maximum features", static_cast<int>(std::sqrt(n_feat)));


    std::cout << "Starting model training...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    if (da_forest_fit_s(f) != da_status_success) {
        std::cerr << "ERROR: da_forest_fit_s\n";
        da_handle_destroy(&f);
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Training time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    std::cout << "Predicting on test set...\n";
    std::vector<int> y_pred(n_test);
    if (da_forest_predict_s(f,
            /* n_rows = */ n_test,
            /* n_cols = */ n_feat,
            /* X = */ Xte_cm.data(),
            /* stride = */ n_test,
            /* y_pred = */ y_pred.data()
        ) != da_status_success)
    {
        std::cerr << "ERROR: da_forest_predict_s\n";
        da_handle_destroy(&f);
        return 1;
    }

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < n_test; ++i) {
        if (y_pred[i] == y_te_data[i]) {
            ++correct;
        }
    }
    std::cout << "Test accuracy: "
              << std::fixed << std::setprecision(4)
              << static_cast<double>(correct) / n_test
              << "\n";

    da_handle_destroy(&f);
    return 0;
}

