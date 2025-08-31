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
#include <likwid.h>

#ifndef HIGGS_CSV_PATH
#define HIGGS_CSV_PATH "/nobackup/wsgp73/HIGGS.csv"
#endif

// (Your load_higgs_csv function remains the same)
bool load_higgs_csv(const std::string &path,
                    std::vector<std::vector<float>> &features,
                    std::vector<int> &labels) {
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "Could not open " << path << std::endl;
        return false;
    }
    std::cout << "Loading data from " << path << ". This may take a moment..." << std::endl;
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
    LIKWID_MARKER_INIT;

    std::cout << "AOCL-DA Decision Forest on HIGGS Dataset with LIKWID\n";
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
    std::cout << "Original test set size: " << n_test << "\n";

    std::cout << "Transposing data to column-major format...\n";
    std::vector<float> Xtr_cm(n_train * n_feat);
    for (int i = 0; i < n_train; ++i) {
        for (int j = 0; j < n_feat; ++j) {
            Xtr_cm[j * n_train + i] = all_features_rm[i][j];
        }
    }
    
    int* y_tr_data = all_labels.data();
    int* y_te_data = all_labels.data() + n_train;

    da_handle f = nullptr;
    da_handle_init_s(&f, da_handle_decision_forest);
    da_options_set_int(f, "logging", 1);
    da_forest_set_training_data_s(f, n_train, n_feat, n_cls, Xtr_cm.data(), n_train, y_tr_data);

    // FIXED: Replaced non-breaking spaces with normal spaces
    da_options_set_int(f, "number of trees", 1);
    da_options_set_int(f, "maximum depth", 35);
    da_options_set_int(f, "node minimum samples", 10);
    da_options_set_string(f, "scoring function", "entropy");
    da_options_set_string(f, "task", "classification");
    da_options_set_int(f, "seed", 42);
    da_options_set_int(f, "maximum features", static_cast<int>(std::sqrt(n_feat)));

    std::cout << "Starting model training...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    da_forest_fit_s(f);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Training time: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    // --- NEW SECTION: CREATE A LARGE TEST SET TO BUST THE CACHE ---
    const long long L3_CACHE_SIZE = 256 * 1024 * 1024;
    const long long TARGET_SIZE = 4 * L3_CACHE_SIZE;
    long long test_set_bytes = (long long)n_test * n_feat * sizeof(float);
    int replication_factor = (test_set_bytes > 0) ? (static_cast<int>(TARGET_SIZE / test_set_bytes) + 1) : 1;
    
    int n_test_large = n_test * replication_factor;
    std::cout << "Creating large test set to bust L3 cache.\n";
    std::cout << "Replicating original test set " << replication_factor << " times.\n";
    std::cout << "New large test set size: " << n_test_large << " samples.\n";

    std::vector<float> Xte_rm_large(n_test_large * n_feat);
    std::vector<int> y_te_large(n_test_large);

    for (int i = 0; i < replication_factor; ++i) {
        // FIXED: Replaced std::copy with a proper loop to flatten the 2D data
        for (int j = 0; j < n_test; ++j) {
            // Calculate the starting position for the current replication
            size_t dest_offset = (size_t)(i * n_test + j) * n_feat;
            // Copy the row data
            std::copy(all_features_rm[n_train + j].begin(), all_features_rm[n_train + j].end(), Xte_rm_large.begin() + dest_offset);
        }
        // Copy labels
        std::copy(y_te_data, y_te_data + n_test, y_te_large.begin() + i * n_test);
    }
    
    std::vector<float> Xte_cm_large_transposed(n_test_large * n_feat);
    for (int i = 0; i < n_test_large; ++i) {
        for (int j = 0; j < n_feat; ++j) {
            Xte_cm_large_transposed[j * n_test_large + i] = Xte_rm_large[i * n_feat + j];
        }
    }
    
    std::cout << "Predicting on LARGE test set to measure DRAM-bound performance...\n";
    std::vector<int> y_pred_large(n_test_large);
    
    LIKWID_MARKER_START("predict_loop_nocache");
    if (da_forest_predict_s(f, n_test_large, n_feat, Xte_cm_large_transposed.data(), n_test_large, y_pred_large.data()) != da_status_success) {
        std::cerr << "ERROR: da_forest_predict_s on large dataset\n";
        da_handle_destroy(&f);
        LIKWID_MARKER_CLOSE;
        return 1;
    }
    LIKWID_MARKER_STOP("predict_loop_nocache");

    int correct_large = 0;
    for (int i = 0; i < n_test_large; ++i) {
        if (y_pred_large[i] == y_te_large[i]) {
            ++correct_large;
        }
    }
    std::cout << "Large test set accuracy: " << std::fixed << std::setprecision(4)
              << static_cast<double>(correct_large) / n_test_large << "\n";

    da_handle_destroy(&f);
    LIKWID_MARKER_CLOSE;
    return 0;
}
