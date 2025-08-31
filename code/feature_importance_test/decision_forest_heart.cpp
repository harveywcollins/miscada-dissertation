#include "aoclda.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <numeric>

#ifndef DATA_DIR
#define DATA_DIR "/nobackup/wsgp73"
#endif

void check_status(da_status status, const std::string& func_name, da_handle handle = nullptr) {
    if (func_name == "get_result (size query)" && status == da_status_invalid_array_dimension) {
        fprintf(stderr, "[DIAG] Received expected status 'da_status_invalid_array_dimension'. This is correct.\n");
        return;
    }

    if (status != da_status_success) {
        fprintf(stderr, "FATAL ERROR in %s with status code %d\n", func_name.c_str(), status);
        if (handle) {
            da_handle_print_error_message(handle);
        }
        exit(1);
    }
}

// Function to load the heart failure CSV data
void load_heart_data(const std::string& path, std::vector<std::string>& feature_names, std::vector<float>& X_data, std::vector<int>& y_data, int& n_samples, int& n_features) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::string line, cell;
    
    // Read header
    if (std::getline(file, line)) {
        std::stringstream header_stream(line);
        while (std::getline(header_stream, cell, ',')) {
            feature_names.push_back(cell);
        }
        feature_names.pop_back();
        n_features = feature_names.size();
    }

    // Read data rows
    n_samples = 0;
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        int col_idx = 0;
        while (std::getline(line_stream, cell, ',')) {
            if (col_idx < n_features) {
                X_data.push_back(std::stof(cell));
            } else {
                y_data.push_back(std::stoi(cell));
            }
            col_idx++;
        }
        n_samples++;
    }
}

int main() {
    printf("[INFO] Starting Heart Failure C++ Test\n");

    std::vector<std::string> feature_names;
    std::vector<float> X_rowmajor;
    std::vector<int> y;
    int n_samples, n_features;

    try {
        printf("[INFO] Loading data...\n");
        load_heart_data(std::string(DATA_DIR) + "/heart_failure_clinical_records_dataset.csv", feature_names, X_rowmajor, y, n_samples, n_features);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR during data loading: %s\n", e.what());
        return 1;
    }
    printf("[INFO] Data loading complete. Samples: %d, Features: %d\n", n_samples, n_features);

    // Simple 80/20 train/test split
    int n_train = static_cast<int>(n_samples * 0.8);
    int n_test = n_samples - n_train;
    int n_cls = *std::max_element(y.begin(), y.end()) + 1;

    printf("[INFO] Transposing data to column-major format...\n");
    std::vector<float> X_tr_colmajor((size_t)n_train * n_features);
    for (int i = 0; i < n_train; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X_tr_colmajor[(size_t)j * n_train + i] = X_rowmajor[(size_t)i * n_features + j];
        }
    }
    std::vector<float> X_te_colmajor((size_t)n_test * n_features);
    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X_te_colmajor[(size_t)j * n_test + i] = X_rowmajor[(size_t)(i + n_train) * n_features + j];
        }
    }
    std::vector<int> y_tr(y.begin(), y.begin() + n_train);
    std::vector<int> y_te(y.begin() + n_train, y.end());
    
    printf("[INFO] Initializing AOCL-DA handle...\n");
    da_handle handle = nullptr;
    check_status(da_handle_init_s(&handle, da_handle_decision_forest), "da_handle_init_s");

    printf("[INFO] Setting options...\n");
    check_status(da_options_set_int(handle, "number of trees", 100), "options: n_trees");
    check_status(da_options_set_int(handle, "maximum depth", 10), "options: max_depth");
    check_status(da_options_set_string(handle, "scoring function", "gini"), "options: criterion");
    check_status(da_options_set_string(handle, "features selection", "sqrt"), "options: feat_selection");
    check_status(da_options_set_int(handle, "seed", 42), "options: seed");

    printf("[INFO] Setting training data...\n");
    check_status(da_forest_set_training_data_s(handle, n_train, n_features, n_cls, X_tr_colmajor.data(), n_train, y_tr.data()), "da_forest_set_training_data_s", handle);

    printf("[INFO] Starting training...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    check_status(da_forest_fit_s(handle), "da_forest_fit_s", handle);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Training completed in %f seconds.\n", std::chrono::duration<double>(t1 - t0).count());

    printf("\n--- Feature Importance Results ---\n");
    da_result query = da_feature_importances;
    da_int importances_dim = 1;
    std::vector<float> dummy_vec(1);
    da_status size_query_status = da_handle_get_result_s(handle, query, &importances_dim, dummy_vec.data());
    check_status(size_query_status, "get_result (size query)", handle);

    if (importances_dim > 1) {
        std::vector<float> importances(importances_dim);
        check_status(da_handle_get_result_s(handle, query, &importances_dim, importances.data()), "get_result (data query)", handle);
        
        std::vector<int> indices(n_features);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return importances[a] > importances[b];
        });

        printf("Features ranked by importance:\n");
        for (int i = 0; i < n_features; ++i) {
            int feat_idx = indices[i];
            printf("  %d. %-25s: %f\n", i + 1, feature_names[feat_idx].c_str(), importances[feat_idx]);
        }
    } else {
        fprintf(stderr, "ERROR: Failed to retrieve feature importances!!!!!\n");
    }

    printf("\n--- Model Performance ---\n");
    float score = 0.0f;
    check_status(da_forest_score_s(handle, n_test, n_features, X_te_colmajor.data(), n_test, y_te.data(), &score), "da_forest_score_s", handle);
    printf("Test accuracy: %.4f\n", score);

    printf("\n[INFO] Destroying handle...\n");
    da_handle_destroy(&handle);
    return 0;
}
