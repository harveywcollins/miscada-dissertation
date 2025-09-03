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
#include <numeric>

#ifndef HIGGS_CSV_PATH
#define HIGGS_CSV_PATH "/nobackup/wsgp73/HIGGS.csv"
#endif

void check_status(da_status status, const std::string& func_name, da_handle handle = nullptr) {
    if (func_name == "get_result (size query)" && status == da_status_invalid_array_dimension) {
        fprintf(stderr, "Received expected status 'da_status_invalid_array_dimension'.\n");
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
        
        // First column is the label
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
    printf("Starting HIGGS Dataset Feature Importance Test\n");

    std::vector<std::vector<float>> all_features_rm; // Row-major
    std::vector<int> all_labels;
    if (!load_higgs_csv(HIGGS_CSV_PATH, all_features_rm, all_labels)) {
        return 1;
    }

    if (all_features_rm.empty()) {
        fprintf(stderr, "ERROR: No data loaded from CSV.\n");
        return 1;
    }

    int n_total = all_labels.size();
    int n_features = all_features_rm[0].size();
    int n_cls = *std::max_element(all_labels.begin(), all_labels.end()) + 1;
    printf("Data loading complete. Samples: %d, Features: %d\n", n_total, n_features);

    std::vector<std::string> feature_names;
    for (int i = 0; i < n_features; ++i) {
        feature_names.push_back("Feature_" + std::to_string(i + 1));
    }

    // 80/20 train/test split
    int n_train = static_cast<int>(n_total * 0.8);
    int n_test = n_total - n_train;

    // Transpose data to column-major format
    printf("Transposing data to column-major format...\n");
    std::vector<float> X_tr_colmajor((size_t)n_train * n_features);
    for (int i = 0; i < n_train; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X_tr_colmajor[(size_t)j * n_train + i] = all_features_rm[i][j];
        }
    }
    std::vector<float> X_te_colmajor((size_t)n_test * n_features);
    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X_te_colmajor[(size_t)j * n_test + i] = all_features_rm[n_train + i][j];
        }
    }
    
    int* y_tr_data = all_labels.data();
    int* y_te_data = all_labels.data() + n_train;

    printf("Initializing AOCL-DA handle...\n");
    da_handle handle = nullptr;
    check_status(da_handle_init_s(&handle, da_handle_decision_forest), "da_handle_init_s");

    printf("Setting options...\n");
    check_status(da_options_set_int(handle, "number of trees", 1), "options: n_trees");
    check_status(da_options_set_int(handle, "maximum depth", 15), "options: max_depth");
    check_status(da_options_set_string(handle, "scoring function", "entropy"), "options: criterion");
    check_status(da_options_set_int(handle, "seed", 42), "options: seed");
    check_status(da_options_set_int(handle, "maximum features", static_cast<int>(std::sqrt(n_features))), "options: max_features");

    printf("Setting training data...\n");
    check_status(da_forest_set_training_data_s(handle, n_train, n_features, n_cls, X_tr_colmajor.data(), n_train, y_tr_data), "da_forest_set_training_data_s", handle);

    printf("Starting training...\n");
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
            printf("  %2d. %-15s: %f\n", i + 1, feature_names[feat_idx].c_str(), importances[feat_idx]);
        }
    } else {
        fprintf(stderr, "ERROR: Failed to retrieve feature importances.\n");
    }

    printf("\n--- Model Performance ---\n");
    float score = 0.0f;
    check_status(da_forest_score_s(handle, n_test, n_features, X_te_colmajor.data(), n_test, y_te_data, &score), "da_forest_score_s", handle);
    printf("Test accuracy: %.4f\n", score);

    printf("\nDestroying handle...\n");
    da_handle_destroy(&handle);
    return 0;
}

