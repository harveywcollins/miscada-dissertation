#include "aoclda.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <cstdio>

#ifndef MNIST_DIR
#define MNIST_DIR "/nobackup/wsgp73"
#endif

void check_status(da_status status, const std::string& func_name, da_handle handle = nullptr) {
    if (func_name == "get_result (size query)" && status == da_status_invalid_array_dimension) {
        fprintf(stderr, "Received expected status code 4 (invalid dimension) for size query. This is correct.\n");
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

static uint32_t readUint32(std::ifstream &ifs) {
    unsigned char b[4];
    ifs.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | (uint32_t(b[3]));
}

static std::vector<float> load_mnist_images(const std::string &path, da_int &n_images, da_int &rows, da_int &cols) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Could not open " + path);
    uint32_t magic_number = readUint32(ifs);
    if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file: " + path);
    n_images = readUint32(ifs);
    rows = readUint32(ifs);
    cols = readUint32(ifs);
    std::vector<unsigned char> raw_data(n_images * rows * cols);
    ifs.read(reinterpret_cast<char*>(raw_data.data()), raw_data.size());
    std::vector<float> data(raw_data.size());
    for (size_t i = 0; i < raw_data.size(); ++i) data[i] = raw_data[i] / 255.0f;
    return data;
}

static std::vector<da_int> load_mnist_labels(const std::string &path, da_int &n_labels) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Could not open " + path);
    uint32_t magic_number = readUint32(ifs);
    if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file: " + path);
    n_labels = readUint32(ifs);
    std::vector<da_int> labels(n_labels);
    for (da_int i = 0; i < n_labels; ++i) {
        unsigned char tmp;
        ifs.read(reinterpret_cast<char*>(&tmp), 1);
        labels[i] = static_cast<da_int>(tmp);
    }
    return labels;
}

int main() {
    fprintf(stderr, "Program Start.\n");

    da_int n_train, n_test, rows, cols, dummy;
    std::vector<float> X_tr_rowmajor, X_te_rowmajor;
    std::vector<da_int> y_tr, y_te;

    try {
        X_tr_rowmajor = load_mnist_images(std::string(MNIST_DIR) + "/train-images.idx3-ubyte", n_train, rows, cols);
        y_tr = load_mnist_labels(std::string(MNIST_DIR) + "/train-labels.idx1-ubyte", dummy);
        X_te_rowmajor = load_mnist_images(std::string(MNIST_DIR) + "/t10k-images.idx3-ubyte", n_test, rows, cols);
        y_te = load_mnist_labels(std::string(MNIST_DIR) + "/t10k-labels.idx1-ubyte", dummy);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR during data loading: %s\n", e.what()); return 1;
    }
    fprintf(stderr, "Data loading complete.\n");

    da_int n_feat = rows * cols;
    da_int n_cls  = *std::max_element(y_tr.begin(), y_tr.end()) + 1;

    fprintf(stderr, "Transposing data to column-major format...\n");
    std::vector<float> X_tr_colmajor(n_train * n_feat);
    for (da_int i = 0; i < n_train; ++i) {
        for (da_int j = 0; j < n_feat; ++j) {
            X_tr_colmajor[j * n_train + i] = X_tr_rowmajor[i * n_feat + j];
        }
    }
    std::vector<float> X_te_colmajor(n_test * n_feat);
    for (da_int i = 0; i < n_test; ++i) {
        for (da_int j = 0; j < n_feat; ++j) {
            X_te_colmajor[j * n_test + i] = X_te_rowmajor[i * n_feat + j];
        }
    }
    
    fprintf(stderr, "Initializing AOCL-DA handle...\n");
    da_handle handle = nullptr;
    check_status(da_handle_init_s(&handle, da_handle_decision_forest), "da_handle_init_s");

    fprintf(stderr, "Setting options...\n");
    check_status(da_options_set_int(handle, "number of trees", 100), "options: n_trees");
    check_status(da_options_set_int(handle, "maximum depth", 15), "options: max_depth");
    check_status(da_options_set_string(handle, "scoring function", "gini"), "options: criterion");
    check_status(da_options_set_int(handle, "seed", 42), "options: seed");
    check_status(da_options_set_string(handle, "features selection", "sqrt"), "options: features selection");

    fprintf(stderr, "Setting training data (n_train=%ld, n_feat=%ld, ldx=%ld)...\n", n_train, n_feat, n_train);
    check_status(da_forest_set_training_data_s(handle, n_train, n_feat, n_cls, X_tr_colmajor.data(), n_train, y_tr.data()), "da_forest_set_training_data_s", handle);

    fprintf(stderr, "Starting training...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    check_status(da_forest_fit_s(handle), "da_forest_fit_s", handle);
    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "Training completed.\n");
    std::cout << "Training completed in " << std::chrono::duration<double>(t1 - t0).count() << " seconds." << std::endl;

    fprintf(stderr, "\nExtracting feature importances...\n");
    da_result query = da_feature_importances;
    da_int importances_dim = 1;
    std::vector<float> dummy_vec(1);

    da_status size_query_status = da_handle_get_result_s(handle, query, &importances_dim, dummy_vec.data());
    check_status(size_query_status, "get_result (size query)", handle);

    if (importances_dim > 1) { 
        std::vector<float> importances(importances_dim);
        check_status(da_handle_get_result_s(handle, query, &importances_dim, importances.data()), "get_result (data query)", handle);
        std::cout << "Successfully extracted " << importances_dim << " importance scores." << std::endl;
        
        std::cout << "First 10 importances:" << std::endl;
        for(int i = 0; i < 10 && i < importances_dim; ++i) {
            std::cout << "  Feature " << i << ": " << std::fixed << std::setprecision(6) << importances[i] << std::endl;
        }

    } else {
        std::cerr << "ERROR: Failed to retrieve the correct dimension for feature importances." << std::endl;
    }

    fprintf(stderr, "\nStarting prediction...\n");
    float score = 0.0f;
    check_status(da_forest_score_s(handle, n_test, n_feat, X_te_colmajor.data(), n_test, y_te.data(), &score), "da_forest_score_s", handle);
    std::cout << "Test accuracy: " << std::fixed << std::setprecision(4) << score << std::endl;

    fprintf(stderr, "Destroying handle...\n");
    da_handle_destroy(&handle);
    fprintf(stderr, "Program End.\n");
    return 0;
}
