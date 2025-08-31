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

#ifndef MNIST_DIR
#define MNIST_DIR "/nobackup/wsgp73"
#endif

static uint32_t readUint32(std::ifstream &ifs) {
    unsigned char b[4];
    ifs.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) |
           (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) << 8)  |
           (uint32_t(b[3]));
}

static std::vector<float> load_mnist_images(const std::string &path,
                                            int &n_images, int &rows, int &cols) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Could not open " + path);
    readUint32(ifs);
    n_images = readUint32(ifs);
    rows = readUint32(ifs);
    cols = readUint32(ifs);
    std::vector<float> data(n_images * rows * cols);
    for (size_t i = 0; i < data.size(); ++i) {
        unsigned char tmp;
        ifs.read(reinterpret_cast<char*>(&tmp), 1);
        data[i] = tmp / 255.0f;
    }
    return data;
}

static std::vector<int> load_mnist_labels(const std::string &path, int &n_labels) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Could not open " + path);
    readUint32(ifs);
    n_labels = readUint32(ifs);
    std::vector<int> labels(n_labels);
    for (int i = 0; i < n_labels; ++i) {
        unsigned char tmp;
        ifs.read(reinterpret_cast<char*>(&tmp), 1);
        labels[i] = tmp;
    }
    return labels;
}

int main() {
    std::cout << "AOCL-DA Decision Forest on MNIST\n";

    // Load MNIST train
    int n_train, n_test, rows, cols, dummy;
    auto X_tr = load_mnist_images(
        std::string(MNIST_DIR) + "/train-images.idx3-ubyte",
        n_train, rows, cols
    );
    auto y_tr = load_mnist_labels(
        std::string(MNIST_DIR) + "/train-labels.idx1-ubyte",
        dummy
    );
    auto X_te = load_mnist_images(
        std::string(MNIST_DIR) + "/t10k-images.idx3-ubyte",
        n_test, rows, cols
    );
    auto y_te = load_mnist_labels(
        std::string(MNIST_DIR) + "/t10k-labels.idx1-ubyte",
        dummy
    );

    int n_feat = rows * cols;
    int n_cls  = *std::max_element(y_tr.begin(), y_tr.end()) + 1;

    // Transpose into column-major for AOCL-DA
    std::vector<float> Xtr_cm(n_train * n_feat);
    for (int i = 0; i < n_train; ++i) {
        for (int j = 0; j < n_feat; ++j) {
            Xtr_cm[j * n_train + i] = X_tr[i * n_feat + j];
        }
    }

    std::vector<float> Xte_cm(n_test * n_feat);
    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_feat; ++j) {
            Xte_cm[j * n_test + i] = X_te[i * n_feat + j];
        }
    }

    // Init handle
    da_handle f = nullptr;
    if (da_handle_init_s(&f, da_handle_decision_forest) != da_status_success) {
        std::cerr << "ERROR: da_handle_init_s\n";
        return 1;
    }
    da_options_set_int(f, "logging", 1);

    // Sanity check labels
    if (y_tr.size() != static_cast<size_t>(n_train)) {
        std::cerr << "ERROR: label count (" << y_tr.size()
                  << ") != n_train (" << n_train << ")\n";
        da_handle_destroy(&f);
        return 1;
    }

    // Set training data
    if (da_forest_set_training_data_s(f,
            /* n_rows    = */ n_train,
            /* n_cols    = */ n_feat,
            /* n_classes = */ n_cls,
            /* X         = */ Xtr_cm.data(),
            /* stride    = */ n_train,
            /* y         = */ y_tr.data()
        ) != da_status_success)
    {
        std::cerr << "ERROR: da_forest_set_training_data_s\n";
        da_handle_destroy(&f);
        return 1;
    }

    // Hyperparameters
    da_options_set_int   (f, "number of trees", 100);
    da_options_set_int   (f, "maximum depth", 15);
    da_options_set_string(f, "scoring function", "entropy");
    da_options_set_string(f, "task", "classification");
    da_options_set_int   (f, "seed", 42);

    // Train
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

    // Predict
    std::vector<int> y_pred(n_test);
    if (da_forest_predict_s(f,
            /* n_rows  = */ n_test,
            /* n_cols  = */ n_feat,
            /* X       = */ Xte_cm.data(),
            /* stride  = */ n_test,
            /* y_pred  = */ y_pred.data()
        ) != da_status_success)
    {
        std::cerr << "ERROR: da_forest_predict_s\n";
        da_handle_destroy(&f);
        return 1;
    }

    // Accuracy
    int correct = 0;
    for (int i = 0; i < n_test; ++i)
        if (y_pred[i] == y_te[i]) ++correct;
    std::cout << "Test accuracy: "
              << std::fixed << std::setprecision(4)
              << double(correct) / n_test
              << "\n";

    da_handle_destroy(&f);
    return 0;
}
