// random_forest_mnist_weak.cpp

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
#include <cstdlib> // For getenv

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

// Loads the MNIST image data.
static std::vector<float> load_mnist_images(const std::string &path,
                                            int &n_images, int &rows, int &cols) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Could not open " + path);
    uint32_t magic_number = readUint32(ifs);
    if (magic_number != 2051) throw std::runtime_error("Invalid magic number in image file " + path);
    n_images = readUint32(ifs);
    rows = readUint32(ifs);
    cols = readUint32(ifs);
    std::vector<unsigned char> raw_data(n_images * rows * cols);
    ifs.read(reinterpret_cast<char*>(raw_data.data()), raw_data.size());
    std::vector<float> data(raw_data.size());
    for (size_t i = 0; i < raw_data.size(); ++i) {
        data[i] = raw_data[i] / 255.0f;
    }
    return data;
}

// Loads the MNIST label data.
static std::vector<int> load_mnist_labels(const std::string &path, int &n_labels) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Could not open " + path);
    uint32_t magic_number = readUint32(ifs);
    if (magic_number != 2049) throw std::runtime_error("Invalid magic number in label file " + path);
    n_labels = readUint32(ifs);
    std::vector<int> labels(n_labels);
    for (int i = 0; i < n_labels; ++i) {
        unsigned char tmp;
        ifs.read(reinterpret_cast<char*>(&tmp), 1);
        labels[i] = static_cast<int>(tmp);
    }
    return labels;
}

int main() {
    std::cout << "AOCL-DA Decision Forest on MNIST - WEAK SCALING TEST\n\n";

    // --- WEAK SCALING START ---
    int n_threads = 1;
    const char* n_threads_env = std::getenv("OMP_NUM_THREADS");
    if (n_threads_env) {
        try {
            n_threads = std::stoi(n_threads_env);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse OMP_NUM_THREADS. Defaulting to 1 thread." << std::endl;
        }
    }
    
    const int MAX_TRAIN_SAMPLES = 60000;
    const int MAX_THREADS_REFERENCE = 128;
    int n_train_weak = static_cast<long long>(n_threads) * MAX_TRAIN_SAMPLES / MAX_THREADS_REFERENCE;
    n_train_weak = std::min(MAX_TRAIN_SAMPLES, n_train_weak); // Ensure we don't exceed total samples

    std::cout << "Running with " << n_threads << " threads." << std::endl;
    std::cout << "Weak Scaling: Using " << n_train_weak << " of " << MAX_TRAIN_SAMPLES << " training samples.\n" << std::endl;
    // --- WEAK SCALING END ---


    int n_train_full, n_test, rows, cols, dummy;
    std::vector<float> X_tr_full, X_te;
    std::vector<int> y_tr_full, y_te;

    try {
        // Load the FULL dataset first
        X_tr_full = load_mnist_images(std::string(MNIST_DIR) + "/train-images.idx3-ubyte", n_train_full, rows, cols);
        y_tr_full = load_mnist_labels(std::string(MNIST_DIR) + "/train-labels.idx1-ubyte", dummy);
        X_te = load_mnist_images(std::string(MNIST_DIR) + "/t10k-images.idx3-ubyte", n_test, rows, cols);
        y_te = load_mnist_labels(std::string(MNIST_DIR) + "/t10k-labels.idx1-ubyte", dummy);
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR: Failed to load MNIST data. " << e.what() << std::endl;
        return 1;
    }

    int n_feat = rows * cols;
    int n_cls  = *std::max_element(y_tr_full.begin(), y_tr_full.end()) + 1;

    // Transpose data into column-major format for AOCL-DA, using only the subset of data for weak scaling
    std::vector<float> Xtr_cm(n_train_weak * n_feat);
    for (int i = 0; i < n_train_weak; ++i) {
        for (int j = 0; j < n_feat; ++j) {
            Xtr_cm[j * n_train_weak + i] = X_tr_full[i * n_feat + j];
        }
    }

    // Initialize AOCL-DA handle
    da_handle f = nullptr;
    if (da_handle_init_s(&f, da_handle_decision_forest) != da_status_success) {
        std::cerr << "ERROR: da_handle_init_s\n"; return 1;
    }
    da_options_set_int(f, "logging", 1);

    // Set training data using the n_train_weak size
    if (da_forest_set_training_data_s(f,
            /* n_rows    = */ n_train_weak,
            /* n_cols    = */ n_feat,
            /* n_classes = */ n_cls,
            /* X         = */ Xtr_cm.data(),
            /* stride    = */ n_train_weak,
            /* y         = */ y_tr_full.data() // Pointer is fine, library will only read n_train_weak elements
        ) != da_status_success)
    {
        std::cerr << "ERROR: da_forest_set_training_data_s\n"; da_handle_destroy(&f); return 1;
    }

    // Set hyperparameters
    da_options_set_int   (f, "number of trees",  32);
    da_options_set_int   (f, "maximum depth",    15);
    da_options_set_string(f, "scoring function", "entropy");
    da_options_set_string(f, "task",             "classification");
    da_options_set_int   (f, "seed",             42);
    
    std::cout << "\nStarting training..." << std::endl;
    
    // Train the model and measure time
    auto t0 = std::chrono::high_resolution_clock::now();
    if (da_forest_fit_s(f) != da_status_success) {
        std::cerr << "ERROR: da_forest_fit_s\n"; da_handle_destroy(&f); return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Training time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    // Prediction on the test set is unchanged
    std::cout << "\nStarting prediction on full test set..." << std::endl;
    std::vector<int> y_pred(n_test);
    std::vector<float> Xte_cm(n_test * n_feat); // Transpose test data
    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_feat; ++j) {
            Xte_cm[j * n_test + i] = X_te[i * n_feat + j];
        }
    }

    if (da_forest_predict_s(f,
            /* n_rows = */ n_test,
            /* n_cols = */ n_feat,
            /* X      = */ Xte_cm.data(),
            /* stride = */ n_test,
            /* y_pred = */ y_pred.data()
        ) != da_status_success)
    {
        std::cerr << "ERROR: da_forest_predict_s\n"; da_handle_destroy(&f); return 1;
    }

    // Calculate and report accuracy
    int correct = 0;
    for (int i = 0; i < n_test; ++i) {
        if (y_pred[i] == y_te[i]) ++correct;
    }
    std::cout << "Test accuracy: "
              << std::fixed << std::setprecision(4)
              << double(correct) / n_test
              << "\n";

    da_handle_destroy(&f);
    return 0;
}
