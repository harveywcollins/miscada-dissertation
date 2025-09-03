// random_forest_mnist.cpp

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

// Helper function to read a 32-bit big-endian integer from the file stream.
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
    
    // Read header information
    uint32_t magic_number = readUint32(ifs);
    if (magic_number != 2051) throw std::runtime_error("Invalid magic number in image file " + path);
    n_images = readUint32(ifs);
    rows = readUint32(ifs);
    cols = readUint32(ifs);
    
    std::cout << "Loading " << n_images << " images (" << rows << "x" << cols << ") from " << path << std::endl;

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

    std::cout << "Loading " << n_labels << " labels from " << path << std::endl;
    
    std::vector<int> labels(n_labels);
    for (int i = 0; i < n_labels; ++i) {
        unsigned char tmp;
        ifs.read(reinterpret_cast<char*>(&tmp), 1);
        labels[i] = static_cast<int>(tmp);
    }
    return labels;
}

int main() {
    std::cout << "AOCL-DA Decision Forest on MNIST Dataset\n\n";

    int n_train, n_test, rows, cols, dummy;
    std::vector<float> X_tr, X_te;
    std::vector<int> y_tr, y_te;

    try {
        X_tr = load_mnist_images(
            std::string(MNIST_DIR) + "/train-images.idx3-ubyte",
            n_train, rows, cols
        );
        y_tr = load_mnist_labels(
            std::string(MNIST_DIR) + "/train-labels.idx1-ubyte",
            dummy
        );
        X_te = load_mnist_images(
            std::string(MNIST_DIR) + "/t10k-images.idx3-ubyte",
            n_test, rows, cols
        );
        y_te = load_mnist_labels(
            std::string(MNIST_DIR) + "/t10k-labels.idx1-ubyte",
            dummy
        );

    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR: Failed to load MNIST data. " << e.what() << std::endl;
        std::cerr << "Files need to be present!!!" << std::endl;
        return 1;
    }

    int n_feat = rows * cols;
    int n_cls  = *std::max_element(y_tr.begin(), y_tr.end()) + 1;
    std::cout << "\nDataset Summary:" << std::endl;
    std::cout << "  Training samples: " << n_train << std::endl;
    std::cout << "  Test samples:     " << n_test << std::endl;
    std::cout << "  Features per sample: " << n_feat << std::endl;
    std::cout << "  Number of classes:   " << n_cls << "\n\n";

    // Transpose data into column-major format for AOCL-DA
    std::cout << "Transposing data to column-major format..." << std::endl;
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

    // Initialize AOCL-DA handle
    da_handle f = nullptr;
    if (da_handle_init_s(&f, da_handle_decision_forest) != da_status_success) {
        std::cerr << "ERROR: da_handle_init_s\n";
        return 1;
    }
    da_options_set_int(f, "logging", 1); // Enable logging

    // Set training data
    if (da_forest_set_training_data_s(f,
            /* n_rows = */ n_train,
            /* n_cols = */ n_feat,
            /* n_classes = */ n_cls,
            /* X = */ Xtr_cm.data(),
            /* stride = */ n_train,
            /* y = */ y_tr.data()
        ) != da_status_success)
    {
        std::cerr << "ERROR: da_forest_set_training_data_s\n";
        da_handle_destroy(&f);
        return 1;
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
        std::cerr << "ERROR: da_forest_fit_s\n";
        da_handle_destroy(&f);
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Training time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    // Predict on the test set
    std::cout << "\nStarting prediction..." << std::endl;
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

    // Calculate and report accuracy
    int correct = 0;
    for (int i = 0; i < n_test; ++i) {
        if (y_pred[i] == y_te[i]) ++correct;
    }
    std::cout << "Test accuracy: "
              << std::fixed << std::setprecision(4)
              << double(correct) / n_test
              << "\n";

    // Clean up
    da_handle_destroy(&f);
    return 0;
}