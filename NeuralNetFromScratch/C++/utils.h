#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

inline uint32_t ntohl_custom(uint32_t netlong) {
    return ((netlong & 0x000000FF) << 24) |
           ((netlong & 0x0000FF00) << 8) |
           ((netlong & 0x00FF0000) >> 8) |
           ((netlong & 0xFF000000) >> 24);
}

namespace utils {
    void read_mnist_train_data(const std::string& path,std::vector<Eigen::VectorXd>& data) {
        std::ifstream file(path, std::ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            int rows = 0;
            int cols = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ntohl_custom(magic_number);
            file.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = ntohl_custom(number_of_images);
            file.read((char*)&rows, sizeof(rows));
            rows = ntohl_custom(rows);
            file.read((char*)&cols, sizeof(cols));
            cols = ntohl_custom(cols);

            for (int i = 0; i < number_of_images; i++) {
                Eigen::VectorXd vec(rows * cols);
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        unsigned char temp = 0;
                        file.read((char*)&temp, sizeof(temp));
                        vec(r * cols + c) = (double)temp / 255.0;

                    }
                }
                data.push_back(vec);
            }
        }
        if (!file.is_open()) {
            std::cerr << "Failed to open: " << path << std::endl;
        }
    }

    void read_mnist_train_label(const std::string& path,
                                std::vector<Eigen::VectorXd>& data) {
        std::ifstream file(path, std::ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            int number_of_items = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ntohl_custom(magic_number);
            file.read((char*)&number_of_items, sizeof(number_of_items));
            number_of_items = ntohl_custom(number_of_items);

            for (int i = 0; i < number_of_items; i++) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                Eigen::VectorXd vec(10);
                vec.setZero();
                vec((int)temp) = 1.0;
                data.push_back(vec);
            }
        }
        if (!file.is_open()) {
            std::cerr << "Failed to open: " << path << std::endl;
        }
    }

    void read_mnist_test_data(const std::string& path,
                                std::vector<Eigen::VectorXd>& data) {
        std::ifstream file(path, std::ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            int rows = 0;
            int cols = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ntohl_custom(magic_number);
            file.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = ntohl_custom(number_of_images);
            file.read((char*)&rows, sizeof(rows));
            rows = ntohl_custom(rows);
            file.read((char*)&cols, sizeof(cols));
            cols = ntohl_custom(cols);

            for (int i = 0; i < number_of_images; i++) {
                Eigen::VectorXd vec(rows * cols);
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        unsigned char temp = 0;
                        file.read((char*)&temp, sizeof(temp));
                        vec(r * cols + c) = (double)temp / 255.0;
                    }
                }
                data.push_back(vec);
            }
        }
        if (!file.is_open()) {
            std::cerr << "Failed to open: " << path << std::endl;
        }
    }

    void read_mnist_test_label(const std::string& path,
                                std::vector<Eigen::VectorXd>& data) {
        std::ifstream file(path, std::ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            int number_of_items = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ntohl_custom(magic_number);
            file.read((char*)&number_of_items, sizeof(number_of_items));
            number_of_items = ntohl_custom(number_of_items);

            for (int i = 0; i < number_of_items; i++) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                Eigen::VectorXd vec(10);
                vec.setZero();
                vec((int)temp) = 1.0;
                data.push_back(vec);
            }
        }
        if (!file.is_open()) {
            std::cerr << "Failed to open: " << path << std::endl;
        }
    }
}

#endif // UTILS_H