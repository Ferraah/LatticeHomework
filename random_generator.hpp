#ifndef RANDOM_GENERATOR_HPP
#define RANDOM_GENERATOR_HPP

#include <random>
#include <chrono>

template<typename T>
class RandomGenerator {
public:
    RandomGenerator() {
        seed(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
    }

    RandomGenerator(unsigned seedValue) {
        seed(seedValue);
    }

    void seed(unsigned seedValue) {
        generator.seed(seedValue);
    }

    T getRandomNumber(T min, T max) {
        if constexpr (std::is_integral<T>::value) {
            std::uniform_int_distribution<T> distribution(min, max);
            return distribution(generator);
        } else {
            std::uniform_real_distribution<T> distribution(min, max);
            return distribution(generator);
        }
    }
    void populateArray(T* array, std::size_t size, T min, T max) {
        for (std::size_t i = 0; i < size; ++i) {
            array[i] = getRandomNumber(min, max);
        }
    }

private:
    std::mt19937 generator;
};

#endif // RANDOM_GENERATOR_HPP