//-----------------------------------------------------------------------------
/// @file module9_thrust.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 9 Main Driver
//-----------------------------------------------------------------------------
#include <iostream>
#include <time.h>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>

#define PRINT_DEBUG 0

//-----------------------------------------------------------------------------
/// @brief Generates a randomized integer device vector.
/// @param vector_size size of the output vector.
/// @return device vector with random integers of size vector_size.
//-----------------------------------------------------------------------------
thrust::device_vector<int> gen_rand_vector(const size_t vector_size)
{
  // Seed RNG
  static time_t seed = time(NULL);
  srand(seed++); // successive calls may have the same seed, ensure update

  // Generate random number on CPU and copy to device vector
  thrust::host_vector<int> host_v(vector_size);
  thrust::generate(host_v.begin(), host_v.end(), rand);
  thrust::device_vector<int> device_v = host_v;
  return device_v;
}

//-----------------------------------------------------------------------------
/// @brief Performs vector-wise modulo operation.
/// @param vector input vector.
/// @param modulus modulus vector.
/// @return remainder vector: rem = vector % modulus.
//-----------------------------------------------------------------------------
thrust::device_vector<int> vector_modulo(const thrust::device_vector<int> vector, const int modulus)
{
  // Set up modulus and remainder vector and perform transformation
  thrust::device_vector<int> modulus_v(vector.size());
  thrust::device_vector<int> remainder_v(vector.size());
  thrust::fill(modulus_v.begin(), modulus_v.end(), modulus);
  thrust::transform(vector.begin(), vector.end(), modulus_v.begin(), remainder_v.begin(), thrust::modulus<int>());
  return remainder_v;
}

int main(int argc, char **argv)
{
  // Read command line arguments
  size_t vector_size = (1 << 20);
  int iterations = 1000;

  if (argc >= 2)
  {
    vector_size = atoi(argv[1]);
  }
  if (argc >= 3)
  {
    iterations = atoi(argv[2]);
  }

  // Generate device vectors with random integers.
  // Perform modulo to limit random max
  thrust::device_vector<int> alpha = vector_modulo(gen_rand_vector(vector_size), 100);
  thrust::device_vector<int> bravo = vector_modulo(gen_rand_vector(vector_size), 100);
  thrust::device_vector<int> delta = vector_modulo(gen_rand_vector(vector_size), 100);
  thrust::device_vector<int> gamma = vector_modulo(gen_rand_vector(vector_size), 100);
  thrust::device_vector<int> omega = vector_modulo(gen_rand_vector(vector_size), 100);
  thrust::device_vector<int> diff(vector_size);
  thrust::device_vector<int> sum(vector_size);
  thrust::device_vector<int> prod(vector_size);
  thrust::device_vector<int> rem(vector_size);

  std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();
  for (int iter = 0; iter < iterations; iter++)
  {
    // Perform addition sum = alpha + bravo
    thrust::transform(alpha.begin(), alpha.end(), bravo.begin(), sum.begin(), thrust::plus<int>());

    // Perform subtraction diff = sum - delta
    thrust::transform(sum.begin(), sum.end(), delta.begin(), diff.begin(), thrust::minus<int>());

    // Perform multiplication prod = delta * gamma
    thrust::transform(diff.begin(), diff.end(), gamma.begin(), prod.begin(), thrust::multiplies<int>());

    // Perform modulo rem = prod % gamma
    thrust::transform(prod.begin(), prod.end(), omega.begin(), rem.begin(), thrust::modulus<int>());
  }
  std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();

#if PRINT_DEBUG
  for (int i = 0; i < vector_size; i++)
  {
    std::cout << alpha[i];
    std::cout << "+" << bravo[i] << "=" << sum[i];
    std::cout << "-" << delta[i] << "=" << diff[i];
    std::cout << "*" << gamma[i] << "=" << prod[i];
    std::cout << "%" << omega[i] << "=" << rem[i];
    std::cout << std::endl;
  }
#endif

  // Print performance metrics
  std::cout << vector_size << ", " << iterations << ", ";
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << std::endl;

  return 0;
}
