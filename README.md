# Neural Network Training Program

This program implements a simple neural network training algorithm using C. It includes functions for feedforward computation, training passes, and checking passes.

## Features

- **Feedforward Computation**: Computes the output of the neural network based on given input and weights.
- **Training Passes**: Adjusts the weights of the neural network based on training data to improve accuracy.
- **Checking Passes**: Evaluates the accuracy of the trained model against test data.
- **OpenMP Parallelization**: Utilizes OpenMP for parallelization, improving performance on multi-core systems.

## Dependencies

- **OpenMP**: Ensure your compiler supports OpenMP and is properly configured to use it for parallelization.

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://gitlab.rz.htw-berlin.de/s0580270/mpt23-neural_network-ankit-ankit.git
    ```

2. Compile the program using a C compiler with OpenMP support:

    ```bash
    make
    ```

## Usage

1. Run the compiled executable:

    ```bash
    ./main
    ```

2. Follow the prompts to perform training and testing passes.

## File Structure

- **`config.h`**: Contains configuration constants, function declarations, and utility functions.
- **`main.c`**: Implements the main functionality of the neural network training program.
- **`test.c`**: Includes unit tests for various functions in the program.
- **`Makefile`**: Defines compilation instructions for the program.

## How It Works

1. The program initializes the neural network's weights and inputs.
2. It performs training passes to adjust the weights based on training data.
3. After training, it evaluates the accuracy of the trained model using checking passes.
4. Finally, it outputs the results and performance metrics.



## License

This program is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
