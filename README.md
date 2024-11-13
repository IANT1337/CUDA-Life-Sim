# Prey vs Predator Simulation

This project simulates a prey vs predator environment using CUDA for parallel processing and Pygame for rendering.

## Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Configure the simulation parameters in `config/animalConfig.json`.

2. Run the simulation:
    ```sh
    python src/main.py
    ```

## Project Structure

- `config/`: Contains configuration files.
- `src/`: Contains the source code.
  - `main.py`: Entry point for the simulation.
  - `simulation/`: Contains simulation logic.
  - `utils/`: Contains utility functions.
- `requirements.txt`: Lists the required dependencies.
- `README.md`: Project documentation.