from utils.config_loader import load_config
from simulation.simulation import Simulation

def main():
    config = load_config('config/animalConfig.json')
    simulation = Simulation(config)
    simulation.run()

if __name__ == "__main__":
    main()