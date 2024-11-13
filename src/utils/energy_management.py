import cupy as cp

def manage_energy(prey_energy, predator_energy, prey_energy_consumption, predator_energy_consumption, max_prey_energy, max_predator_energy):
    prey_energy -= prey_energy_consumption
    predator_energy -= predator_energy_consumption

    prey_energy = cp.clip(prey_energy, 0, max_prey_energy)
    predator_energy = cp.clip(predator_energy, 0, max_predator_energy)

    return prey_energy, predator_energy

def handle_reproduction(prey_energy, predator_energy, reproduction_threshold_prey, reproduction_threshold_predator, prey_reproduce_flags, predator_reproduce_flags):
    prey_reproduce_flags[prey_energy >= reproduction_threshold_prey] = 1
    predator_reproduce_flags[predator_energy >= reproduction_threshold_predator] = 1

    return prey_reproduce_flags, predator_reproduce_flags