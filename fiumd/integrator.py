import torch
import torch.nn as nn
import fiumd

# Constants
TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
PICOSEC2TIMEU = 1000.0 / TIMEFACTOR

# TIMEFACTOR = torch.tensor(48.88821, dtype=torch.float32)
# BOLTZMAN = torch.tensor(0.001987191, dtype=torch.float32)
# PICOSEC2TIMEU = torch.tensor(1000.0, dtype=torch.float32) / TIMEFACTOR

class Integrator(nn.Module):
    """
    Abstract base class for integrators.
    Subclasses should override the `forward` method.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self):
        """
        Advance the ensemble state by one timestep of length dt.
        """
        raise NotImplementedError("Integrator subclasses must implement the step method.")

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
PICOSEC2TIMEU = 1000.0 / TIMEFACTOR

class VelocityVerletIntegrator(Integrator):
    """
    Velocity Verlet integrator for deterministic time evolution.

    Parameters
    ----------
    dt : float
        Integration timestep (in simulation units, e.g. ps)

    Assumes the ensemble exposes:
      - .positions : Tensor of shape (n_atoms, 3)
      - .velocities : Tensor of shape (n_atoms, 3)
      - .forces : Tensor of shape (n_atoms, 3)
      - .masses : Tensor of shape (n_atoms,)
      - .update_pairs_geometry_() : method to update pair geometry (e.g. distances, directions)
      - .compute_forces() : method to recompute forces
    """

    def __init__(self, dt: float, ensemble: fiumd.ensembles.Ensemble):
        super().__init__()

        self.ensemble = ensemble
        self._device = ensemble.device
        self._dtype = ensemble.dtype
        self.dt = nn.parameter.Buffer( torch.tensor(dt / TIMEFACTOR) )
        self.dv = nn.parameter.Buffer( torch.empty_like(ensemble.velocities),
                                        persistent=False)
        self.dx = nn.parameter.Buffer( torch.empty_like(ensemble.velocities),
                                        persistent=False)
        
        # self.half_dt = 0.5 * dt

    def forward(self):
        """
        Advances the ensemble by one integration step using Velocity Verlet.
        """
        # v(t + dt/2) = v(t) + (F(t)/m) * dt/2
        self.dv.copy_( self.ensemble.forces )
        self.dv.mul_( 0.5 * self.dt )
        self.dv.mul_( self.ensemble.inv_masses )
        
        self.ensemble.velocities.add_(self.dv)

        # x(t + dt) = x(t) + v(t + dt/2) * dt
        self.dx.copy_(self.ensemble.velocities)
        self.dx.mul_(self.dt)
        self.ensemble.positions.add_(self.dx)

        # Update geometry (e.g. bond vectors, unit directions, etc.)
        self.ensemble._update_pairs_geometry_()

        # F(t + dt)
        self.ensemble.compute_forces()

        # v(t + dt) = v(t + dt/2) + (F(t + dt)/m) * dt/2
        self.dv.copy_(self.ensemble.forces)
        self.dv.mul_(self.dt)
        self.dv.mul_( 0.5 * self.ensemble.inv_masses )
        self.ensemble.velocities.add_( self.dv )

    def attrs_info(self):
        fiumd.utils.attrs_info(self)

    def methods_info(self):
        fiumd.utils.methods_info(self)
        
class LangevinThermostat(nn.Module):
    def __init__(self, gamma, temperature, dt, ensemble: fiumd.ensembles.Ensemble):
        """
        Langevin thermostat for use in molecular dynamics simulations.

        Parameters
        ----------
        gamma : float
            Friction coefficient (1/ps)
        temperature : float or Tensor
            Scalar or (nsystems,) tensor of temperatures in Kelvin
        dt : float
            Time step (ps)
        ensemble : Ensemble
            The ensemble object, needed for atom/system info
        kB : float
            Boltzmann constant (defaults to 1.0 in reduced units)
        """
        super().__init__()
        
        self.kB = BOLTZMAN
        self.ensemble = ensemble
        self._device = ensemble.device
        self._dtype = ensemble.dtype
        self.gamma = torch.tensor(gamma / PICOSEC2TIMEU, 
                                  device=self._device,
                                  dtype=self._dtype)



        # Preallocated noise buffer: shape (n_total_atoms, 3)
        self._noise = torch.empty((ensemble.total_atoms, 3),
                                  device=self._device, dtype=self._dtype)

        self.set_temperature(temperature)
        self.set_dt(dt)
        
    def set_temperature(self, temperature):
        """Set the temperature(s), supporting scalar or per-system values."""
        
        if isinstance(temperature, (float, int)):
            temperature = torch.full((self.ensemble.nsystems,),
                                     temperature, device=self._device,
                                     dtype=self._dtype)
        elif isinstance(temperature, torch.Tensor):
            assert temperature.shape == (self.ensemble.nsystems,)
            temperature = temperature.to(self._device, self._dtype)
        else:
            raise TypeError(f"temperature must be float or tensor of shape (nsystems,)={(self.ensemble.nsystems,)}")

        # Expand to per-atom temperatures
        self._temperature_per_atom = torch.cat([
            temperature[i].expand(natoms, 1)
            for i, natoms in enumerate(self.ensemble.natoms_per_system)
        ], dim=0)  # shape: (total_atoms, 1)

    def set_dt(self, dt):
        """Set the timestep and update cached Langevin coefficients."""
        self.dt = torch.tensor(dt / TIMEFACTOR, device=self._device, dtype=self._dtype)
        self._exp_factor = torch.exp(-self.gamma * self.dt)
        
        self._noise_std = torch.sqrt(
            (1 - self._exp_factor**2) * self.kB * self._temperature_per_atom / self.ensemble.masses
        ).expand(self._noise.shape)

        self._noise_mean = torch.zeros_like(self._noise_std)

    def forward(self):
        """
        Applies the Langevin thermostat: velocity scaling and random noise.
        Should be called after velocity updates in each integration step.
        """
        torch.normal(self._noise_mean, self._noise_std,
                    out=self._noise)

        self.ensemble.velocities.mul_(self._exp_factor).add_(self._noise)


    def attrs_info(self):
        fiumd.utils.attrs_info(self)

    def methods_info(self):
        fiumd.utils.methods_info(self)


def maxwell_boltzmann(masses, T, replicas=1):
    natoms = len(masses)
    velocities = []
    for i in range(replicas):
        velocities.append(
            torch.sqrt(T * BOLTZMAN / masses) * torch.randn((natoms, 3)).type_as(masses)
        )

    return torch.stack(velocities, dim=0)