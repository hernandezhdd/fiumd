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


class VelocityVerletIntegrator(Integrator):
    """
    Velocity Verlet integrator for deterministic time evolution.

    Parameters
    ----------
    dt : float
        Integration timestep (in fs)

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
        gamma : float or Tensor
            Friction coefficient (1/fs). Scalar or (nsystems,) tensor.
        temperature : float or Tensor
            Temperature(s) in Kelvin. Scalar or (nsystems,) tensor.
        dt : float or Tensor
            Time step (fs). Scalar or (nsystems,) tensor.
        ensemble : Ensemble
            The ensemble object, needed for atom/system information.
        """
        super().__init__()

        self.kB = BOLTZMAN
        self.ensemble = ensemble
        self._device = ensemble.device
        self._dtype = ensemble.dtype

        # Preallocated noise buffer: shape (total_atoms, 3)
        self._noise = torch.empty(
            (ensemble.total_atoms, 3),
            device=self._device,
            dtype=self._dtype,
        )

        self.set_gamma(gamma)
        self.set_temperature(temperature)
        self.set_dt(dt)

    def _system_parameter(self, value, name):
        """
        Convert a scalar or per-system tensor to a tensor of shape
        (nsystems,).
        """

        if isinstance(value, (float, int)):
            return torch.full(
                (self.ensemble.nsystems,),
                value,
                device=self._device,
                dtype=self._dtype,
            )

        elif isinstance(value, torch.Tensor):
            if value.shape != (self.ensemble.nsystems,):
                raise ValueError(
                    f"{name} must be a scalar or tensor of shape "
                    f"({self.ensemble.nsystems},), got {value.shape}"
                )

            return value.to(
                device=self._device,
                dtype=self._dtype,
            )

        else:
            raise TypeError(
                f"{name} must be a float, int, or tensor of shape "
                f"({self.ensemble.nsystems},)"
            )

    def _expand_to_atoms(self, values):
        """Expand per-system values to per-atom values."""

        return torch.cat([
            values[i].expand(natoms, 1)
            for i, natoms in enumerate(self.ensemble.natoms_per_system)
            ], dim=0)

    def set_gamma(self, gamma):
        """Set friction coefficient(s), scalar or per-system."""

        gamma = self._system_parameter(gamma, "gamma")

        # Convert from 1/ps to internal time units.
        self._gamma = gamma / PICOSEC2TIMEU

        # Expand to per-atom values.
        self._gamma_per_atom = self._expand_to_atoms(self._gamma)

        self._update_coefficients()

    def set_temperature(self, temperature):
        """Set temperature(s), scalar or per-system."""

        self._temperature = self._system_parameter(
            temperature,
            "temperature",
        )

        # Expand to per-atom values.
        self._temperature_per_atom = self._expand_to_atoms(
            self._temperature
        )

        self._update_coefficients()

    def set_dt(self, dt):
        """Set timestep(s), scalar or per-system."""

        dt = self._system_parameter(dt, "dt")

        # Convert from ps to internal time units.
        self._dt = dt / TIMEFACTOR

        # Expand to per-atom values.
        self._dt_per_atom = self._expand_to_atoms(self._dt)

        self._update_coefficients()

    def _update_coefficients(self):
        """
        Update all cached Langevin coefficients.

        Does nothing until gamma, temperature, and dt have all
        been initialized.
        """

        if not all(hasattr(self, attr) for attr in (
            "_gamma_per_atom",
            "_temperature_per_atom",
            "_dt_per_atom",
        )):
            return

        # Exact Ornstein-Uhlenbeck propagation factor:
        #
        # v' = exp(-gamma * dt) * v + noise
        #
        self._exp_factor = torch.exp(
            -self._gamma_per_atom * self._dt_per_atom
        )

        # Exact OU noise standard deviation:
        #
        # sigma = sqrt(
        #     (1 - exp(-2 * gamma * dt))
        #     * kB * T / m
        # )
        #
        
        self._noise_std = torch.sqrt(
            (1 - self._exp_factor**2)
            * self.kB
            * self._temperature_per_atom
            / self.ensemble.masses
        )#.unsqueeze(-1)

    def forward(self):
        """
        Apply the Langevin thermostat (OU step) to the velocities.
        """

        torch.randn(
            self._noise.size(),
            out=self._noise,
        )

        self._noise.mul_(self._noise_std)

        self.ensemble.velocities.mul_(
            self._exp_factor#.unsqueeze(-1)
        )

        self.ensemble.velocities.add_(self._noise)

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