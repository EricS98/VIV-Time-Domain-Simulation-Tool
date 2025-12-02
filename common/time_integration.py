# common/time_integration.py
"""
Time Integration Module
================================================================

Newmark-beta integration methods for VIV analysis of generalized SDOF systems.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Union, Callable, Optional, Dict
from dataclasses import dataclass
import warnings

@dataclass
class NewmarkParameters:
    """Newmark integration parameters with validation."""
    gamma: float = 0.5
    beta: float = 0.25

@dataclass
class NonLinearConvergenceParams:
    """Convergence parameters for non-linear Newmark algorithm."""
    max_iterations: int = 20
    tolerance: float = 1e-6        

class NewmarkIntegrator:
    """Newmark-beta time integrator optimized for generalized 1-DOF systems."""

    def __init__(self, parameters: NewmarkParameters = None):
        """
        Initialize integrator.

        Parameters:
        -----------
        parameters : NewmarkParameters, optional
            Integration parameters. Defaults to average constant acceleration.
        """
        self.params = parameters or NewmarkParameters()

    def integrate_krenk(self, M: Union[float, np.ndarray], 
                       C: Union[float, np.ndarray], 
                       K: Union[float, np.ndarray],
                       f: np.ndarray, 
                       u0: float, 
                       udot0: float,
                       dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Newmark integration using Krenk's formulation.

        Parameters:
        -----------
        M, C, K : float or np.ndarray
            Modal mass, damping, and stiffness [scalars for generalized SDOF]
        f : np.ndarray
            Force time series [n_steps] or [1 x n_steps]
        u0, udot0 : float
            Initial displacement and velocity
        dt : float
            Time step size [s]
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (displacements, velocities, accelerations) [n_steps] each
        """
        # Convert force to 1D if needed (for generalized SDOF systems)
        f = np.atleast_1d(f).flatten()
        n_steps = len(f)
        
        # Ensure scalar system parameters (generalized SDOF)
        M = float(M)
        C = float(C) 
        K = float(K)
        
        # Validate inputs
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if M <= 0:
            raise ValueError("Mass must be positive")

        gamma, beta = self.params.gamma, self.params.beta

        # Initialize response arrays
        u = np.zeros(n_steps)
        udot = np.zeros(n_steps)
        udotdot = np.zeros(n_steps)

        # Set initial conditions
        u[0] = float(u0)
        udot[0] = float(udot0)

        # Calculate modified mass (Krenk's approach) - scalar for SDOF
        M_star = M + gamma * dt * C + beta * dt**2 * K

        # Calculate initial acceleration
        udotdot[0] = (f[0] - C * udot[0] - K * u[0]) / M

        # Time integration loop
        for n in range(n_steps - 1):
            # Predictor step
            udot_star = udot[n] + (1 - gamma) * dt * udotdot[n]
            u_star = u[n] + dt * udot[n] + (0.5 - beta) * dt**2 * udotdot[n]

            # Corrector step - solve for acceleration at next time step
            force_effective = f[n+1] - C * udot_star - K * u_star
            udotdot[n+1] = force_effective / M_star

            # Update velocity and displacement
            udot[n+1] = udot_star + gamma * dt * udotdot[n+1]
            u[n+1] = u_star + beta * dt**2 * udotdot[n+1]

        return u, udot, udotdot
    
    def integrate_nonlinear_krenk(self, M: float, K: float,
                                  damping_func, damping_derivative_func,
                                  f: np.ndarray,
                                  u0: float, udot0:float,
                                  dt: float,
                                  convergence_params: Optional[NonLinearConvergenceParams] = None,
                                  store_rms_history: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict]]:
        """
        Non-linear Newmark integration using Krenk's Algorithm.

        Solves: M*ÿ + C(y)*ẏ + K*y = f(t)
        Where C(y) = damping_func(y, ẏ) is amplitude-dependent

        Parameters:
        -----------
        M : float
            Modal mass [kg]
        K : float  
            Modal stiffness [N/m]
        damping_func : callable
            Function C(u, u_dot) returning total damping coefficient [Ns/m]
            For VIV: C(u, u_dot) = 2*M*ωₙ*[ζₛ - ζₐ(u)]
        damping_derivative_func : callable
            Function ∂C/∂u for tangent matrix calculation
        f : np.ndarray
            External force time series [N]
        u0 : float
            Initial displacement [m]
        udot0 : float
            Initial velocity [m/s]
        dt : float
            Time step [s]
        convergence_params : NonLinearConvergenceParams, optional
            Convergence control parameters
        store_rms_history : bool
            Whether to store RMS and damping history (default: True)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict]]
            (displacements, velocities, accelerations) time series
        """
        # Convert force to 1D and validate
        f = np.atleast_1d(f).flatten()
        n_steps = len(f)

        # Ensure scalar system parameters
        M = float(M)
        K = float(K)

        # Setup convergence parameters
        if convergence_params is None:
            convergence_params = NonLinearConvergenceParams()

        # Validate inputs
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if M <= 0:
            raise ValueError("Mass must be positive")
        if K <= 0:
            raise ValueError("Stiffness must be positive")
        
        gamma, beta = self.params.gamma, self.params.beta

        # Initialize response arrays
        u = np.zeros(n_steps)
        udot = np.zeros(n_steps)
        udotdot = np.zeros(n_steps)

        # Initialize history storage if requested
        if store_rms_history:
            rms_history = np.zeros(n_steps)
            zeta_aero_history = np.zeros(n_steps)
            zeta_total_history = np.zeros(n_steps)
            zeta_struct_history = np.zeros(n_steps)

        # Set initial conditions
        u[0] = float(u0)
        udot[0] = float(udot0)

        # Calculate initial acceleration from equation of motion
        # M*ü + C(u,u̇)*u̇ + K*u = f
        C_initial = damping_func(u[0], udot[0])
        udotdot[0] = (f[0] - C_initial * udot[0] - K * u[0]) / M

        # Store initial RMS and damping if available
        if store_rms_history and hasattr(damping_func, 'vb_damping'):
            vb = damping_func.vb_damping
            rms_history[0] = vb.get_current_rms()
            damping_info = vb.get_damping_info(u[0], 0.0)
            zeta_aero_history[0] = damping_info['zeta_a']
            zeta_total_history[0] = damping_info['zeta_total']
            zeta_struct_history[0] = damping_info['zeta_s']

        # Statistics tracking
        convergence_failures = 0
        max_iterations_used = 0

        # Time integration loop
        for n in range(n_steps - 1):
            current_time = (n + 1) * dt

            # Update time in damping function
            if hasattr(damping_func, 'set_current_time'):
                damping_func.set_current_time(current_time)

            # Step 1: Prediction step
            udot_predict = udot[n] + (1 - gamma) * dt * udotdot[n]
            u_predict = u[n] + dt * udot[n] + (0.5 - beta) * dt**2 * udotdot[n]

            # Initial guess for Newton iteration
            u_iter = u_predict
            udot_iter = udot_predict

            # Modified Newton iteration according to Chopra
            C_eff = damping_func(u_predict, udot_predict)
            K_tangent = K + (1.0 / (beta*dt**2))*M + (gamma / (beta*dt))*C_eff

            # Step 2: Newton iterations to solve non-linear equation
            converged = False
            iteration = 0
            for iteration in range(convergence_params.max_iterations):
                udotdot_iter = (u_iter - u_predict) / (beta * dt**2)
                udot_iter = udot_predict + (gamma / (beta * dt)) * (u_iter - u_predict)

                # Calculate residual: r = f_{n+1} - M*ü_{n+1} - C_eff_star*u̇_{n+1}
                residual = f[n+1] - (M*udotdot_iter + C_eff*udot_iter + K*u_iter)

                # Check convergence
                residual_norm = abs(residual)
                if residual_norm < convergence_params.tolerance:
                    converged = True
                    break

                # Step 3: Solve with frozen tangent
                if abs(K_tangent) < 1e-14:
                    warnings.warn(f"Near-singular tangent stiffness at time step {n+1}, iteration {iteration}")
                    break

                delta_u = residual / K_tangent

                # Step 4: Update guess
                u_iter += delta_u

            # Store iteration statistics
            max_iterations_used = max(max_iterations_used, iteration + 1)

            if not converged:
                convergence_failures += 1
                warnings.warn(f"Newton iteration did not converge at time step {n+1} "
                            f"(residual = {residual_norm:.2e})")
                
            # Step 6: Update solution vectors
            u[n+1] = u_iter
            udot[n+1] = udot_iter
            udotdot[n+1] = (u[n+1] - u_predict) / (beta * dt**2)

            # Store RMS and damping history
            if store_rms_history and hasattr(damping_func, 'vb_damping'):
                vb = damping_func.vb_damping
                rms_history[n+1] = vb.get_current_rms()
                damping_info = vb.get_damping_info(u[n+1], current_time)
                zeta_aero_history[n+1] = damping_info['zeta_a']
                zeta_total_history[n+1] = damping_info['zeta_total']
                zeta_struct_history[n+1] = damping_info['zeta_s']

        # Print statistics if there were convergence issues
        if convergence_failures > 0:
            print(f"   Warning: {convergence_failures}/{n_steps-1} time steps failed to converge")
            print(f"   Maximum iterations used: {max_iterations_used}")

        # Package history data
        history = None
        if store_rms_history:
            history = {
                'rms_history': rms_history,
                'zeta_aero': zeta_aero_history,
                'zeta_total': zeta_total_history,
                'zeta_struct': zeta_struct_history,
                'time': np.arange(n_steps) * dt
            }
            return u, udot, udotdot, history    
        else:
            return u, udot, udotdot
        

class RungeKuttaIntegrator:
    """Simple Runge-Kutta time integrator using scipy's solve_ivp for validation."""

    def integrate_linear(self, M: float, C: float, K: float,
                         f: np.ndarray, u0: float, udot0: float,
                         dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runge-Kutta integration for linear systems using scipy.

        Parameters:
        -----------
        M, C, K : float
            Modal mass, damping, and stiffness
        f : np.ndarray
            Force time series [n_steps]
        u0, udot0 : float
            Initial displacement and velocity
        dt : float
            Time step size [s]
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (displacements, velocities, accelerations)
        """
        # Validate inputs
        f = np.atleast_1d(f).flatten()
        n_steps = len(f)
        M, C, K = float(M), float(C), float(K)

        if dt <= 0:
            raise ValueError("Time step must be positive")
        if M <= 0:
            raise ValueError("Mass must be positive")
        
        # Time vector
        t = np.arange(n_steps) * dt

        # Create interpolated force function
        from scipy.interpolate import interp1d
        force_func = interp1d(t, f, kind='linear', bounds_error=False, fill_value=(f[0], f[-1]))

        # Define ODE system: [u, udot] -> [udot, udotdot]
        # M*udotdot + C*udot + K*u = f(t)
        # udotdot = (f(t) - C*udot - K*u) / M
        def ode_system(t, y):
            u, udot = y
            force = force_func(t)
            udotdot = (force - C * udot - K * u) / M
            return [udot, udotdot]
        
        # Solve using RK45
        sol = solve_ivp(
            ode_system,
            t_span=(t[0], t[-1]),
            y0=[u0, udot0],
            t_eval=t,
            method='RK45',
            rtol=1e-6,
            atol=1e-9,
            max_step=dt
        )
        if not sol.success:
            raise RuntimeError(f"RK45 failed: {sol.message}")

        # Extract solution
        u = sol.y[0, :]
        udot = sol.y[1, :]

        # Calculate accelerations
        udotdot = np.zeros(n_steps)
        for i in range(n_steps):
            udotdot[i] = (f[i] - C * udot[i] - K * u[i]) / M

        return u, udot, udotdot
    
    def integrate_nonlinear(self, M: float, K: float,
                            damping_func: Callable,
                            f: np.ndarray, u0: float, udot0: float,
                            dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Runge-Kutta integration for nonlinear systems using scipy.

        Parameters:
        -----------
        M : float
            Modal mass [kg]
        K : float  
            Modal stiffness [N/m]
        damping_func : callable
            Function C(u, u_dot) returning total damping coefficient [Ns/m]
        f : np.ndarray
            External force time series [N]
        u0 : float
            Initial displacement [m]
        udot0 : float
            Initial velocity [m/s]
        dt : float
            Time step [s]
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (displacements, velocities, accelerations)
        """
        # Validate inputs
        f = np.atleast_1d(f).flatten()
        n_steps = len(f)
        M, K = float(M), float(K)
        
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if M <= 0:
            raise ValueError("Mass must be positive")
        if K <= 0:
            raise ValueError("Stiffness must be positive")
        
        # Time vector
        t = np.arange(n_steps) * dt

        # Create interpolated force function
        from scipy.interpolate import interp1d
        force_func = interp1d(t, f, kind='linear', bounds_error=False, fill_value=(f[0], f[-1]))
        
        # Define ODE system with nonlinear damping
        def ode_system(t_curr, y):
            u, udot = y

            # Update time in damping function if needed
            if hasattr(damping_func, 'set_current_time'):
                damping_func.set_current_time(t_curr)

            C = damping_func(u, udot)
            force = force_func(t_curr)
            udotdot = (force - C * udot - K * u) / M
            return [udot, udotdot]
        
        # Solve using RK45
        sol = solve_ivp(
            ode_system,
            t_span=(t[0], t[-1]),
            y0=[u0, udot0],
            t_eval=t,
            method='RK45',
            rtol=1e-6,
            atol=1e-9,
            max_step=dt
        )
        if not sol.success:
            raise RuntimeError(f"RK45 failed: {sol.message}")

        # Extract solution
        u = sol.y[0, :]
        udot = sol.y[1, :]

        # Calculate accelerations
        udotdot = np.zeros(n_steps)
        for i in range(n_steps):
            if hasattr(damping_func, 'set_current_time'):
                damping_func.set_current_time(t[i])
            C = damping_func(u[i], udot[i])
            udotdot[i] = (f[i] - C * udot[i] - K * u[i]) / M
        
        return u, udot, udotdot

def newmark_integration(M: Union[float, np.ndarray], 
                       C: Union[float, np.ndarray], 
                       K: Union[float, np.ndarray],
                       f: np.ndarray, 
                       u0: float, 
                       udot0: float, 
                       dt: float,
                       gamma: float = 0.5, 
                       beta: float = 0.25,
                       method: str = 'krenk') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function for Newmark integration - optimized for generalized 1-DOF systems.

    Parameters:
    -----------
    M, C, K : float or array_like
        Mass, damping, and stiffness (scalars for generalized SDOF systems)
    f : np.ndarray
        Force time series [n_steps]
    u0, udot0 : float
        Initial displacement and velocity
    dt : float
        Time step size [s]
    gamma, beta : float
        Newmark parameters (default: average constant acceleration)
    method : str
        Integration method ('krenk')
        Note: Removed 'standard' method as it's less efficient
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (displacements, velocities, accelerations) [n_steps] each
    """
    # Create integrator
    params = NewmarkParameters(gamma=gamma, beta=beta)
    integrator = NewmarkIntegrator(params)
    
    # Select method
    if method.lower() == 'krenk':
        return integrator.integrate_krenk(M, C, K, f, u0, udot0, dt)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'krenk'")
