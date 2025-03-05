"""
Thermal model of the robot. Parameters adopted from:
Fei, Teng, et al. "A body temperature model for lizards as estimated from the thermal environment." Journal of Thermal Biology 37.1 (2012): 56-64.

Torque-Heat model is inspired by the thermo dynamic model of the electric motor
Venkataraman, B., et al. "Fundamentals of a motor thermal model and its applications in motor protection." 58th Annual Conference for Protective Relay Engineers, 2005.. IEEE, 2005.

Absolute temperature is used only inside of the module and all methods use Celsius degree.
"""

import numpy as np
from enum import Enum, auto


class Climate(Enum):
    default = auto()
    sunny = auto()
    cloudy = auto()


class ThermalModule():
    M = 0.19  # [kg] lizard data. (ant torso mass is np.pi * (3/4) * 0.25 ** 3 * 5 = 0.184 [kg])
    alpha_L = 0.936  # [-]
    C_l = 3762  # [JK^-1kg^-1]
    delta = 0.015  # [m]
    K_l = 0.502  # [WK^-1m^-1]
    h_L = 10.45  # [Wm^-2K^-1]
    a = 0.0314  # shape coeff
    A_L = a * np.pi * M ** (2./3)  # [m^2]
    A_p = 0.4 * A_L  # [m^2]
    A_down = 0.3 * A_L  # [m^2]
    A_up = 0.6 * A_L  # [m^2]
    A_air = 0.9 * A_L  # [m^2]
    A_contact = 0.1 * A_L  # [m^2]
    eps_skin = 0.95  # [-]
    eps_land = 0.95  # [-]
    sigma = 5.67e-8  # Stefan-Boltzmann constant [Wm^-2K^-4]

    motor_coef = 5.  # Motor-heat coefficient

    initial_T = 311.  # default. 38 C

    def __init__(self, temp_init=38.):
        """

        :param temp_init: Initial body temperature in Celsius degree
        """
        self.T = np.array(temp_init + 273.)
        self.initial_T = np.array(temp_init + 273.)

    def reset(self, temp_init=None):
        """

        :param temp_init: Initial body temperature in Celsius degree
        :return:
        """
        if temp_init:
            self.initial_T = np.array(temp_init + 273.)

        self.T = np.array(self.initial_T)

    def _dQ_solar(self, climate: Climate):
        if climate is Climate.default:
            Q_solar_now = 300  # [Wm^-2]  (300 in original paper)
        elif climate is Climate.sunny:
            Q_solar_now = 600  # [Wm^-2] Assuming many cloud though. Full sunny day means ~1000 but agent cannot control in our environment!
        elif climate is Climate.cloudy:
            Q_solar_now = 300  # [Wm^-2]
        else:
            raise ValueError(f"Invalid climate: {climate}")

        return self.alpha_L * self.A_p * Q_solar_now

    def _dQ_conv(self, T_now, temp_air):
        return self.h_L * self.A_air * (temp_air - T_now)

    def _dQ_longwave(self, T_now, temp_air, temp_earth):
        q_earth = self.eps_land * self.A_down * self.sigma * (temp_earth ** 4 - T_now ** 4)
        q_air = self.eps_skin * self.A_up * self.sigma * (temp_air ** 4 - T_now ** 4)
        return q_earth + q_air

    def _dQ_cond(self, T_now, temp_earth):
        return self.A_contact * self.K_l * (temp_earth - T_now) / (self.delta / 2.)

    def _delta_Q(self, T_now, action, evaporative_action, temp_air, temp_earth, climate):
        """
        Assuming all actions sould be scaled into [-1, +1]
        """

        dq1 = self._dQ_solar(climate) # solar ratiation
        dq2 = self._dQ_conv(T_now, temp_air)  # convection heat
        dq3 = self._dQ_longwave(T_now, temp_air, temp_earth)  # long-wave heat
        dq4 = self._dQ_cond(T_now, temp_earth)  # conductive heat
        dq5 = self.motor_coef * sum(np.square(action))  # motor heat production

        max_ev = 0.3
        min_ev = 0.272 * self.M
        dq6 = 0.5 * (max_ev - min_ev) * (evaporative_action + 1) + min_ev

        dQ = dq1 + dq2 + dq3 + dq4 + dq5 - dq6
        return dQ

    def _grad_T(self, T_now, action, evaporative_action, temp_air, temp_earth, climate):
        return self._delta_Q(T_now, action, evaporative_action, temp_air, temp_earth, climate) / (self.C_l * self.M)

    def step(self, motor_action, evaporative_action, temp_air_c, temp_earth_c, dt, climate=Climate.default, mode="RK4"):
        """
        One-step progress of the thermal model. return the latest body temperature in Celsius degree
        :param motor_action:
        :param evaporative_action:
        :param temp_air_c: air/sky temperature in Celsius degree
        :param temp_earth_c: earth/soil temperature in Celsius degree
        :param dt: time tick of the thermal model
        :param climate: Climate to choose solar radiation
        :param mode: Choice of numerical simulation methods. 4th order Runge-Kutta (RK4) is default.
        :return:
        """

        # Eular
        if mode == "Eular":
            T_now = self.T
            dT = self._grad_T(T_now, motor_action, evaporative_action, temp_air_c + 273., temp_earth_c + 273., climate)
            dT *= dt
        elif mode == "RK4":
            # RK4 based on https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method
            T_now = self.T
            k1 = self._grad_T(T_now, motor_action, evaporative_action, temp_air_c + 273., temp_earth_c + 273., climate)
            k2 = self._grad_T(T_now + dt * k1 / 2., motor_action, evaporative_action, temp_air_c + 273., temp_earth_c + 273., climate)
            k3 = self._grad_T(T_now + dt * k2 / 2., motor_action, evaporative_action, temp_air_c + 273., temp_earth_c + 273., climate)
            k4 = self._grad_T(T_now + dt * k3, motor_action, evaporative_action, temp_air_c + 273., temp_earth_c + 273., climate)
            #print(k1, k2, k3, k4)
            dT = (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6.
            #print("ave_k; ", (k1 + 2 * k2 + 2 * k3 + k4)/6.)
        else:
            raise ValueError("mode error")

        self.T += dT

        return self.T - 273.0

    def get_temp_now(self):
        return self.T.copy() - 273.0
