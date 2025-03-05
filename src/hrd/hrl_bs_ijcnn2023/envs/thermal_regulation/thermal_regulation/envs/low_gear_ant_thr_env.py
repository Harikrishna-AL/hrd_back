from gym import utils

from thermal_regulation.envs.thermal_regulation_env import ThermalRegulationEnv
from thermal_regulation.envs.ant_thr_env import MyAntEnv


class MyLowGearAntEnv(MyAntEnv, utils.EzPickle):
    # TODO: MaKe low-gear version as an option in AntGather
    FILE = "low_gear_ratio_ant.xml"


class LowGearAntThermalRegulationEnv(ThermalRegulationEnv):
    """
    Two Resource Problem with approximated scale of Konidaris & Barto paper
    """
    MODEL_CLASS = MyLowGearAntEnv
    ORI_IND = 3


class LowGearAntSmallThermalRegulationEnv(ThermalRegulationEnv):
    """
    Small-sized Two Resource Problem with the scale of original GatherEnv
    """
    MODEL_CLASS = MyLowGearAntEnv
    ORI_IND = 3

    def __init__(self,
                 activity_range=6.,
                 n_bins=20,
                 sensor_range=12.,
                 *args, **kwargs):
        super().__init__(
            n_bins=n_bins,
            activity_range=activity_range,
            sensor_range=sensor_range,
            *args, **kwargs
        )
