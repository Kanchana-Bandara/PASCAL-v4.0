#import sys
#sys.path.append(r'/home/michael/Projects/Migratory_crossroads/Models/pascal_modular/opendrift_pascal')
from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray

import numpy as np

class PascalEnv(Lagrangian3DArray):
    """Extending Lagrangian3DArray with specific properties for biofoulable plastic
    """

    variables = Lagrangian3DArray.add_variables([
        ('unfouled_diameter', {'dtype': np.float32,
                      'units': 'm',
                      'default': 0.0014}),  # 
        ('unfouled_density', {'dtype': np.float32,
                                       'units':'kg/m^3',
                                       'default': 1028}),  # 
        ('biofilm_no_attached_algae', {'dtype': np.float32,
                                       'units': '',
                                       'default': 0}),
        ('total_density', {'dtype': np.float32,
                     'units': 'kg/m^3',
                     'default': 1028.}),
        ('total_diameter', {'dtype': np.float32,
                      'units': 'm',
                      'default': 0.0014})])

class PascalDrift(OceanDrift):
    ElementType = PascalEnv

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        #'sea_surface_wave_significant_height': {'fallback': 0},
        #'sea_ice_area_fraction': {'fallback': 0},
        #'x_wind': {'fallback': 0},
        #'y_wind': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 1000},
        'ocean_vertical_diffusivity': {'fallback': 0.02, 'profiles': True},
        'mld': {'fallback': 50},
        'temperature': {'fallback': 10, 'profiles': True},
        #'sea_water_salinity': {'fallback': 34, 'profiles': True},
        #'surface_downward_x_stress': {'fallback': 0},
        #'surface_downward_y_stress': {'fallback': 0},
        #'turbulent_kinetic_energy': {'fallback': 0},
        #'turbulent_generic_length_scale': {'fallback': 0},
        'upward_sea_water_velocity': {'fallback': 0},
        'food1concentration':{'fallback':0, 'profiles': True},
        'irradiance':{'fallback':0, 'profiles': True},
        'pred1dens':{'fallback':0, 'profiles': True},
        'pred1lightdep':{'fallback':0, 'profiles': True},
      }

    # Default colors for plotting
    status_colors = {'initial': 'green', 'active': 'blue',
                     'hatched': 'red', 'eaten': 'yellow', 'died': 'magenta'}


    def __init__(self, *args, **kwargs):

        # Calling general constructor of parent class
        super(PascalDrift, self).__init__(*args, **kwargs)

    def update(self):
        """Update positions and properties of plastic particles."""
        # Turbulent Mixing
        self.update_terminal_velocity()
        self.vertical_mixing()

        # Horizontal advection
        self.advect_ocean_current()
        
        # Vertical advection
        if self.get_config('drift:vertical_advection') is True:
            self.vertical_advection()

        # Reproduction and reseeding

