import openmdao.api as om
import numpy as np
from bodyODE import body
from trackingODE import tracking
from timeODE import TimeODE
from timeAdder import TimeAdder
from curvature import Curvature
from massODE import mass

class CombinedODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='curv',
                           subsys=Curvature(num_nodes=nn), promotes_outputs=['kappa'])

        self.add_subsystem(name='mass',subsys=mass(num_nodes=nn),promotes_inputs=['starting_fuel','M_fuel'],promotes_outputs=['fuel_constraint'])

        self.add_subsystem(name='tracking',  
                           subsys=tracking(num_nodes=nn),promotes_inputs=['V','U','alpha','n','kappa','Omega'],promotes_outputs=['sdot','ndot','alphadot'])
        
        self.add_subsystem(name='body',
                           subsys=body(num_nodes=nn),promotes_inputs = ['V','U','Omega','M_fuel','Cd','r','r_spin','thrust','density'],promotes_outputs=['Omegadot','Vdot','Udot'])

        self.add_subsystem(name='time',subsys=TimeODE(num_nodes=nn),promotes_inputs=['ndot','sdot','Omegadot','Vdot','Udot','alphadot','Mdot'],promotes_outputs=['dn_ds','dV_ds','dOmega_ds','dalpha_ds','dU_ds','dM_ds'])

        self.add_subsystem(name='timeAdder',subsys=TimeAdder(num_nodes=nn),promotes_inputs=['sdot'],promotes_outputs=['dt_ds'])


        
