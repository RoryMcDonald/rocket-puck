import openmdao.api as om
import numpy as np

class mass(om.ExplicitComponent):
    #This serves to keep track of the elapsed time. Since we performed a change of variables we need to integrate time by adding dt/ds

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('starting_fuel', val=100,  desc='starting fuel mass', units='kg')

        #states
        self.add_input('M_fuel', val=np.zeros(nn), desc='fuel mass', units='kg')
        
        #outputs
        self.add_output('fuel_constraint', val=np.zeros(nn), desc='constrains the starting fuel to 1', units=None)


        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #partials
        self.declare_partials(of='fuel_constraint', wrt='M_fuel', rows=arange, cols=arange)
        self.declare_partials(of='fuel_constraint', wrt='starting_fuel')


    def compute(self, inputs, outputs):
        starting_fuel = inputs['starting_fuel']
        M_fuel = inputs['M_fuel']

        outputs['fuel_constraint'] = M_fuel/starting_fuel

    def compute_partials(self, inputs, jacobian):
        starting_fuel = inputs['starting_fuel']
        M_fuel = inputs['M_fuel']

        jacobian['fuel_constraint', 'starting_fuel'] = -M_fuel/starting_fuel**2
        jacobian['fuel_constraint', 'M_fuel'] = 1/starting_fuel

        








