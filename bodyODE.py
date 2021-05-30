import openmdao.api as om
import numpy as np

class body(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('rho', val=1.22, desc='Air density', units='kg/m**3')
        self.add_input('Cd', val=1.0, desc='Drag coefficient', units=None)
        self.add_input('r', val=0.15, desc='Puck Radius', units='m')
        self.add_input('density', val=750, desc='Material density', units='kg/m**3')
        
        #states
        self.add_input('U', val=np.zeros(nn), desc='Longitudinal', units='m/s')
        self.add_input('V', val=np.zeros(nn), desc='Lateral velocity', units='m/s')
        self.add_input('M_fuel', val=np.zeros(nn), desc='Puck mass', units='kg')
        self.add_input('Omega', val=np.zeros(nn), desc='Puck yaw rate', units='rad/s')
        
        #controls
        self.add_input('r_spin', val=np.zeros(nn), desc='Rear spin thruster', units='N')
        self.add_input('thrust', val=np.zeros(nn), desc='longitudinal thrust', units='N')

        #outputs
        self.add_output('Udot', val=np.zeros(nn), desc='Longitudinal', units='m/s**2')
        self.add_output('Vdot', val=np.zeros(nn), desc='Lateral velocity', units='m/s**2')
        self.add_output('Omegadot', val=np.zeros(nn), desc='Puck yaw rate', units='rad/s**2')
        self.add_output('Mdot', val=np.zeros(nn), desc='Mass ejection', units='kg/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #partials
        self.declare_partials(of='Udot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='Udot', wrt='U', rows=arange, cols=arange)
        self.declare_partials(of='Udot', wrt='Omega', rows=arange, cols=arange)
        self.declare_partials(of='Udot', wrt='M_fuel', rows=arange, cols=arange)
        self.declare_partials(of='Udot', wrt='thrust', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='U', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='Omega', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='M_fuel', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='r_spin', rows=arange, cols=arange)
        self.declare_partials(of='Omegadot', wrt='M_fuel', rows=arange, cols=arange)
        self.declare_partials(of='Omegadot', wrt='r_spin', rows=arange, cols=arange)
        self.declare_partials(of='Mdot', wrt='thrust', rows=arange, cols=arange)
        self.declare_partials(of='Mdot', wrt='r_spin', rows=arange, cols=arange)



    def compute(self, inputs, outputs):
        r = inputs['r']
        rho = inputs['rho']
        Cd = inputs['Cd']
        density = inputs['density']

        V = inputs['V']
        U = inputs['U']
        Omega = inputs['Omega']
        M_fuel = inputs['M_fuel']
        thrust = inputs['thrust']
        r_spin = inputs['r_spin']

        A = np.pi*r**2

        M = M_fuel + density*4*np.pi*r**3/3
        Vdrag = V**2*rho*Cd*A/2
        Udrag = U**2*rho*Cd*A/2
        Iz = (2*M*r**2)/5

        outputs['Udot'] = Omega*V + (thrust-Udrag)/M
        outputs['Vdot'] = -Omega*U + (r_spin-Vdrag)/M
        outputs['Omegadot'] = r*(r_spin)/Iz
        outputs['Mdot'] = -(abs(thrust)+abs(r_spin))/10

    def compute_partials(self, inputs, jacobian):
        r = inputs['r']
        rho = inputs['rho']
        Cd = inputs['Cd']
        density = inputs['density']

        V = inputs['V']
        U = inputs['U']
        Omega = inputs['Omega']
        M_fuel = inputs['M_fuel']
        thrust = inputs['thrust']
        r_spin = inputs['r_spin']

        A = np.pi*r**2

        jacobian['Udot', 'V'] = Omega
        jacobian['Udot', 'U'] = -(A*Cd*U*rho)/(M_fuel + (4*density*np.pi*r**3)/3)
        jacobian['Udot', 'Omega'] = V
        jacobian['Udot', 'M_fuel'] = -(thrust - (A*Cd*U**2*rho)/2)/(M_fuel + (4*density*r**3*np.pi)/3)**2
        jacobian['Udot', 'thrust'] = 1/(M_fuel + (4*density*np.pi*r**3)/3)
        jacobian['Vdot', 'V'] = -(A*Cd*V*rho)/(M_fuel + (4*density*np.pi*r**3)/3)
        jacobian['Vdot', 'U'] = -Omega
        jacobian['Vdot', 'Omega'] = -U
        jacobian['Vdot', 'M_fuel'] = (r_spin + (A*Cd*V**2*rho)/2)/(M_fuel + (4*density*r**3*np.pi)/3)**2
        jacobian['Vdot', 'r_spin'] = 1/(M_fuel + (4*density*np.pi*r**3)/3)
        jacobian['Omegadot', 'M_fuel'] = -(10*(r_spin))/(r*(2*M_fuel + (8*density*r**3*np.pi)/3)**2)
        jacobian['Omegadot', 'r_spin'] = 5/(r*(2*M_fuel + (8*density*np.pi*r**3)/3))
        jacobian['Mdot', 'thrust'] = -1/10
        jacobian['Mdot', 'r_spin'] = -1/10
                
        








