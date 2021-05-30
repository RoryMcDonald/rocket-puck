import openmdao.api as om
import numpy as np

class tracking(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #states
        self.add_input('U', val=np.zeros(nn), desc='Longitudinal', units='m/s')
        self.add_input('V', val=np.zeros(nn), desc='Lateral velocity', units='m/s')
        self.add_input('alpha', val=np.zeros(nn), desc='Puck heading', units='rad')
        self.add_input('n', val=np.zeros(nn), desc='Road lateral position', units='m')
        self.add_input('kappa', val=np.zeros(nn), desc='Road curvature', units='1/m')
        self.add_input('Omega', val=np.zeros(nn), desc='Puck yaw rate', units='rad/s')

        #outputs
        self.add_output('sdot', val=np.zeros(nn), desc='distance along track', units='m/s')
        self.add_output('ndot', val=np.zeros(nn), desc='distance perpendicular to centerline', units='m/s')
        self.add_output('alphadot', val=np.zeros(nn), desc='angle relative to centerline', units='rad/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #partials
        self.declare_partials(of='sdot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='U', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='alpha', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='n', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='kappa', rows=arange, cols=arange)
        self.declare_partials(of='ndot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='ndot', wrt='U', rows=arange, cols=arange)
        self.declare_partials(of='ndot', wrt='alpha', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='U', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='alpha', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='n', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='kappa', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='Omega', rows=arange, cols=arange)
                


    def compute(self, inputs, outputs):
        V = inputs['V']
        U = inputs['U']
        alpha = inputs['alpha']
        n = inputs['n']
        kappa = inputs['kappa']
        Omega = inputs['Omega']

        outputs['sdot'] = (U*np.cos(alpha)-V*np.sin(alpha))/(1-n*kappa)
        outputs['ndot'] = U*np.sin(alpha) + V*np.cos(alpha)
        outputs['alphadot'] =  Omega - kappa*(V*np.cos(alpha)-U*np.sin(alpha))/(1-n*kappa) 


    def compute_partials(self, inputs, jacobian):
        V = inputs['V']
        U = inputs['U']
        alpha = inputs['alpha']
        n = inputs['n']
        kappa = inputs['kappa']
        Omega = inputs['Omega']

        jacobian['sdot', 'V'] = np.sin(alpha)/(kappa*n - 1)
        jacobian['sdot', 'U'] = -np.cos(alpha)/(kappa*n - 1)
        jacobian['sdot', 'alpha'] = (V*np.cos(alpha) + U*np.sin(alpha))/(kappa*n - 1)
        jacobian['sdot', 'n'] = (kappa*(U*np.cos(alpha) - V*np.sin(alpha)))/(kappa*n - 1)**2
        jacobian['sdot', 'kappa'] = (n*(U*np.cos(alpha) - V*np.sin(alpha)))/(kappa*n - 1)**2
        jacobian['ndot', 'V'] = np.cos(alpha)
        jacobian['ndot', 'U'] = np.sin(alpha)
        jacobian['ndot', 'alpha'] = U*np.cos(alpha) - V*np.sin(alpha)
        jacobian['alphadot', 'V'] = (kappa*np.cos(alpha))/(kappa*n - 1)
        jacobian['alphadot', 'U'] = -(kappa*np.sin(alpha))/(kappa*n - 1)
        jacobian['alphadot', 'alpha'] = -(kappa*(U*np.cos(alpha) + V*np.sin(alpha)))/(kappa*n - 1)
        jacobian['alphadot', 'n'] = -(kappa**2*(V*np.cos(alpha) - U*np.sin(alpha)))/(kappa*n - 1)**2
        jacobian['alphadot', 'kappa'] = (V*np.cos(alpha) - U*np.sin(alpha))/(kappa*n - 1) - (kappa*n*(V*np.cos(alpha) - U*np.sin(alpha)))/(kappa*n - 1)**2
        jacobian['alphadot', 'Omega'] = 1

        
        








