import openmdao.api as om
import numpy as np

class TimeODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #states
        self.add_input('sdot', val=np.zeros(nn), desc='distance along track', units='m/s')
        self.add_input('ndot', val=np.zeros(nn), desc='distance perpendicular to centerline', units='m/s')
        self.add_input('alphadot', val=np.zeros(nn), desc='angle relative to centerline', units='rad/s')
        self.add_input('Vdot', val=np.zeros(nn), desc='lateral speed', units='m/s**2')
        self.add_input('Udot', val=np.zeros(nn), desc='longitudinal speed', units='m/s**2')
        self.add_input('Mdot', val=np.zeros(nn), desc='Mass ejection', units='kg/s')
        self.add_input('Omegadot', val=np.zeros(nn), desc='yaw rate', units='rad/s**2')


        #outputs
        self.add_output('dn_ds', val=np.zeros(nn), desc='distance perpendicular to centerline', units='m/m')
        self.add_output('dalpha_ds', val=np.zeros(nn), desc='angle relative to centerline', units='rad/m')
        self.add_output('dV_ds', val=np.zeros(nn), desc='lateral speed', units='1/s')
        self.add_output('dU_ds', val=np.zeros(nn), desc='longitudinal speed', units='1/s')
        self.add_output('dM_ds', val=np.zeros(nn), desc='Mass ejection', units='kg/m')
        self.add_output('dOmega_ds', val=np.zeros(nn), desc='yaw rate', units='rad/(s*m)')


        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #partials

        self.declare_partials(of='dn_ds', wrt='ndot', rows=arange, cols=arange)
        self.declare_partials(of='dn_ds', wrt='sdot', rows=arange, cols=arange)

        self.declare_partials(of='dalpha_ds', wrt='alphadot', rows=arange, cols=arange)
        self.declare_partials(of='dalpha_ds', wrt='sdot', rows=arange, cols=arange)

        self.declare_partials(of='dV_ds', wrt='Vdot', rows=arange, cols=arange)
        self.declare_partials(of='dV_ds', wrt='sdot', rows=arange, cols=arange)
        
        self.declare_partials(of='dU_ds', wrt='Udot', rows=arange, cols=arange)
        self.declare_partials(of='dU_ds', wrt='sdot', rows=arange, cols=arange)

        self.declare_partials(of='dM_ds', wrt='Mdot', rows=arange, cols=arange)
        self.declare_partials(of='dM_ds', wrt='sdot', rows=arange, cols=arange)

        self.declare_partials(of='dOmega_ds', wrt='Omegadot', rows=arange, cols=arange)
        self.declare_partials(of='dOmega_ds', wrt='sdot', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        Omegadot = inputs['Omegadot']
        sdot = inputs['sdot']
        Vdot = inputs['Vdot']
        Udot = inputs['Udot']
        ndot = inputs['ndot']
        Mdot = inputs['Mdot']
        alphadot = inputs['alphadot']

        outputs['dOmega_ds'] = Omegadot/sdot
        outputs['dV_ds'] = Vdot/sdot
        outputs['dU_ds'] = Udot/sdot
        outputs['dM_ds'] = Mdot/sdot
        outputs['dalpha_ds'] = alphadot/sdot
        outputs['dn_ds'] = ndot/sdot
        

    def compute_partials(self, inputs, jacobian):
        Omegadot = inputs['Omegadot']
        sdot = inputs['sdot']
        Vdot = inputs['Vdot']
        Udot = inputs['Udot']
        ndot = inputs['ndot']
        Mdot = inputs['Mdot']
        alphadot = inputs['alphadot']

        jacobian['dn_ds', 'sdot'] = -ndot/sdot**2
        jacobian['dn_ds', 'ndot'] = 1/sdot

        jacobian['dalpha_ds', 'sdot'] = -alphadot/sdot**2
        jacobian['dalpha_ds', 'alphadot'] = 1/sdot

        jacobian['dOmega_ds', 'sdot'] = -Omegadot/sdot**2
        jacobian['dOmega_ds', 'Omegadot'] = 1/sdot

        jacobian['dM_ds', 'sdot'] = -Mdot/sdot**2
        jacobian['dM_ds', 'Mdot'] = 1/sdot

        jacobian['dV_ds', 'sdot'] = -Vdot/sdot**2
        jacobian['dV_ds', 'Vdot'] = 1/sdot

        jacobian['dU_ds', 'sdot'] = -Udot/sdot**2
        jacobian['dU_ds', 'Udot'] = 1/sdot

        








