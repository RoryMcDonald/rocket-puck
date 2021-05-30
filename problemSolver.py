import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from combinedODE import CombinedODE
import matplotlib as mpl

#track curvature imports
from scipy import interpolate
from scipy import signal
from Track import Track
import tracks
from spline import getSpline,getTrackPoints,getGateNormals,reverseTransformGates,setGateDisplacements,transformGates
from linewidthhelper import *

print('Config: RWD single thrust')

track = tracks.ovaltrack #change track here and in curvature.py. Tracks are defined in tracks.py
plot = True #plot track and telemetry

points = getTrackPoints(track) #generate nodes along the centerline for curvature calculation (different than collocation nodes)
finespline,gates,gatesd,curv,slope = getSpline(points,s=0.0) #fit the centerline spline. by default 10000 points
s_final = track.getTotalLength()

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(ode_class=CombinedODE,
		     transcription=dm.Radau(num_segments=30, order=3,compressed=True))

traj.add_phase(name='phase0', phase=phase)

phase.set_refine_options(refine = True, tol = 1e-7, min_order = 3, max_order=9)#, smoothness_factor= 1.2)

# Set the time options, in this problem we perform a change of variables. So 'time' is actually 's' (distance along the track centerline)
# This is done to fix the collocation nodes in space, which saves us the calculation of the rate of change of curvature.
# The state equations are written with respect to time, the variable change occurs in timeODE.py
phase.set_time_options(fix_initial=True,fix_duration=True,duration_val=s_final,targets=['curv.s'],units='m',duration_ref=s_final,duration_ref0=s_final/100)

#Define states
phase.add_state('t', fix_initial=True, fix_final=False, units='s', lower = 0,rate_source='dt_ds',ref=s_final/100) #time
phase.add_state('n', fix_initial=True, fix_final=False, units='m', upper = 4.0, lower = -4.0, rate_source='dn_ds',targets=['n'],ref=4.0) #normal distance to centerline. The bounds on n define the width of the track
phase.add_state('U', fix_initial=True, fix_final=False, units='m/s', ref = 10, ref0=5,rate_source='dU_ds', targets=['U']) #lateral velocity
phase.add_state('V', fix_initial=True, fix_final=False, units='m/s', lower = 0.1, ref = 5, ref0=1, rate_source='dV_ds', targets=['V']) #longitudinal velocity
phase.add_state('alpha', fix_initial=True, fix_final=False, units='rad', rate_source='dalpha_ds',targets=['alpha'],ref=0.015) #vehicle heading angle with respect to centerline
phase.add_state('Omega', fix_initial=True, fix_final=False, units='rad/s',rate_source='dOmega_ds',targets=['Omega'],ref=0.01) #yaw rate
phase.add_state('M_fuel', fix_initial=True, fix_final=False, units='kg', lower = 0, rate_source='dM_ds',targets=['M_fuel'],ref=0.3) #yaw rate


#Define Controls
phase.add_control(name='r_spin', lower = -1, upper = 1, units='N',fix_initial=False,fix_final=False, targets=['r_spin'],ref = 0.01) #
phase.add_control(name='thrust', lower = -100, upper = 100, units='N',fix_initial=False,fix_final=False, targets=['thrust'],ref = 80) #the thrust controls the longitudinal force of the rear tires and is positive while accelerating, negative while braking


#Some of the vehicle design parameters are available to set here. Other parameters can be found in their respective ODE files.
#phase.add_parameter('starting_fuel',val=10.0,units='kg',opt=False,targets=['starting_fuel'],dynamic=False) #vehicle mass
#phase.add_parameter('r',val=0.15,units='m',opt=False,targets=['r'],dynamic=False) #vehicle mass
#phase.add_parameter('Cd',val=1.0,units=None,opt=False,targets=['Cd'],dynamic=False) #drag coefficient*area
#phase.add_parameter('density',val=750.0,units='kg/m**3',opt=False,targets=['density'],dynamic=False) 




#Minimize final time.
phase.add_objective('t', loc='final') #note that we use the 'state' time instead of Dymos 'time'

#Add output timeseries
phase.add_timeseries_output('*')


#Link the states at the start and end of the phase in order to ensure a continous lap
#traj.link_phases(phases=['phase0', 'phase0'], vars=['V','n','alpha','Omega','lambda','ax','ay'], locs=('++', '--'))

# Set the driver. IPOPT or SNOPT are recommended but SLSQP might work.
IPOPT = True
if IPOPT:
	p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
	p.driver.opt_settings['linear_solver'] = 'ma27'
	p.driver.opt_settings['mu_init'] = 1e-3
	p.driver.opt_settings['max_iter'] = 500
	p.driver.opt_settings['acceptable_tol'] = 1e-3
	p.driver.opt_settings['constr_viol_tol'] = 1e-3
	p.driver.opt_settings['compl_inf_tol'] = 1e-3
	p.driver.opt_settings['acceptable_iter'] = 0
	p.driver.opt_settings['tol'] = 1e-3
	p.driver.opt_settings['hessian_approximation'] = 'exact'
	p.driver.opt_settings['nlp_scaling_method'] = 'none'
	p.driver.opt_settings['print_level'] = 5
else:
	p.driver = om.ScipyOptimizeDriver()
	p.driver.options['optimizer'] = 'SLSQP'
	p.driver.options['tol'] = 1e-9
	p.driver.options['disp'] = True
	p.driver.options['maxiter'] = 500

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of Dymos.
p.driver.declare_coloring()

# Setup the problem
p.setup(check=True) #force_alloc_complex=True
# Now that the OpenMDAO problem is setup, we can set the values of the states.

#States
p.set_val('traj.phase0.states:U',phase.interpolate(ys=[1.0,20], nodes='state_input'),units='m/s') #non-zero velocity in order to protect against 1/0 errors.
p.set_val('traj.phase0.states:V',phase.interpolate(ys=[0.0,20], nodes='state_input'),units='m/s') #non-zer o velocity in order to protect against 1/0 errors.
p.set_val('traj.phase0.states:Omega',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad/s')
p.set_val('traj.phase0.states:alpha',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad')
p.set_val('traj.phase0.states:n',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='m')
p.set_val('traj.phase0.states:t',phase.interpolate(ys=[0.0,10.0], nodes='state_input'),units='s') #initial guess for what the final time should be
p.set_val('traj.phase0.states:M_fuel',phase.interpolate(ys=[10.0,10.0], nodes='state_input'),units='kg') #initial guess for what the final time should be

#Controls
p.set_val('traj.phase0.controls:r_spin',phase.interpolate(ys=[0.0, 0.1], nodes='control_input'),units='N') #
p.set_val('traj.phase0.controls:thrust',phase.interpolate(ys=[0.1, 0.1], nodes='control_input'),units='N') #

dm.run_problem(p, run_driver=True, simulate=False, refine_iteration_limit=5)
#p.run_model()
print('Optimization finished')

#Get optimized time series
n = p.get_val('traj.phase0.timeseries.states:n')
t = p.get_val('traj.phase0.timeseries.states:t')
s = p.get_val('traj.phase0.timeseries.time')
V = p.get_val('traj.phase0.timeseries.states:V')
thrust = p.get_val('traj.phase0.timeseries.controls:thrust')

#Plotting
if plot:
	print("Plotting")


	#We know the optimal distance from the centerline (n). To transform this into the racing line we fit a spline to the displaced points. This will let us plot the racing line in x/y coordinates
	trackLength = track.getTotalLength()
	normals = getGateNormals(finespline,slope)
	newgates = []
	newnormals = []
	newn = []
	for i in range(len(n)):
		index = ((s[i]/s_final)*np.array(finespline).shape[1]).astype(int) #interpolation to find the appropriate index
		if index[0]==np.array(finespline).shape[1]:
			index[0] = np.array(finespline).shape[1]-1
		if i>0 and s[i] == s[i-1]:
			continue
		else:
			newgates.append([finespline[0][index[0]],finespline[1][index[0]]])
			newnormals.append(normals[index[0]])
			newn.append(n[i][0])

	newgates = reverseTransformGates(newgates)
	displacedGates = setGateDisplacements(newn,newgates,newnormals)
	displacedGates = np.array((transformGates(displacedGates)))

	npoints = 1000
	displacedSpline,gates,gatesd,curv,slope = getSpline(displacedGates,1/npoints,0) #fit the racing line spline to npoints

	plt.rcParams.update({'font.size': 12})


	def plotTrackWithData(state,s):
		#this function plots the track
		state = np.array(state)[:,0]
		s = np.array(s)[:,0]
		s_new = np.linspace(0,s_final,npoints)

		#Colormap and norm of the track plot
		cmap = mpl.cm.get_cmap('viridis')
		norm = mpl.colors.Normalize(vmin=np.amin(state),vmax=np.amax(state))

		fig, ax = plt.subplots(figsize=(15,6))
		plt.plot(displacedSpline[0],displacedSpline[1],linewidth=0.1,solid_capstyle="butt") #establishes the figure axis limits needed for plotting the track below

		plt.axis('equal')
		plt.plot(finespline[0],finespline[1],'k',linewidth=linewidth_from_data_units(8.5,ax),solid_capstyle="butt") #the linewidth is set in order to match the width of the track
		plt.plot(finespline[0],finespline[1],'w',linewidth=linewidth_from_data_units(8,ax),solid_capstyle="butt") #8 is the width, and the 8.5 wide line draws 'kerbs'
		plt.xlabel('x (m)')
		plt.ylabel('y (m)')

		#plot spline with color
		for i in range(1,len(displacedSpline[0])):
			s_spline = s_new[i]
			index_greater = np.argwhere(s>=s_spline)[0][0]
			index_less = np.argwhere(s<s_spline)[-1][0]

			x = s_spline
			xp = np.array([s[index_less],s[index_greater]])
			fp = np.array([state[index_less],state[index_greater]])
			interp_state = np.interp(x,xp,fp) #interpolate the given state to calculate the color

			#calculate the appropriate color
			state_color = norm(interp_state)
			color = cmap(state_color)
			color = mpl.colors.to_hex(color)

			#the track plot consists of thousands of tiny lines:
			point = [displacedSpline[0][i],displacedSpline[1][i]]
			prevpoint = [displacedSpline[0][i-1],displacedSpline[1][i-1]]
			if i <=5 or i == len(displacedSpline[0])-1:
				plt.plot([point[0],prevpoint[0]],[point[1],prevpoint[1]],color,linewidth=linewidth_from_data_units(1.5,ax),solid_capstyle="butt",antialiased=True)
			else:
				plt.plot([point[0],prevpoint[0]],[point[1],prevpoint[1]],color,linewidth=linewidth_from_data_units(1.5,ax),solid_capstyle="projecting",antialiased=True)

		clb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),fraction = 0.02, pad=0.04) #add colorbar

		if np.array_equal(state,V[:,0]):
			clb.set_label('Velocity (m/s)')
		elif np.array_equal(state,thrust[:,0]):
			clb.set_label('Thrust')

		plt.tight_layout()
		plt.grid()

	#Create the plots
	plotTrackWithData(V,s)
	plotTrackWithData(thrust,s)


	#Plot the main vehicle telemetry
	fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 8))

	#U vs s
	axes[0].plot(s,
			p.get_val('traj.phase0.timeseries.states:U'), label='solution')

	axes[0].set_xlabel('s (m)')
	axes[0].set_ylabel('U (m/s)')
	axes[0].grid()
	axes[0].set_xlim(0,s_final)

	#V vs s
	axes[1].plot(s,
			p.get_val('traj.phase0.timeseries.states:V'), label='solution')

	axes[1].set_xlabel('s (m)')
	axes[1].set_ylabel('V (m/s)')
	axes[1].grid()
	axes[1].set_xlim(0,s_final)

	#thrust vs s
	axes[2].plot(s,thrust)

	axes[2].set_xlabel('s (m)')
	axes[2].set_ylabel('thrust')
	axes[2].grid()
	axes[2].set_xlim(0,s_final)

	#sdot vs s
	axes[3].plot(s,p.get_val('traj.phase0.timeseries.sdot'))

	axes[3].set_xlabel('s (m)')
	axes[3].set_ylabel('sdot')
	axes[3].grid()
	axes[3].set_xlim(0,s_final)



	fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 8))
	#thrust vs s
	axes[0].plot(s,p.get_val('traj.phase0.timeseries.controls:r_spin'))

	axes[0].set_xlabel('s (m)')
	axes[0].set_ylabel('Spinners')
	axes[0].grid()
	axes[0].set_xlim(0,s_final)
	axes[0].legend()

	axes[1].plot(s,p.get_val('traj.phase0.timeseries.states:Omega'))

	axes[1].set_xlabel('s (m)')
	axes[1].set_ylabel('Yaw Rate')
	axes[1].grid()
	axes[1].set_xlim(0,s_final)

	axes[2].plot(s,p.get_val('traj.phase0.timeseries.states:alpha'))

	axes[2].set_xlabel('s (m)')
	axes[2].set_ylabel('Heading')
	axes[2].grid()
	axes[2].set_xlim(0,s_final)

	#fuel vs s
	axes[3].plot(s,p.get_val('traj.phase0.timeseries.states:M_fuel'))

	axes[3].set_xlabel('s (m)')
	axes[3].set_ylabel('sdot')
	axes[3].grid()
	axes[3].set_xlim(0,s_final)

	plt.tight_layout()
	plt.show()