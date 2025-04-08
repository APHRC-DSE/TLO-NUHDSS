from tlo import Date, Simulation
#from tests.test_enhanced_lifestyle import resourcefilepath
from tlo.methods import demography_nuhdss_slums, hivlite
from pathlib import Path

start_date = Date(ts_input=2010, year=1, month=1)
end_date = Date(ts_input= 2012,  year=1, month=1)

resourcefilepath = Path('./resources')
#initialising the simulation object/class
sim = Simulation(start_date=start_date, seed=0)

#registering all required modules in simulation
sim.register(demography_nuhdss_slums.DemographySlums(resourcefilepath=resourcefilepath),
             hivlite.HivLite(resourcefilepath=resourcefilepath))

sim.make_initial_population(n=1000)

sim.simulate(end_date=end_date)