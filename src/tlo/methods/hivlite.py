
from pathlib import Path
import pandas as pd
from tlo import Module, Parameter, Types, Property, Simulation, Population
from tlo.methods import Metadata
from tlo.events import RegularEvent, PopulationScopeEventMixin

#define a module (HivLite)
class HivLite(Module):
    def __init__(self, resourcefilepath = None):
        super().__init__()

        #define metadat
    METADATA = {
        Metadata.DISEASE_MODULE
        }

    #dedine hivlite parameters, assumptions

    PARAMETERS = {
        "inf_rate": Parameter(Types.REAL, description = "hiv infection rate"),
        'aids_prog_rate': Parameter(Types.REAL, description = 'aids progression rate')

    }

    #characteristics of the individuals
    PROPERTIES = {
        'hl_hiv_status': Property(Types.STRING, description = 'HIV status'),
        'hl_date_inf': Property(Types.DATE, description= 'date of infection')
    }
    def read_parameters(self, data_folder : str | Path)-> None:
        """"read and asign values to parameters"""
        param = self.parameters
        param['inf_rate'] = 0.05
        param['aids_prog_rate'] = 0.01

    def initialise_population (self, population: Population) -> None :
        """assign hiv characteristics to the population""" 
        df = population.props    #create a dataframe for the population
        df.loc[df.is_alive, 'hl_hiv_status'] = "Negative"
        df.loc[df.is_alive, "hl_date_inf"] = pd.NaT

        print(f'hiv status before even {df}')


    def initialise_simulation (self, sim:Simulation):
        """ include all events here"""
        sim.schedule_event(HivInfectionEvent(self), sim.date + pd.DateOffset(months=1))
        

class HivInfectionEvent (RegularEvent, PopulationScopeEventMixin):
    def __int__ (self, module):
        super().__init__(module, frequency=pd.DateOffset(months=1))

    
    def apply(self, target):
        """ include all activities here"""
        #print the default dataframe
        print(f'The population dataframe is {self.module.population.props}')


