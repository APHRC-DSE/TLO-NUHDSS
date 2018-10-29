"""
HIV infection event
"""

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin


# read in data files #
file_path = 'Q:/Thanzi la Onse/HIV/Method_HIV.xlsx'  # for desktop
# file_path = '/Users/Tara/Documents/Method_HIV.xlsx'  # for laptop

method_hiv_data = pd.read_excel(file_path, sheet_name=None, header=0)
hiv_prev, hiv_death, hiv_inc, cd4_base, time_cd4, initial_state_probs, \
age_distr = method_hiv_data['prevalence'], method_hiv_data['deaths'], \
            method_hiv_data['incidence'], \
            method_hiv_data['CD4_distribution'], method_hiv_data['Time_spent_by_CD4'], \
            method_hiv_data['Initial_state_probs'], method_hiv_data['age_distribution']


class hiv(Module):
    """
    baseline hiv infection
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'prob_infant_fast_progressor':
            Parameter(Types.LIST, 'Probabilities that infants are fast or slow progressors'),
        'infant_fast_progression':
            Parameter(Types.BOOL, 'Classification of infants as fast progressor'),
        'exp_rate_mort_infant_fast_progressor':
            Parameter(Types.REAL, 'Exponential rate parameter for mortality in infants fast progressors'),
        'weibull_scale_mort_infant_slow_progressor':
            Parameter(Types.REAL, 'Weibull scale parameter for mortality in infants slow progressors'),
        'weibull_shape_mort_infant_slow_progressor':
            Parameter(Types.REAL, 'Weibull shape parameter for mortality in infants slow progressors'),
        'weibull_shape_mort_adult':
            Parameter(Types.REAL, 'Weibull shape parameter for mortality in adults'),
        'proportion_high_sexual_risk_male':
            Parameter(Types.REAL, 'proportion of men who have high sexual risk behaviour'),
        'proportion_high_sexual_risk_female':
            Parameter(Types.REAL, 'proportion of women who have high sexual risk behaviour'),
        'rr_HIV_high_sexual_risk':
            Parameter(Types.REAL, 'relative risk of acquiring HIV with high risk sexual behaviour'),
        'proportion_on_ART_infectious':
            Parameter(Types.REAL, 'proportion of people on ART contributing to transmission as not virally suppressed'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'has_hiv': Property(Types.BOOL, 'HIV status'),
        'date_hiv_infection': Property(Types.DATE, 'Date acquired HIV infection'),
        'date_aids_death': Property(Types.DATE, 'Projected time of AIDS death if untreated'),
        'sexual_risk_group': Property(Types.REAL, 'Relative risk of HIV based on sexual risk high/low'),
        'date_death': Property(Types.DATE, 'Date of death'),
        'cd4_state': Property(Types.CATEGORICAL, 'CD4 state', categories=[500, 350, 250, 200, 100, 50, 0]),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters
        params['prob_infant_fast_progressor'] = [0.36, 1 - 0.36]
        params['infant_progression_category'] = ['FAST', 'SLOW']
        params['exp_rate_mort_infant_fast_progressor'] = 1.08
        params['weibull_scale_mort_infant_slow_progressor'] = 16
        params['weibull_size_mort_infant_slow_progressor'] = 1
        params['weibull_shape_mort_infant_slow_progressor'] = 2.7
        params['weibull_shape_mort_adult'] = 2
        params['proportion_high_sexual_risk_male'] = 0.0913
        params['proportion_high_sexual_risk_female'] = 0.0095
        params['rr_HIV_high_sexual_risk'] = 2
        params['proportion_on_ART_infectious'] = 0.2


    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props
        age = population.age

        df['has_hiv'] = False
        df['date_hiv_infection'] = pd.NaT
        df['date_aids_death'] = pd.NaT
        df['sexual_risk_group'] = 1

        # print('hello')
        self.high_risk(population)  # assign high sexual risk
        self.baseline_prevalence(population)  # allocate baseline prevalence


    def high_risk(self, population):
        """ Stratify the adult (age >15) population in high or low sexual risk """

        df = population.props
        age = population.age

        male_sample = df[(df.sex == 'M') & (age.years >= 15)].sample(
            frac=self.parameters['proportion_high_sexual_risk_male']).index
        female_sample = df[(df.sex == 'F') & (age.years >= 15)].sample(
            frac=self.parameters['proportion_high_sexual_risk_female']).index

        # these individuals have higher risk of hiv
        df.loc[male_sample | female_sample, 'sexual_risk_group'] = self.parameters['rr_HIV_high_sexual_risk']

        print('hurray it works')

    def get_index(self, population, has_hiv, sex, age_low, age_high, cd4_state):

        df = population.props
        age = population.age

        index = df.index[
            df.has_hiv &
            (df.sex == sex) &
            (age.years >= age_low) & (age.years < age_high) &
            (df.cd4_state == cd4_state)]

        return index

    def baseline_prevalence(self, population):
        """
        """

        now = self.sim.date
        df = population.props
        age = population.age

        prevalence = hiv_prev.loc[hiv_prev.year == now.year, ['age_from', 'sex', 'prev_prop']]

        # add age to population.props
        df_with_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        # merge all susceptible individuals with their hiv probability based on sex and age
        df_with_age_hivprob = df_with_age.merge(prevalence,

                                                left_on=['years', 'sex'],

                                                right_on=['age_from', 'sex'],

                                                how='left')

        # no prevalence in ages 80+ so fill missing values with 0
        df_with_age_hivprob['prev_prop'] = df_with_age_hivprob['prev_prop'].fillna(0)

        # print(df_with_age.head(10))
        print(df_with_age_hivprob.head(20))
        # df_with_age_hivprob.to_csv('Q:/Thanzi la Onse/HIV/test.csv', sep=',')  # output a test csv file
        # print(list(df_with_age_hivprob.head(0)))  # prints list of column names in merged df

        assert df_with_age_hivprob.prev_prop.isna().sum() == 0  # check that we found a probability for every individual

        # get a list of random numbers between 0 and 1 for each infected individual
        random_draw = self.rng.random_sample(size=len(df_with_age_hivprob))

        # if random number < probability of HIV, assign has_hiv = True
        hiv_index = df_with_age_hivprob.index[(df_with_age_hivprob.prev_prop > random_draw)]

        print(hiv_index)
        test = hiv_index.isnull().sum()  # sum number of nan
        print("number of nan: ", test)

        df.loc[hiv_index, 'has_hiv'] = True
        df.loc[hiv_index, 'date_hiv_infection'] = now


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        event = hiv_event(self)
        sim.schedule_event(event, sim.date + DateOffset(months=12))


    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        pass


class hiv_event(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event
    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """One line summary here
        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props

        # 1. get (and hold) index of currently uninfected individuals
        currently_uninfected = df.index[~df.has_hiv]

        # 2. handle new infections
        now_infected = np.random.choice([True, False], size=len(currently_uninfected),
                                        p=[0.1, 0.9])
        # if any are infected
        if now_infected.sum():
            infected_idx = currently_uninfected[now_infected]

            df.loc[infected_idx, 'has_hiv'] = True
            df.loc[infected_idx, 'date_hiv_infection'] = self.sim.date


