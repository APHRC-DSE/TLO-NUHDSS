"""
running demographt module for nuhdss to check on simulated data
"""
from tlo import Date, logging
from tlo.methods import demography_nuhdss, contraception_nuhdss, hiv
from tlo.scenario import BaseScenario


class RunAnalysisCo(BaseScenario):
    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2040, 12, 31),
            initial_population_size=10000,  # selected size for the Tim C at al. 2023 paper: 250K
            number_of_draws=1,  # <- one scenario
            runs_per_draw=1,  # <- repeated this many times
        )

    def log_configuration(self):
        return {
            'filename': 'run_analysis_nuhdss',  # <- (specified only for local running)
            'directory': './outputs',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.demography_nuhdss": logging.INFO,
                 "tlo.methods.contraception_nuhdss": logging.INFO
            }
        }

    def modules(self):
        return [
             # Core Modules
            demography_nuhdss.DemographySlums(resourcefilepath=self.resources),
            #healthsystem.HealthSystem(resourcefilepath=self.resources,
             #                         cons_availability="all"),

            # - Contraception and replacement for Labour etc.
            contraception_nuhdss.ContraceptionSlums(resourcefilepath=self.resources,
                                         use_healthsystem=False  # default: True <-- using HealthSystem
            #                             # if True initiation and switches to contraception require an HSI
                                        ),
            contraception_nuhdss.SimplifiedPregnancyAndLabour(),

            # # - Supporting Module required by Contraception
             hiv.DummyHivModule(),
        ]
        

    def draw_parameters(self, draw_number, rng):
        # Using default parameters in all cases
        return


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
