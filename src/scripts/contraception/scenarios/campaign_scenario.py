import warnings

from tlo import Date, logging
from tlo.methods import contraception_nuhdss_slums, hiv, demography_nuhdss_slums
from tlo.scenario import BaseScenario




# Ignore warnings to avoid cluttering output from simulation - generally you do not
# need (and generally shouldn't) do this as warnings can contain useful information but
# we will do so here for the purposes of this example to keep things simple.
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))


class PeriodicCampaign(BaseScenario):

    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2030, 1, 1),
            initial_population_size=5000,
            number_of_draws=2,  # Run simulation twice: once without, once with campaign
            runs_per_draw=1,
        )

    def log_configuration(self):
        return {
            'filename': "Periodic_Campaign",  # Unique per draw
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.contraception_nuhdss_slums": logging.INFO,
                "tlo.methods.demography_nuhdss_slums": logging.INFO
            }
        }
    def modules(self):
        return [
            demography_nuhdss_slums.DemographySlums(resourcefilepath=self.resources),
            contraception_nuhdss_slums.ContraceptionSlums(resourcefilepath=self.resources, use_healthsystem=False),
            contraception_nuhdss_slums.SimplifiedPregnancyAndLabour(),
        ]

    def draw_parameters(self, draw_number, rng):
        self.draw_number = draw_number
        return {
            'contraception_nuhdss_slums': {
                'co_contraception': ['default', 'all'][draw_number]
            },
            'interventions_start_date': str(Date(2025, 1, 1))   # Campaign starts in 2025
        }
    
    def run(self, sim):
        """Runs the simulation, scheduling the campaign only for draw 1."""
        if self.draw_number == 1:  # Only for draw 1
            sim.schedule_event(
                contraception_nuhdss_slums.PeriodicCampaignEvent(self),
                self.parameters['interventions_start_date']  # Now correctly defined
            )

        sim.run()
if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])