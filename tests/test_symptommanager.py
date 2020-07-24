import os
from pathlib import Path

from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    labour,
    mockitis,
    pregnancy_supervisor,
    symptommanager,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = './resources'

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)
popsize = 200


def test_no_symptoms_if_no_diseases():
    sim = Simulation(start_date=start_date)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    generic_symptoms = list(sim.modules['SymptomManager'].parameters['generic_symptoms'])

    for symp in generic_symptoms:
        # No one should have any symptom currently (as no disease modules registered)
        assert list() == sim.modules['SymptomManager'].who_has(symp)
        # *** Errors: who_has return the list of all alive persons


def test_adding_symptoms():
    sim = Simulation(start_date=start_date)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(),
                 dx_algorithm_child.DxAlgorithmChild(),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 mockitis.Mockitis())

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Check that symptoms are add to checked and removed correctly
    df = sim.population.props

    symp = 'backache'  # this is a generic symptoms that is not added by any disease module that is registered

    # No one should have any symptom currently
    assert list() == sim.modules['SymptomManager'].who_has(symp)

    # check adding symptoms
    ids = list(sim.rng.choice(list(df.index[df.is_alive]), 5))

    sim.modules['SymptomManager'].change_symptom(
        symptom_string=symp,
        person_id=ids,
        add_or_remove='+',
        disease_module=sim.modules['Mockitis']
    )

    has_symp = sim.modules['SymptomManager'].who_has(symp)
    assert set(has_symp) == set(ids)

    # check causes of the symptoms:
    for person in ids:
        causes = sim.modules['SymptomManager'].causes_of(person, symp)
        assert 'Mockitis' in causes
        assert 1 == len(causes)

    # Remove the symptoms:
    for person in ids:
        sim.modules['SymptomManager'].clear_symptoms(person, disease_module=sim.modules['Mockitis'])

    assert list() == sim.modules['SymptomManager'].who_has(symp)