import numbers
from enum import Enum, auto
from typing import Any, Callable, Union

import numpy as np
import pandas as pd

from tlo import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Predictor(object):
    def __init__(self, property_name: str = None, external: bool = False):
        """A Predictor variable for the regression model. The property_name is a property of the
         population dataframe e.g. age, sex, etc."""
        self.property_name = property_name

        # If this is a property that is not part of the population dataframe
        if external:
            assert property_name is not None, "Can't have an unnamed external predictor"
            # It will be an private column appended to the dataframe
            self.property_name = f'__{self.property_name}__'

        self.conditions = list()
        self.callback = None
        self.has_otherwise = False

    def when(self, condition: Union[str, float, bool], value: float) -> 'Predictor':
        assert self.callback is None, "Can't use `when` on Predictor with function"
        return self._coeff(condition=condition, coefficient=value)

    def otherwise(self, value: float) -> 'Predictor':
        assert self.property_name is not None, "Can't use `otherwise` condition on unnamed Predictor"
        assert self.callback is None, "Can't use `otherwise` on Predictor with function"
        return self._coeff(coefficient=value)

    def apply(self, callback: Callable[[Any], float]) -> 'Predictor':
        assert self.property_name is not None, "Can't use `apply` on unnamed Predictor"
        assert len(self.conditions) == 0, "Can't specify `apply` on Predictor with when/otherwise conditions"
        assert self.callback is None, "Can't specify more than one callback for a Predictor"
        self.callback = callback
        return self

    def _coeff(self, *, coefficient, condition=None) -> 'Predictor':
        """Adds the coefficient for the Predictor. The arguments can be two:
                `coeff(condition, value)` where the condition evaluates the property value to true/false
                `coeff(value)` where the value is given to all unconditioned values of the property
        The second style (unconditioned value) only makes sense after one or more conditioned values
        """
        # If there isn't a property name
        if self.property_name is None:
            # We use the supplied condition literally
            self.conditions.append((condition, coefficient))
            return self

        # Otherwise, the condition is applied on a specific property
        if isinstance(condition, str):
            # Handle either a complex condition (begins with an operator) or implicit equality
            if condition[0] in ['!', '=', '<', '>', '~', '(', '.']:
                parsed_condition = f'({self.property_name}{condition})'
            else:
                # numeric values don't need to be quoted
                if condition.isnumeric():
                    parsed_condition = f'({self.property_name} == {condition})'
                else:
                    parsed_condition = f'({self.property_name} == "{condition}")'
        elif isinstance(condition, bool):
            if condition:
                parsed_condition = f'({self.property_name} == True)'
            else:
                parsed_condition = f'({self.property_name} == False)'
        elif isinstance(condition, numbers.Number):
            parsed_condition = f'({self.property_name} == {condition})'
        elif condition is None:
            assert not self.has_otherwise, "You can only give one unconditioned value to predictor"
            self.has_otherwise = True
            parsed_condition = None
        else:
            raise RuntimeError(f"Unhandled condition: {condition}")

        self.conditions.append((parsed_condition, coefficient))
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Will add the value(s) of this predictor to the output series, checking values in the supplied
        dataframe"""

        # We want to "short-circuit" values i.e. if an individual in a population matches a certain
        # condition, we don't want that individual to be matched in any subsequent conditions

        # result of assigning coefficients on this predictor
        output = pd.Series(data=np.nan, index=df.index)

        # if this predictor's coefficient is calculated by callback, run it and return
        if self.callback:
            # note that callback only works on single column
            output = df[self.property_name].apply(self.callback)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(key='debug', data=f'predictor: {self.property_name}; function: {self.callback}; '
                                               f'matched: {len(df)}')
            return output

        # keep a record of all rows matched by this predictor's conditions
        matched = pd.Series(False, index=df.index)

        for condition, value in self.conditions:
            # don't include rows that were matched by a previous condition
            if condition:
                unmatched_condition = f'{condition} & (~@matched)'
            else:
                unmatched_condition = '~@matched'

            # rows matching the current conditional
            mask = df.eval(unmatched_condition)

            # test if mask includes rows that were already matched by a previous condition
            assert not (matched & mask).any(), f'condition "{unmatched_condition}" matches rows already matched'

            # update elements in the output series with the corresponding value for the condition
            output[mask] = value

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(key='debug', data=f'predictor: {self.property_name}; condition: {condition}; '
                                               f'value: {value}; matched rows: {mask.sum()}/{len(mask)}')

            # add this condition's matching rows to the list of matched rows
            matched = (matched | mask)
        return output

    def __str__(self):
        if self.property_name and self.property_name.startswith('__'):
            name = f'{self.property_name.strip("__")} (external)'
        else:
            name = self.property_name
        if self.callback:
            return f"{name} -> callback({self.callback})"
        out = []
        previous_condition = None
        for condition, value in self.conditions:
            if condition is None:
                out.append(f'{" " * len(previous_condition)} -> {value} (otherwise)')
            else:
                out.append(f"{condition} -> {value}")
                previous_condition = condition
        return "\n  ".join(out)


class LinearModelType(Enum):
    """
    The type of model specifies how the results from the predictor are combined:
    'additive' -> adds the effect_sizes from the predictors
    'logisitc' -> multiples the effect_sizes from the predictors and applies the transform x/(1+x)
    [Thus, the intercept can be taken to be an Odds and effect_sizes Odds Ratios,
    and the prediction is a probability.]
    'multiplicative' -> multiplies the effect_sizes from the predictors
    """
    ADDITIVE = auto()
    LOGISTIC = auto()
    MULTIPLICATIVE = auto()


class LinearModel(object):
    def __init__(self, lm_type: LinearModelType, intercept: float, *args: Predictor):
        """
        A linear model has an intercept and zero or more Predictor variables.
        """
        assert lm_type in LinearModelType, 'Model should be one of the prescribed LinearModelTypes'
        self.lm_type = lm_type

        assert isinstance(intercept, (float, int)), "Intercept is not specified or wrong type."
        self.intercept = intercept

        self.predictors = list()
        for predictor in args:
            assert isinstance(predictor, Predictor)
            self.predictors.append(predictor)

    def predict(self, df: pd.DataFrame, rng: np.random.RandomState = None, **kwargs) -> pd.Series:
        """Will call each Predictor's `predict` methods passing the supplied dataframe"""

        # addition external variables used in the model but not part of the population dataframe
        if kwargs:
            new_columns = {}
            for column_name, value in kwargs.items():
                new_columns[f'__{column_name}__'] = kwargs[column_name]
            df = df.assign(**new_columns)

        indicator_property_names_not_in_df = [p.property_name for p in self.predictors
                                              if (p.property_name is not None)
                                              and (p.property_name not in df.columns)]

        assert not indicator_property_names_not_in_df,\
            f"Predictor variables not in df: {indicator_property_names_not_in_df}"

        assert all([p.property_name in df.columns
                    for p in self.predictors
                    if p.property_name is not None]), "Predictor variables not in dataframe"

        # Store the result of the calculated values of Predictors
        res_by_predictor = pd.DataFrame(index=df.index)
        res_by_predictor[f'__intercept{id(self)}__'] = self.intercept

        for predictor in self.predictors:
            res_by_predictor[predictor] = predictor.predict(df)

        # Do appropriate transformation on output
        if self.lm_type is LinearModelType.ADDITIVE:
            # print("Linear Model: Prediction will be sum of each effect size.")
            res_by_predictor = res_by_predictor.sum(axis=1, skipna=True)

        elif self.lm_type is LinearModelType.LOGISTIC:
            # print("Logistic Regression Model: Prediction will be transform to probabilities. " \
            #       "Intercept assumed to be Odds and effect sizes assumed to be Odds Ratios.")
            odds = res_by_predictor.prod(axis=1, skipna=True)
            res_by_predictor = odds / (1 + odds)

        elif self.lm_type is LinearModelType.MULTIPLICATIVE:
            # print("Multiplicative Model: Prediction will be multiplication of each effect size.")
            res_by_predictor = res_by_predictor.prod(axis=1, skipna=True)

        else:
            raise ValueError(f'Unhandled linear model type: {self.lm_type}')

        # if the user supplied a random number generator then they want outcomes, not probabilities
        if rng:
            outcome = rng.random_sample(len(res_by_predictor)) < res_by_predictor
            # pop the boolean out of the series if we have a single row, otherwise return the series
            if len(outcome) == 1:
                return outcome.iloc[0]
            else:
                return outcome

        # return the raw result from the model
        return res_by_predictor

    @staticmethod
    def multiplicative(*predictors: Predictor):
        """Returns a multplicative LinearModel with intercept=1.0

        :param predictors: One or more Predictor objects defining the model
        """
        return LinearModel(LinearModelType.MULTIPLICATIVE, 1.0, *predictors)

    def __str__(self):
        out = "LinearModel(\n"\
              f"  {self.lm_type},\n"\
              f"  intercept = {self.intercept},\n"
        for predictor in self.predictors:
            out += f'  {predictor}\n'
        out += ")"
        return out