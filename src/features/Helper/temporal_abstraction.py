import numpy as np
import scipy.stats as st
import pandas as pd


class NumericalAbstraction:
    def aggregate_value(self, aggregation_function: str):
        """
        Return the numpy aggregation function based on the specified aggregation function name.

        Args:
            aggregation_fn: The name of the aggregation function.

        Returns:
            The corresponding numpy aggregation function.

        Raises:
            None."""

        if aggregation_function == "mean":
            return np.mean
        elif aggregation_function == "max":
            return np.max
        elif aggregation_function == "min":
            return np.min
        elif aggregation_function == "median":
            return np.median
        elif aggregation_function == "std":
            return np.std
        else:
            return np.nan

    def abstract_numerical(
        self,
        data_table: pd.DataFrame,
        cols: list,
        window_size: int,
        aggregation_function: str,
    ):
        """
        Abstract numerical features in a data table by applying a rolling window and aggregating values using the specified aggregation function.

        Args:
            data_table: The data table to abstract numerical features from.
            cols: The columns in the data table to abstract.
            window_size: The size of the rolling window.
            aggregation_function: The name of the aggregation function to use.

        Returns:
            The data table with the abstracted numerical features added.

        Raises:
            None."""

        # Create new columns for the temporal data, pass over the dataset and compute values
        for col in cols:
            data_table[
                col + "_temp_" + aggregation_function + "_ws_" + str(window_size)
            ] = (
                data_table[col]
                .rolling(window_size)
                .apply(self.aggregate_value(aggregation_function))
            )

        return data_table
