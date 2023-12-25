from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, filtfilt
import copy
import pandas as pd

# Creating classes for Low-Pass-Butter-filter


class LowPass:
    """
    A class for applying a low-pass filter to a column of a data table.

    Methods:
        low_pass_filter(data_table, col, sampling_frequency, cutoff, order=5, do_phase=True):
            Apply a low-pass filter to a column of a data table.

    Args:
        data_table: The data table to apply the filter to.
        col: The name of the column to filter.
        sampling_frequency: The sampling frequency of the data.
        cutoff: The cutoff frequency of the filter.
        order (optional): The order of the filter. Defaults to 5.
        do_phase (optional): Whether to apply phase correction. Defaults to True.

    Returns:
        The data table with the filtered column added.

    Examples:
        >>> data_table = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        >>> low_pass_filter(data_table, 'col1', 100, 10)
           col1  col1_lowpass
        0     1      0.000000
        1     2      0.000000
        2     3      0.000000
        3     4      0.000000
        4     5      0.000000
        5     6      0.000000
        6     7      0.000000
        7     8      0.000000
        8     9      0.000000
        9    10      0.000000"""

    def low_pass_filter(
        self, data_table, col, sampling_frequency, cutoff_frequency, order=5, phase_shift=True
    ):
        """
        Apply a low-pass filter to a column of a data table.

        Args:
            data_table: The data table to apply the filter to.
            col: The name of the column to filter.
            sampling_frequency: The sampling frequency of the data.
            cutoff: The cutoff frequency of the filter.
            order (optional): The order of the filter. Defaults to 5.
            phase_shift (optional): Whether to apply phase correction. Defaults to True.

        Returns:
            The data table with the filtered column added.

        Examples:
            >>> data_table = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
            >>> low_pass_filter(data_table, 'col1', 100, 10)
               col1  col1_lowpass
            0     1      0.000000
            1     2      0.000000
            2     3      0.000000
            3     4      0.000000
            4     5      0.000000
            5     6      0.000000
            6     7      0.000000
            7     8      0.000000
            8     9      0.000000
            9    10      0.000000"""

        #!The value of Nyquist Fre is supposed to be half the sampling fre
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq

        b, a = butter(order, cut, btype="low", output="ba", analog=False)
        if phase_shift:
            data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
        else:
            data_table[col + "_lowpass"] = lfilter(b, a, data_table[col])
        print(data_table[col+"_lowpass"])
        return data_table


class PrincipalComponentAnalysis:
    pca = []

    def __init__(self):
        self.pca = []

    def normalize_dataset(self, data_table, columns):
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max()
                - data_table[col].min()
                # data_table[col].std()
            )
        return dt_norm

    # Perform the PCA on the selected columns and return the explained variance.
    def determine_pc_explained_variance(self, data_table, cols):
        """
        Determine the explained variance ratios of the principal components obtained from performing PCA on a dataset.

        Args:
            data_table: The dataset to perform PCA on.
            cols: The columns of the dataset to include in the PCA.

        Returns:
            A numpy array containing the explained variance ratios of the principal components.

        Raises:
            None.

        Example:
            >>> data_table = pd.DataFrame({'col1': [1, 2, 3, 4, 5], 'col2': [6, 7, 8, 9, 10]})
            >>> determine_pc_explained_variance(data_table, ['col1', 'col2'])
            array([0.5, 0.5])"""

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_

    # Apply a PCA given the number of components we have selected.
    # We add new pca columns.
    def apply_pca(self, data_table, cols, number_comp):
        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.pca.transform(dt_norm[cols])

        # And add the new ones:
        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]

        return data_table
