import itertools
import math


class Bins:

    def __init__(self, series, num_bins, bin_filters={}):

        self._bins = []
        self._bin_name_index_map = {}

        max_bins = float("-inf")

        # Compute the max # of bins possible.
        sorted_series = {}
        for x in series.get_x():

            y = set(series.get_y(x))
            y = sorted(y)

            # Store the sorted data.
            sorted_series[x] = y

            max_bins = max(max_bins, len(y))

        # Apply the filters.
        # Currently, only a max filter is supported.
        for x in sorted_series.keys():

            if x in bin_filters:

                y = sorted_series[x]

                new_y = list(itertools.filterfalse(
                    lambda y: y > bin_filters[x],
                    y))

                sorted_series[x] = new_y

        # Fit the total bins to the user specified bin sizes.
        num_bins = min(max_bins, num_bins)
        assert num_bins > 0 and math.isfinite(num_bins)
        for bin_no in range(num_bins):

            _bin = {}
            for x in sorted_series:

                y = sorted_series[x]
                bin_size = math.floor(len(sorted_series[x]) / num_bins)

                # Dimensions which cannot be fit into num_bins get collapsed
                # into a single bin.
                if bin_size == 0:

                    start_idx = 0
                    end_idx = len(y) - 1
                elif bin_no == num_bins - 1:

                    start_idx = bin_size * bin_no
                    end_idx = len(y) - 1
                else:

                    start_idx = bin_size * bin_no
                    end_idx = min(start_idx + bin_size - 1, len(y) - 1)

                _bin[x] = (y[start_idx], y[end_idx])

            self._bins.append(_bin)

            bin_name = self.get_bin_name(len(self._bins) - 1)
            self._bin_name_index_map[bin_name] = len(self._bins) - 1

        # Add the last infinity bin.
        _bin = {}
        last_bin = self._bins[-1]
        for x in sorted_series:

            _bin[x] = (last_bin[x][1] + 1, float("inf"))

        self._bins.append(_bin)

    def get_bin_index(self, document):

        for i in range(len(self._bins)):

            passed = True
            _bin = self._bins[i]
            for x in _bin:

                min_value, max_value = _bin[x]
                if document["problem"][x] < min_value \
                        or document["problem"][x] > max_value:

                    passed = False
                    break

            if passed:

                return i

        return float("inf")

    def __len__(self):

        return len(self._bins)

    def get_bin_name(self, bin_no):

        try:
            _bin = self._bins[bin_no]

            string = ""
            for x in _bin:

                string += "(%s:[%s:%s])" % (x, _bin[x][0], _bin[x][1])

            return string
        except IndexError:

            return "infinity"

    def get_index_from_name(self, bin_name):

        return self._bin_name_index_map.get(bin_name, float("inf"))

    def get_name(self):

        if len(self._bins) == 0:

            return "()"
        else:

            _bin = self._bins[0]

            string = "("
            for x in _bin:

                string += "%s " % (x)

            string = string.strip()
            string += ")"
            return string
