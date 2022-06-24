import argparse
import textwrap

import configargparse

# from falcon import __version__

class NewlineTextHelpFormatter(argparse.HelpFormatter):

    def _fill_text(self, text, width, indent):
        return '\n'.join(
            textwrap.fill(line, width, initial_indent=indent,
                          subsequent_indent=indent,
                          replace_whitespace=False).strip()
            for line in text.splitlines(keepends=True))


class Config:
    """
    Commandline and file-based configuration.

    Configuration settings can be specified in a config.ini file (by default in
    the working directory), or as command-line arguments.
    """

    def __init__(self) -> None:
        """
        Initialize the configuration settings and provide sensible default
        values if possible.
        """

        self._parser = configargparse.ArgParser(
            description=f'HD-MS-Cluster: Accelerated spectrum clustering library using HD'
                        f'==============================================='
                        f'==================\n\n'
                        f'Official code website: '
                        f'https://github.com/bittremieux/falcon\n\n',
            default_config_files=['config.ini'],
            args_for_setting_config_path=['-c', '--config'],
            formatter_class=NewlineTextHelpFormatter)

        # IO
        self._parser.add_argument(
            'input_filepath', type=str,
            help='Input peak files (supported format: .MGF).')
        self._parser.add_argument(
            'output_filename', type=str, 
            help='Output file name.')
        self._parser.add_argument(
            '--file_type', default='mgf',
            choices=['mgf'],
            help='Spectra file type (default: %(default)s).')
        self._parser.add_argument(
            '--work_dir', default=None,
            help='Working directory (default: temporary directory).')
        self._parser.add_argument(
            '--overwrite', action='store_true',
            help="Overwrite existing results (default: don't overwrite).")
        self._parser.add_argument(
            '--export_representatives', action='store_true',
            help='Export cluster representatives to an MGF file '
                 '(default: no export).')

        # SYSTEM
        self._parser.add_argument(
            '--cpu_core', default=6, type=int,
            help='Enabled CPU cores for parallel processing'
                 '(default: %(default)s).')
        self._parser.add_argument(
            '--cpu_core_cluster', default=6, type=int,
            help='Enabled CPU cores for CPU clustering step'
                 '(default: %(default)s).')
        self._parser.add_argument(
            '--cpu_core_dbscan', default=6, type=int,
            help='Enabled CPU cores for DBSCAN'
                 '(default: %(default)s).')
        self._parser.add_argument(
            '--batch_size', default=32, type=int,
            help='Number of spectra to process simultaneously '
                 '(default: %(default)s).')
        self._parser.add_argument(
            '--mode', default='stage', type=str,
            choices=['stage', 'pipeline'],
            help='HD encoding and clustering computation mode'
            'stage: encoding and clustering are separated'
            'pipeline: encoding and clustering are merged (default: %(default)s).')
        self._parser.add_argument(
            '--use_gpu_cluster', action='store_true',
            help='If use GPU\' DBSCAN clustering (default: %(default)s).')
        

        # PREPROCESSING
        self._parser.add_argument(
            '--min_peaks', default=5, type=int,
            help='Discard spectra with fewer than this number of peaks '
                 '(default: %(default)s).')
        self._parser.add_argument(
            '--min_mz_range', default=250., type=float,
            help='Discard spectra with a smaller mass range '
                 '(default: %(default)s m/z).')
        self._parser.add_argument(
            '--min_mz', default=101., type=float,
            help='Minimum peak m/z value (inclusive, '
                 'default: %(default)s m/z).')
        self._parser.add_argument(
            '--max_mz', default=1500., type=float,
            help='Maximum peak m/z value (inclusive, '
                 'default: %(default)s m/z).')
        self._parser.add_argument(
            '--remove_precursor_tol', default=1.5, type=float,
            help='Window around the precursor mass to remove peaks '
                 '(default: %(default)s m/z).')
        self._parser.add_argument(
            '--min_intensity', default=0.01, type=float,
            help='Remove peaks with a lower intensity relative to the base '
                 'intensity (default: %(default)s).')
        self._parser.add_argument(
            '--max_peaks_used', default=50, type=int,
            help='Only use the specified most intense peaks in the spectra '
                 '(default: %(default)s).')
        self._parser.add_argument(
            '--scaling', default='off', type=str,
            choices=['off', 'root', 'log', 'rank'],
            help='Peak scaling method used to reduce the influence of very '
                 'intense peaks (default: %(default)s).')

        # CLUSTERING
        self._parser.add_argument(
            '--hd_dim', default=2048, type=int,
            help='HD dimension D (default: %(default)s).')
        self._parser.add_argument(
            '--hd_Q', default=32, type=int,
            help='HD\'s quantization level Q (default: %(default)s).')

        self._parser.add_argument(
            '--cluster_charges', nargs="+", type=int, default=[2],
            help='Charges to cluster (%(default)s)')
        self._parser.add_argument(
            '--precursor_tol', nargs=2, default=[20, 'ppm'],
            help='Precursor tolerance mass and mode (default: 20 ppm). '
                 'Mode should be either "ppm" or "Da".')
        self._parser.add_argument(
            '--rt_tol', type=float, default=None,
            help='Retention time tolerance (default: no retention time '
                 'filtering).')
        self._parser.add_argument(
            '--fragment_tol', type=float, default=0.05,
            help='Fragment mass tolerance in m/z (default: %(default)s m/z).')

        self._parser.add_argument(
            '--eps', type=float, default=0.6,
            help='The eps parameter (Hamming distance) for DBSCAN clustering '
                 '(default: %(default)s). Relevant Hamming distance thresholds '
                 'are typically around 0.6.')

        # self._parser.add_argument(
        #     '--mz_interval', type=int, default=1,
        #     help='Precursor m/z interval (centered around x.5 Da) to process '
        #          'spectra simultaneously (default: %(default)s m/z).')

        # Filled in 'parse', contains the specified settings.
        self._namespace = None


    def parse(self, args_str: str = None) -> None:
        """
        Parse the configuration settings.

        Parameters
        ----------
        args_str : str
            If None, the arguments are taken from sys.argv. Arguments that are
            not explicitly specified are taken from the configuration file.
        """
        self._namespace = vars(self._parser.parse_args(args_str))

        self._namespace['precursor_tol'][0] = \
            float(self._namespace['precursor_tol'][0])

    def __getattr__(self, option):
        if self._namespace is None:
            raise RuntimeError('The configuration has not been initialized')
        return self._namespace[option]

    def __getitem__(self, item):
        return self.__getattr__(item)


config = Config()
