import argparse
import ast


class Options(object):
    def __init__(self):
        # Handle command line arguments
        self.parser = argparse.ArgumentParser(
            description="Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments."
        )

        ## Run from config file
        self.parser.add_argument(
            "--config",
            dest="config_filepath",
            help="Configuration .json file (optional). Overwrites existing command-line args!",
        )

        self.parser.add_argument(
            "--output_dir",
            default="./output",
            help="Root output directory. Must exist. Time-stamped directories will be created inside.",
        )
        self.parser.add_argument("--data_dir", default="./data", help="Data directory")

        self.parser.add_argument(
            "--name",
            dest="experiment_name",
            default="",
            help="A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp",
        )
        self.parser.add_argument(
            "--patch_length", dest="patch_length", default=20, type=int
        )
        self.parser.add_argument("--num_layers", dest="num_layers", default=4, type=int)
        self.parser.add_argument("--split_id", dest="split_id", default="1", type=str)
        self.parser.add_argument(
            "--num_classes", dest="num_classes", default=2, type=int
        )
        self.parser.add_argument(
            "--comment",
            type=str,
            default="",
            help="A comment/description of the experiment",
        )
        self.parser.add_argument(
            "--no_timestamp",
            action="store_true",
            help="If set, a timestamp will not be appended to the output directory name",
        )
        self.parser.add_argument(
            "--records_file",
            default="./records.xls",
            help="Excel file keeping all records of experiments",
        )

        self.parser.add_argument(
            "--gpu", type=str, default="0", help="GPU index, -1 for CPU"
        )
        self.parser.add_argument(
            "--n_proc",
            type=int,
            default=-1,
            help="Number of processes for data loading/preprocessing. By default, equals num. of available cores.",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="dataloader threads. 0 for single-thread.",
        )
        self.parser.add_argument(
            "--seed",
            help="Seed used for splitting sets. None by default, set to an integer for reproducibility",
        )
        # Dataset
        self.parser.add_argument(
            "--data_class",
            type=str,
            default="ar",
            help="Which type of data should be processed.",
        )
        self.parser.add_argument(
            "--epochs", type=int, default=10, help="Number of training epochs"
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="learning rate (default holds for batch size 64)",
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=32, help="Training batch size"
        )
        self.parser.add_argument(
            "--d_model",
            type=int,
            default=32,
            help="Internal dimension of transformer embeddings",
        )

        self.parser.add_argument(
            "--dropout",
            type=float,
            default=0.2,
            help="Dropout applied to most transformer encoder layers",
        )

        self.parser.add_argument(
            "--normalization",
            choices={
                "standardization",
                "minmax",
                "none",
            },
            default="minmax",
            help="If specified, will apply normalization on the input features of a dataset.",
        )

    def parse(self):
        args = self.parser.parse_args()
        return args
