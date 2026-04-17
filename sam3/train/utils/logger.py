# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import atexit
import functools
import logging
import sys
import uuid
from typing import Any, Dict, Optional, Union

from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from numpy import ndarray
from sam3.train.utils.train_utils import get_machine_local_and_dist_rank, makedir
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

Scalar = Union[Tensor, ndarray, int, float]


def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    makedir(log_dir)
    summary_writer_method = SummaryWriter
    return TensorBoardLogger(
        path=log_dir, summary_writer_method=summary_writer_method, **writer_kwargs
    )


def make_wandb_logger(project: str, name: Optional[str] = None, **kwargs: Any):
    return WandbLogger(project=project, name=name, **kwargs)


def make_mlflow_logger(
    experiment_name: str,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    **kwargs: Any,
):
    return MlflowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        **kwargs,
    )


class TensorBoardWriterWrapper:
    """
    A wrapper around a SummaryWriter object.
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: str = None,
        summary_writer_method: Any = SummaryWriter,
        **kwargs: Any,
    ) -> None:
        """Create a new TensorBoard logger.
        On construction, the logger creates a new events file that logs
        will be written to.  If the environment variable `RANK` is defined,
        logger will only log if RANK = 0.

        NOTE: If using the logger with distributed training:
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing distributed process group.

        Args:
            path (str): path to write logs to
            *args, **kwargs: Extra arguments to pass to SummaryWriter
        """
        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path
        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        else:
            logging.debug(
                f"Not logging meters on this host because env RANK: {self._rank} != 0"
            )
        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        """Writes pending logs to disk."""

        if not self._writer:
            return

        self._writer.flush()

    def close(self) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        if not self._writer:
            return

        self._writer.close()
        self._writer = None


class TensorBoardLogger(TensorBoardWriterWrapper):
    """
    A simple logger for TensorBoard.
    """

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int, Optional): step value to record
        """
        if not self._writer:
            return
        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int, optional): step value to record
        """
        if not self._writer:
            return
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            meters (dict): dictionary of name of meter and corersponding values
        """
        if not self._writer:
            return
        self._writer.add_hparams(hparams, meters)


class WandbLogger:
    """
    A logger for Weights & Biases. Only initializes on rank 0.
    """

    def __init__(
        self, project: str, name: Optional[str] = None, config: Optional[Dict] = None, **kwargs: Any
    ) -> None:
        self._run = None
        _, self._rank = get_machine_local_and_dist_rank()
        if self._rank == 0:
            import wandb

            self._run = wandb.init(
                project=project, name=name, config=config, **kwargs
            )
            # SAM 3 logs at two different step axes: per-batch (via `log()`) and
            # per-epoch (via `log_dict()`). W&B's default global monotonic step
            # drops epoch-level metrics because their step is smaller than the
            # latest batch step. Decouple them via `define_metric`.
            wandb.define_metric("train/step")
            wandb.define_metric("train/epoch")
            wandb.define_metric("Step_Stats/*", step_metric="train/step")
            wandb.define_metric("*", step_metric="train/epoch")
            logging.info(
                f"W&B run initialized: {self._run.url}"
            )
        atexit.register(self.close)

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if not self._run:
            return
        # log_dict is called per-epoch in SAM 3's trainer
        self._run.log({**payload, "train/epoch": step})

    def log(self, name: str, data: Scalar, step: int) -> None:
        if not self._run:
            return
        # log is called per-batch in SAM 3's trainer
        self._run.log({name: data, "train/step": step})

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if not self._run:
            return
        self._run.config.update(hparams, allow_val_change=True)
        for k, v in meters.items():
            self._run.summary[k] = v

    def close(self) -> None:
        if not self._run:
            return
        import wandb

        wandb.finish()
        self._run = None


class MlflowLogger:
    """
    A logger for MLflow. Only initializes on rank 0.
    MLflow does not support "/" in metric names, so they are replaced with ".".
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._run = None
        _, self._rank = get_machine_local_and_dist_rank()
        if self._rank == 0:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._run = mlflow.start_run(run_name=run_name, **kwargs)
            logging.info(
                f"MLflow run initialized: {self._run.info.run_id}"
            )
        atexit.register(self.close)

    @staticmethod
    def _sanitize_key(name: str) -> str:
        return name.replace("/", ".")

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if not self._run:
            return
        import mlflow

        mlflow.log_metrics(
            {self._sanitize_key(k): float(v) for k, v in payload.items()},
            step=step,
        )

    def log(self, name: str, data: Scalar, step: int) -> None:
        if not self._run:
            return
        import mlflow

        mlflow.log_metric(self._sanitize_key(name), float(data), step=step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if not self._run:
            return
        import mlflow

        mlflow.log_params(
            {self._sanitize_key(k): v for k, v in hparams.items()}
        )
        mlflow.log_metrics(
            {self._sanitize_key(k): float(v) for k, v in meters.items()}
        )

    def close(self) -> None:
        if not self._run:
            return
        import mlflow

        mlflow.end_run()
        self._run = None


class Logger:
    """
    A logger class that fans out to multiple backends: TensorBoard, W&B, MLflow.
    Each backend is optional and controlled by the corresponding field in logging_conf.
    """

    def __init__(self, logging_conf):
        # TensorBoard (always present in config)
        tb_config = logging_conf.tensorboard_writer
        tb_should_log = tb_config and tb_config.pop("should_log", True)
        self.tb_logger = instantiate(tb_config) if tb_should_log else None

        # W&B (may not exist on older configs)
        wandb_config = getattr(logging_conf, "wandb_writer", None)
        wandb_should_log = wandb_config and wandb_config.pop("should_log", True)
        self.wandb_logger = instantiate(wandb_config) if wandb_should_log else None

        # MLflow (may not exist on older configs)
        mlflow_config = getattr(logging_conf, "mlflow_writer", None)
        mlflow_should_log = mlflow_config and mlflow_config.pop("should_log", True)
        self.mlflow_logger = instantiate(mlflow_config) if mlflow_should_log else None

    def _for_each(self, method: str, *args: Any, **kwargs: Any) -> None:
        for logger in (self.tb_logger, self.wandb_logger, self.mlflow_logger):
            if logger is not None:
                getattr(logger, method)(*args, **kwargs)

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        self._for_each("log_dict", payload, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        self._for_each("log", name, data, step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        self._for_each("log_hparams", hparams, meters)


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # we tune the buffering value so that the logs are updated
    # frequently.
    log_buffer_kb = 10 * 1024  # 10KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
):
    """
    Setup various logging streams: stdout and file handlers.
    For file handlers, we only setup for the master gpu.
    """
    # get the filename if we want to log to the file as well
    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # Cleanup any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)

    # we log to file as well if user wants
    if log_filename and rank == 0:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


def shutdown_logging():
    """
    After training is done, we ensure to shut down all the logger streams.
    """
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()
