"""
Core drift monitoring logic orchestrating Evidently reports and Prometheus metrics.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Optional

import fsspec
import pandas as pd
import requests
from prometheus_client import CollectorRegistry, Gauge

from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

from src.drift_monitoring.config import DriftSettings
from src.utils.drift import load_dataset_from_uri

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    data_drift_detected: bool
    data_drift_share: float
    prediction_drift_detected: bool
    prediction_drift_score: Optional[float]
    prediction_psi: Optional[float]
    feature_metrics: List[tuple[str, bool, Optional[float]]]
    current_rows: int
    evaluated_at: datetime


class DriftMonitor:
    """
    Encapsulates reference data loading, current data retrieval, Evidently execution,
    and Prometheus metric updates.
    """

    def __init__(self, settings: DriftSettings, registry: CollectorRegistry):
        self.settings = settings
        self.registry = registry
        self.reference_df: pd.DataFrame | None = None
        self.column_mapping: ColumnMapping | None = None
        self.feature_columns: list[str] = []
        self.prediction_column: str | None = None
        self.target_column: str | None = None
        self._shutdown_event = asyncio.Event()
        self._task: asyncio.Task | None = None

        # Prometheus metrics
        self.data_drift_status = Gauge(
            "evidently_data_drift_status",
            "Binary flag equal to 1 when Evidently detects dataset drift.",
            registry=registry,
        )
        self.data_drift_share = Gauge(
            "evidently_data_drift_share",
            "Share of columns detected as drifting by Evidently.",
            registry=registry,
        )
        self.prediction_drift_status = Gauge(
            "evidently_prediction_drift_status",
            "Binary flag equal to 1 when Evidently detects prediction drift.",
            registry=registry,
        )
        self.prediction_drift_score = Gauge(
            "evidently_prediction_drift_score",
            "Drift score reported for the prediction column (statistical test specific).",
            registry=registry,
        )
        self.prediction_psi_metric = Gauge(
            "evidently_prediction_psi_score",
            "Population Stability Index computed for the model predictions.",
            registry=registry,
        )
        self.feature_drift_status = Gauge(
            "evidently_feature_drift_status",
            "Per-feature drift flag (1 when drifting).",
            ["feature"],
            registry=registry,
        )
        self.feature_drift_score = Gauge(
            "evidently_feature_drift_score",
            "Per-feature drift score produced by Evidently.",
            ["feature"],
            registry=registry,
        )
        self.current_rows_metric = Gauge(
            "drift_monitor_current_row_count",
            "Number of rows in the current production sample used for drift calculation.",
            registry=registry,
        )
        self.reference_rows_metric = Gauge(
            "drift_monitor_reference_row_count",
            "Number of rows in the reference dataset.",
            registry=registry,
        )
        self.last_run_ts = Gauge(
            "drift_monitor_last_run_timestamp",
            "Unix timestamp (UTC seconds) of the last successful drift evaluation.",
            registry=registry,
        )

    async def initialize(self) -> None:
        """
        Load the reference dataset and compute derived configuration.
        """
        logger.info("Loading reference dataset from %s", self.settings.reference_dataset_uri)
        df = load_dataset_from_uri(self.settings.reference_dataset_uri)
        if self.settings.max_rows is not None:
            df = df.head(self.settings.max_rows)
        self.reference_df = self._prepare_dataframe(df)
        if self.reference_df.empty:
            raise ValueError("Reference dataset is empty; cannot run drift monitoring.")

        self.reference_rows_metric.set(len(self.reference_df))

        self.feature_columns = [col for col in self.reference_df.columns if col.startswith("feature_")]
        if not self.feature_columns:
            raise ValueError(
                "Reference dataset must contain feature columns named feature_0, feature_1, ... for drift analysis."
            )
        self.prediction_column = "prediction" if "prediction" in self.reference_df.columns else None
        self.target_column = "target" if "target" in self.reference_df.columns else None

        mapping = ColumnMapping()
        mapping.features = list(self.feature_columns)
        if self.prediction_column:
            mapping.prediction = self.prediction_column
        if self.target_column:
            mapping.target = self.target_column
        self.column_mapping = mapping
        logger.info(
            "Reference dataset ready with %d rows, %d features. Prediction column: %s",
            len(self.reference_df),
            len(self.feature_columns),
            self.prediction_column,
        )

    async def run(self) -> None:
        """
        Periodically evaluate drift until shutdown is requested.
        """
        self._shutdown_event.clear()
        while not self._shutdown_event.is_set():
            try:
                await self.evaluate_once()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Drift evaluation failed: %s", exc)
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.settings.evaluation_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue

    async def evaluate_once(self) -> None:
        """
        Execute a single Evidently run and update metrics.
        """
        if self.reference_df is None or self.column_mapping is None:
            raise RuntimeError("DriftMonitor must be initialized before evaluating.")

        current_df = await asyncio.to_thread(self._load_current_dataframe)
        if current_df is None or current_df.empty or len(current_df) < self.settings.min_rows:
            logger.info(
                "Not enough current data to evaluate drift (rows=%s, min_rows=%s).",
                0 if current_df is None else len(current_df),
                self.settings.min_rows,
            )
            self.current_rows_metric.set(0 if current_df is None else len(current_df))
            return

        if self.settings.max_rows is not None:
            current_df = current_df.head(self.settings.max_rows)

        current_df = self._ensure_columns(current_df)

        report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        report.run(
            reference_data=self.reference_df,
            current_data=current_df,
            column_mapping=self.column_mapping,
        )

        prediction_psi = self._compute_prediction_psi(current_df)
        metrics = self._extract_metrics(report.as_dict(), len(current_df), prediction_psi)
        self._update_prometheus(metrics)

    async def shutdown(self) -> None:
        """
        Stop background evaluation task.
        """
        self._shutdown_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def start_background_loop(self) -> None:
        """
        Spawn the evaluation loop as an asyncio Task. Safe to call after initialize().
        """
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [col.strip() for col in df.columns]
        return df

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = pd.NA
        if self.prediction_column and self.prediction_column not in df.columns:
            df[self.prediction_column] = pd.NA
        if self.target_column and self.target_column not in df.columns:
            df[self.target_column] = pd.NA

        ordered_cols = list(self.feature_columns)
        if self.prediction_column:
            ordered_cols.append(self.prediction_column)
        if self.target_column and self.target_column not in ordered_cols:
            ordered_cols.append(self.target_column)
        return df[ordered_cols]

    def _compute_prediction_psi(self, current_df: pd.DataFrame) -> Optional[float]:
        if self.prediction_column is None or self.reference_df is None:
            return None

        ref_series = self.reference_df[self.prediction_column].dropna()
        cur_series = current_df[self.prediction_column].dropna()
        if ref_series.empty or cur_series.empty:
            return None

        ref_dist = ref_series.value_counts(normalize=True)
        cur_dist = cur_series.value_counts(normalize=True)
        categories = set(ref_dist.index) | set(cur_dist.index)
        psi = 0.0
        eps = 1e-6
        for category in categories:
            ref_pct = max(float(ref_dist.get(category, eps)), eps)
            cur_pct = max(float(cur_dist.get(category, eps)), eps)
            psi += (ref_pct - cur_pct) * math.log(ref_pct / cur_pct)
        return psi

    def _load_current_dataframe(self) -> Optional[pd.DataFrame]:
        if self.settings.current_dataset_uri:
            return self._load_from_uri(self.settings.current_dataset_uri)
        if self.settings.loki_base_url:
            events = self._collect_from_loki()
            return self._events_to_dataframe(events)
        return None

    def _load_from_uri(self, uri: str) -> Optional[pd.DataFrame]:
        if uri.endswith(".json") or uri.endswith(".jsonl"):
            events = self._read_json_events(uri)
            return self._events_to_dataframe(events)
        df = load_dataset_from_uri(uri)
        return df

    def _read_json_events(self, uri: str) -> list[dict[str, Any]]:
        resolved_uri = uri
        if uri.startswith("file://"):
            resolved_uri = uri[len("file://") :]
        events: list[dict[str, Any]] = []
        with fsspec.open(resolved_uri, mode="rt", encoding="utf8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON log line: %s", line)
                    continue
                events.append(payload)
        return events

    def _collect_from_loki(self) -> list[dict[str, Any]]:
        if not self.settings.loki_base_url or not self.settings.loki_query:
            return []
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=self.settings.loki_range_minutes)
        params = {
            "query": self.settings.loki_query,
            "start": int(start.timestamp() * 1e9),
            "end": int(end.timestamp() * 1e9),
            "direction": "forward",
            "limit": 5000,
        }
        url = self.settings.loki_base_url.rstrip("/") + "/loki/api/v1/query_range"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
        events: list[dict[str, Any]] = []
        data = payload.get("data", {})
        for stream in data.get("result", []):
            for _ts, line in stream.get("values", []):
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON Loki log line: %s", line)
                    continue
                # Loki wraps log line as string, but our logger emits JSON string already.
                if isinstance(parsed, str):
                    try:
                        parsed = json.loads(parsed)
                    except json.JSONDecodeError:
                        continue
                if isinstance(parsed, dict):
                    events.append(parsed)
        return events

    def _events_to_dataframe(self, events: Iterable[dict[str, Any]]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for event in events:
            features_batch = event.get("features") or []
            predictions = event.get("predictions") or []
            targets = event.get("targets") or event.get("target") or []

            for idx, feature_row in enumerate(features_batch):
                row: dict[str, Any] = {}
                for col_idx, col_name in enumerate(self.feature_columns):
                    try:
                        row[col_name] = feature_row[col_idx]
                    except IndexError:
                        row[col_name] = None
                if self.prediction_column:
                    try:
                        row[self.prediction_column] = predictions[idx]
                    except (IndexError, TypeError):
                        row[self.prediction_column] = None
                if self.target_column:
                    try:
                        row[self.target_column] = targets[idx]
                    except (IndexError, TypeError):
                        row[self.target_column] = None
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=self.feature_columns)
        return pd.DataFrame(rows)

    def _extract_metrics(
        self,
        report_dict: dict[str, Any],
        current_rows: int,
        prediction_psi: Optional[float],
    ) -> DriftMetrics:
        data_drift_detected = False
        data_drift_share = 0.0
        prediction_drift_detected = False
        prediction_drift_score: float | None = None
        feature_metrics: list[tuple[str, bool, Optional[float]]] = []

        for metric in report_dict.get("metrics", []):
            metric_name = metric.get("metric")
            result = metric.get("result", {})
            if metric_name == "DatasetDriftMetric":
                data_drift_detected = bool(result.get("dataset_drift", False))
                data_drift_share = float(result.get("share_of_drifted_columns", 0.0))
            elif metric_name == "DataDriftTable":
                drift_by_columns = result.get("drift_by_columns", {})
                for feature, details in drift_by_columns.items():
                    drift_flag = bool(details.get("drift_detected"))
                    drift_score = details.get("drift_score")
                    feature_metrics.append((feature, drift_flag, drift_score))
            elif metric_name == "ColumnDriftMetric" and result.get("column_name") == self.prediction_column:
                prediction_drift_detected = bool(result.get("drift_detected", False))
                prediction_drift_score = result.get("drift_score")

        return DriftMetrics(
            data_drift_detected=data_drift_detected,
            data_drift_share=data_drift_share,
            prediction_drift_detected=prediction_drift_detected,
            prediction_drift_score=None if prediction_drift_score is None else float(prediction_drift_score),
            prediction_psi=None if prediction_psi is None else float(prediction_psi),
            feature_metrics=feature_metrics,
            current_rows=current_rows,
            evaluated_at=datetime.now(timezone.utc),
        )

    def _update_prometheus(self, metrics: DriftMetrics) -> None:
        self.data_drift_status.set(1 if metrics.data_drift_detected else 0)
        self.data_drift_share.set(metrics.data_drift_share)
        self.prediction_drift_status.set(1 if metrics.prediction_drift_detected else 0)
        if metrics.prediction_drift_score is not None:
            self.prediction_drift_score.set(metrics.prediction_drift_score)
        else:
            self.prediction_drift_score.set(0)
        if metrics.prediction_psi is not None:
            self.prediction_psi_metric.set(metrics.prediction_psi)
        else:
            self.prediction_psi_metric.set(0)
        self.current_rows_metric.set(metrics.current_rows)
        self.last_run_ts.set(metrics.evaluated_at.timestamp())

        # reset feature gauges before updating to contain only latest metrics
        self.feature_drift_status.clear()
        self.feature_drift_score.clear()
        for feature, drift_flag, drift_score in metrics.feature_metrics:
            self.feature_drift_status.labels(feature=feature).set(1 if drift_flag else 0)
            if drift_score is not None:
                self.feature_drift_score.labels(feature=feature).set(float(drift_score))
