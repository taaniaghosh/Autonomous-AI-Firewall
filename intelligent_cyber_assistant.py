from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class DetectionOutput:
    sample_index: int
    probability_attack: float
    anomaly_score: float
    confidence_score: float
    prediction: int
    attack_type: str
    source_ip: str
    proto: str
    service: str
    state: str
    source_profile: str
    destination_profile: str


@dataclass
class CorrelatedEvent:
    event_id: str
    start_idx: int
    end_idx: int
    size: int
    avg_attack_probability: float
    avg_anomaly_score: float
    confidence: float
    severity: str
    event_type: str
    source_ip: str
    top_proto: str
    top_service: str
    top_state: str
    incident_statement: str


class HybridDetector:
    """Hybrid detector (CNN-proxy + LSTM-proxy + RF + Autoencoder)."""

    def __init__(self, random_state: int = 42, fast_mode: bool = False) -> None:
        self.scaler = StandardScaler()

        if fast_mode:
            lstm_hidden = (32, 16)
            cnn_hidden = (64, 32)
            ae_hidden = (48, 24, 48)
            clf_max_iter = 120
            ae_max_iter = 150
            rf_estimators = 120
            rf_depth = 12
        else:
            lstm_hidden = (64, 32)
            cnn_hidden = (128, 64)
            ae_hidden = (96, 48, 96)
            clf_max_iter = 220
            ae_max_iter = 260
            rf_estimators = 200
            rf_depth = 18

        self.lstm_proxy = MLPClassifier(
            hidden_layer_sizes=lstm_hidden,
            activation="tanh",
            max_iter=clf_max_iter,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=random_state,
        )
        self.cnn_proxy = MLPClassifier(
            hidden_layer_sizes=cnn_hidden,
            activation="relu",
            max_iter=clf_max_iter,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=random_state + 1,
        )
        self.rf = RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=rf_depth,
            n_jobs=-1,
            random_state=random_state,
        )

        # Autoencoder-like reconstruction model for anomaly scoring.
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=ae_hidden,
            activation="relu",
            max_iter=ae_max_iter,
            random_state=random_state + 2,
            early_stopping=True,
            n_iter_no_change=12,
        )

        self._train_recon_min = 0.0
        self._train_recon_max = 1.0

    @staticmethod
    def _add_temporal_features(x: np.ndarray) -> np.ndarray:
        lag = np.vstack([np.zeros((1, x.shape[1])), x[:-1]])
        delta = x - lag
        return np.hstack([x, lag, delta])

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_train_temporal = self._add_temporal_features(x_train_scaled)

        self.lstm_proxy.fit(x_train_temporal, y_train)
        self.cnn_proxy.fit(x_train_scaled, y_train)
        self.rf.fit(x_train, y_train)

        self.autoencoder.fit(x_train_scaled, x_train_scaled)
        recon = self.autoencoder.predict(x_train_scaled)
        train_err = np.mean((x_train_scaled - recon) ** 2, axis=1)
        self._train_recon_min = float(np.min(train_err))
        self._train_recon_max = float(np.max(train_err))
        if self._train_recon_max <= self._train_recon_min:
            self._train_recon_max = self._train_recon_min + 1e-6

    def predict_components(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_scaled = self.scaler.transform(x)
        x_temporal = self._add_temporal_features(x_scaled)

        p_lstm = self.lstm_proxy.predict_proba(x_temporal)[:, 1]
        p_cnn = self.cnn_proxy.predict_proba(x_scaled)[:, 1]
        p_rf = self.rf.predict_proba(x)[:, 1]

        p_hybrid = np.clip((0.35 * p_lstm) + (0.30 * p_cnn) + (0.35 * p_rf), 0.0, 1.0)

        recon = self.autoencoder.predict(x_scaled)
        recon_err = np.mean((x_scaled - recon) ** 2, axis=1)
        anomaly = (recon_err - self._train_recon_min) / (self._train_recon_max - self._train_recon_min)
        anomaly = np.clip(anomaly, 0.0, 1.0)

        confidence = np.clip((0.72 * p_hybrid) + (0.28 * anomaly), 0.0, 1.0)
        return p_hybrid, anomaly, confidence


class CorrelationEngine:
    """Groups related alerts into incident-level events."""

    def __init__(self, window_size: int = 8, similarity_threshold: float = 0.58) -> None:
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold

    def correlate(self, detections: List[DetectionOutput]) -> List[CorrelatedEvent]:
        attack_only = [d for d in detections if d.prediction == 1]
        if not attack_only:
            return []

        events: List[CorrelatedEvent] = []
        group: List[DetectionOutput] = [attack_only[0]]
        event_counter = 1

        for current in attack_only[1:]:
            prev = group[-1]
            close_in_time = (current.sample_index - prev.sample_index) <= self.window_size
            similar = self._alert_similarity(prev, current) >= self.similarity_threshold

            if close_in_time and similar:
                group.append(current)
            else:
                if len(group) >= 2:
                    events.append(self._build_event(group, event_counter))
                    event_counter += 1
                group = [current]

        if len(group) >= 2:
            events.append(self._build_event(group, event_counter))

        return events

    @staticmethod
    def _alert_similarity(a: DetectionOutput, b: DetectionOutput) -> float:
        score = 0.0
        if a.source_ip == b.source_ip:
            score += 0.32
        if a.proto == b.proto:
            score += 0.20
        if a.service == b.service:
            score += 0.18
        if a.state == b.state:
            score += 0.14
        if abs(a.anomaly_score - b.anomaly_score) <= 0.12:
            score += 0.08
        if abs(a.probability_attack - b.probability_attack) <= 0.15:
            score += 0.08
        return score

    @staticmethod
    def _majority(values: List[str], fallback: str = "unknown") -> str:
        if not values:
            return fallback
        return str(pd.Series(values).mode().iloc[0])

    @staticmethod
    def _infer_event_type(service: str, state: str, attack_type: str, size: int) -> str:
        s = service.lower()
        t = attack_type.lower()
        st = state.lower()
        if "brute" in t or "ftp" in s:
            return "Brute Force Attack"
        if "flood" in t or ("dns" in s and size >= 4):
            return "Flood/DoS-like Activity"
        if "recon" in t or "syn" in st or "int" in st:
            return "Reconnaissance Activity"
        if size >= 6:
            return "Coordinated Attack Campaign"
        return "Suspicious Attack Activity"

    def _build_event(self, group: List[DetectionOutput], event_counter: int) -> CorrelatedEvent:
        probs = [g.probability_attack for g in group]
        anom = [g.anomaly_score for g in group]

        top_source = self._majority([g.source_ip for g in group], fallback="unknown-source")
        top_proto = self._majority([g.proto for g in group])
        top_service = self._majority([g.service for g in group])
        top_state = self._majority([g.state for g in group])
        top_attack_type = self._majority([g.attack_type for g in group], fallback="Unknown")

        avg_p = float(np.mean(probs))
        avg_anomaly = float(np.mean(anom))
        confidence = float(np.clip((0.62 * avg_p) + (0.38 * avg_anomaly), 0.0, 1.0))

        if confidence >= 0.85 or len(group) >= 8:
            severity = "High"
        elif confidence >= 0.70 or len(group) >= 4:
            severity = "Medium"
        else:
            severity = "Low"

        event_type = self._infer_event_type(top_service, top_state, top_attack_type, len(group))
        statement = (
            f"Multiple suspicious {top_service} activities from source {top_source} "
            f"observed between sample {group[0].sample_index} and {group[-1].sample_index}."
        )

        return CorrelatedEvent(
            event_id=f"EVT-{event_counter:04d}",
            start_idx=group[0].sample_index,
            end_idx=group[-1].sample_index,
            size=len(group),
            avg_attack_probability=avg_p,
            avg_anomaly_score=avg_anomaly,
            confidence=confidence,
            severity=severity,
            event_type=event_type,
            source_ip=top_source,
            top_proto=top_proto,
            top_service=top_service,
            top_state=top_state,
            incident_statement=statement,
        )


class ReasoningLayer:
    """LLM-first reasoning layer with template fallback."""

    def __init__(
        self,
        use_llm: bool = True,
        model: str = "gpt-4o-mini",
        provider: str = "auto",
    ) -> None:
        self.use_llm = use_llm
        self.model = model
        self.provider = provider.lower().strip()
        self.api_key = ""
        self._openai_client: Any = None
        self._gemini_model: Any = None

        if not self.use_llm:
            return

        if self.provider == "auto":
            if self.model.lower().startswith("gemini"):
                self.provider = "gemini"
            else:
                self.provider = "openai"

        if self.provider in {"openai", "auto"}:
            self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if self.api_key:
                try:
                    from openai import OpenAI

                    self._openai_client = OpenAI(api_key=self.api_key)
                except Exception:
                    self._openai_client = None

        gemini_model_name = self.model if self.model.lower().startswith("gemini") else os.getenv(
            "GEMINI_MODEL", "gemini-1.5-flash"
        )
        if self.provider in {"gemini", "auto", "openai"}:
            self.api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv(
                "GOOGLE_API_KEY", ""
            ).strip()
            if self.api_key:
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=self.api_key)
                    self._gemini_model = genai.GenerativeModel(model_name=gemini_model_name)
                except Exception:
                    self._gemini_model = None

    @staticmethod
    def recommend_actions(severity: str) -> str:
        if severity == "High":
            return "Block source IP/profile for 10 minutes, enable hard rate limiting, alert administrator"
        if severity == "Medium":
            return "Apply temporary rate limits, challenge suspicious sessions, and notify SOC channel"
        return "Keep source under watch, increase logging, and monitor trend"

    def _template_explanation(self, event: CorrelatedEvent) -> Dict[str, str]:
        reason = (
            f"This appears to be {event.event_type.lower()} due to repeated suspicious "
            f"{event.top_service} behavior in a short window with confidence {event.confidence:.2f}."
        )
        return {
            "event_id": event.event_id,
            "attack_type": event.event_type,
            "severity": event.severity,
            "reason": reason,
            "recommended_action": self.recommend_actions(event.severity),
            "confidence": f"{event.confidence:.3f}",
            "llm_used": "no",
        }

    def _llm_explanation(self, event: CorrelatedEvent) -> Dict[str, str] | None:
        if self.provider == "gemini":
            return self._gemini_explanation(event)
        if self._openai_client is None:
            return self._gemini_explanation(event)

        payload = {
            "event_id": event.event_id,
            "incident_statement": event.incident_statement,
            "event_type": event.event_type,
            "severity": event.severity,
            "confidence": round(event.confidence, 3),
            "avg_attack_probability": round(event.avg_attack_probability, 3),
            "avg_anomaly_score": round(event.avg_anomaly_score, 3),
            "source_ip": event.source_ip,
            "top_service": event.top_service,
            "top_proto": event.top_proto,
        }

        system_msg = (
            "You are a SOC analyst assistant. Return only strict JSON with keys: "
            "attack_type, severity, explanation, recommended_action, confidence."
        )
        user_msg = f"Analyze this incident and generate response guidance: {json.dumps(payload)}"

        try:
            response = self._openai_client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            parsed = json.loads(response.choices[0].message.content)
            confidence = str(parsed.get("confidence", f"{event.confidence:.3f}"))
            return {
                "event_id": event.event_id,
                "attack_type": str(parsed.get("attack_type", event.event_type)),
                "severity": str(parsed.get("severity", event.severity)),
                "reason": str(parsed.get("explanation", "No explanation provided.")),
                "recommended_action": str(
                    parsed.get("recommended_action", self.recommend_actions(event.severity))
                ),
                "confidence": confidence,
                "llm_used": "yes",
            }
        except Exception:
            gemini_result = self._gemini_explanation(event)
            if gemini_result is not None:
                return gemini_result
            return None

    def _gemini_explanation(self, event: CorrelatedEvent) -> Dict[str, str] | None:
        if self._gemini_model is None:
            return None

        payload = {
            "event_id": event.event_id,
            "incident_statement": event.incident_statement,
            "event_type": event.event_type,
            "severity": event.severity,
            "confidence": round(event.confidence, 3),
            "avg_attack_probability": round(event.avg_attack_probability, 3),
            "avg_anomaly_score": round(event.avg_anomaly_score, 3),
            "source_ip": event.source_ip,
            "top_service": event.top_service,
            "top_proto": event.top_proto,
        }

        prompt = (
            "You are a SOC analyst assistant. Return strict JSON only with keys: "
            "attack_type, severity, explanation, recommended_action, confidence. "
            f"Incident: {json.dumps(payload)}"
        )

        try:
            response = self._gemini_model.generate_content(prompt)
            text = getattr(response, "text", "").strip()
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                return None
            parsed = json.loads(text[start : end + 1])

            return {
                "event_id": event.event_id,
                "attack_type": str(parsed.get("attack_type", event.event_type)),
                "severity": str(parsed.get("severity", event.severity)),
                "reason": str(parsed.get("explanation", "No explanation provided.")),
                "recommended_action": str(
                    parsed.get("recommended_action", self.recommend_actions(event.severity))
                ),
                "confidence": str(parsed.get("confidence", f"{event.confidence:.3f}")),
                "llm_used": "yes",
            }
        except Exception:
            return None

    def explain_event(self, event: CorrelatedEvent) -> Dict[str, str]:
        llm_result = self._llm_explanation(event)
        if llm_result is not None:
            return llm_result
        return self._template_explanation(event)


class HoneybadgerLayer:
    """Deception and containment playbook layer."""

    @staticmethod
    def build_playbook(event: CorrelatedEvent) -> Dict[str, str]:
        if event.severity == "High":
            return {
                "honeybadger_mode": "aggressive",
                "honeybadger_action": "Redirect source to honeypot, hard block at edge, and trigger pager alert",
            }
        if event.severity == "Medium":
            return {
                "honeybadger_mode": "guarded",
                "honeybadger_action": "Mirror suspicious traffic to decoy service and apply soft block/rate limits",
            }
        return {
            "honeybadger_mode": "observe",
            "honeybadger_action": "Tag source for watchlist and increase forensic telemetry",
        }


def _safe_col(df: pd.DataFrame, candidates: List[str], fallback: str) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col].astype(str)
    return pd.Series([fallback] * len(df), index=df.index, dtype=str)


def _build_context_profiles(df: pd.DataFrame) -> pd.DataFrame:
    context = pd.DataFrame(index=df.index)
    context["proto"] = _safe_col(df, ["proto"], "unknown")
    context["service"] = _safe_col(df, ["service"], "unknown")
    context["state"] = _safe_col(df, ["state"], "unknown")
    context["attack_hint"] = _safe_col(df, ["attack_cat"], "Unknown")
    source_ip_raw = _safe_col(df, ["srcip", "src_ip", "source_ip"], "")

    # If dataset has no real IP fields, derive a stable source entity id for regrouping.
    if source_ip_raw.str.strip().eq("").all():
        fp_cols = []
        for col in ["ct_srv_src", "ct_src_ltm", "ct_src_dport_ltm", "proto", "service", "state"]:
            if col in df.columns:
                fp_cols.append(df[col].astype(str))

        if fp_cols:
            fingerprint = fp_cols[0]
            for s in fp_cols[1:]:
                fingerprint = fingerprint + "|" + s
            fp_hash = pd.util.hash_pandas_object(fingerprint, index=False).astype(str).str[-6:]
            context["source_ip"] = "entity-" + fp_hash
        else:
            context["source_ip"] = "entity-unknown"
    else:
        context["source_ip"] = source_ip_raw

    src_port_behavior = _safe_col(df, ["ct_src_dport_ltm"], "0")
    dst_port_behavior = _safe_col(df, ["ct_dst_sport_ltm"], "0")

    context["source_profile"] = (
        context["source_ip"]
        + ":"
        + context["proto"]
        + ":"
        + context["service"]
        + ":srcgrp-"
        + src_port_behavior
    )
    context["destination_profile"] = context["state"] + ":dstgrp-" + dst_port_behavior
    return context


def _attack_type_from_hint(attack_hint: str, service: str, state: str) -> str:
    hint = str(attack_hint).lower()
    service_l = str(service).lower()
    state_l = str(state).lower()

    if "normal" in hint:
        return "Normal"
    if "backdoor" in hint or "worm" in hint:
        return "Backdoor/Worm Activity"
    if "fuzz" in hint or "exploit" in hint:
        return "Exploitation Attempt"
    if "dos" in hint or "ddos" in hint:
        return "DoS/DDoS"
    if "recon" in hint or "analysis" in hint:
        return "Reconnaissance"
    if "ftp" in service_l:
        return "Brute Force Attack"
    if "syn" in state_l or "int" in state_l:
        return "Reconnaissance"
    return "Suspicious Attack"


def _events_to_dicts(events: List[CorrelatedEvent]) -> List[Dict[str, Any]]:
    table: List[Dict[str, Any]] = []
    for event in events:
        table.append(
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "severity": event.severity,
                "confidence": round(event.confidence, 3),
                "source_ip": event.source_ip,
                "size": event.size,
                "avg_attack_probability": round(event.avg_attack_probability, 3),
                "avg_anomaly_score": round(event.avg_anomaly_score, 3),
                "top_proto": event.top_proto,
                "top_service": event.top_service,
                "top_state": event.top_state,
                "incident_statement": event.incident_statement,
            }
        )
    return table


def _cache_base_paths(csv_path: str, sample_size: int | None, target_col: str) -> Dict[str, Path]:
    source = Path(csv_path)
    cache_dir = source.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    stat = source.stat()
    key = f"{source.stem}_s{sample_size}_t{target_col}_m{int(stat.st_mtime)}_z{stat.st_size}"
    base = cache_dir / key
    return {
        "x_parquet": base.with_name(base.name + "_x.parquet"),
        "y_parquet": base.with_name(base.name + "_y.parquet"),
        "ctx_parquet": base.with_name(base.name + "_ctx.parquet"),
        "x_pickle": base.with_name(base.name + "_x.pkl"),
        "y_pickle": base.with_name(base.name + "_y.pkl"),
        "ctx_pickle": base.with_name(base.name + "_ctx.pkl"),
    }


def _read_cached_preprocessed(
    csv_path: str, sample_size: int | None, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame] | None:
    paths = _cache_base_paths(csv_path, sample_size, target_col)

    try:
        if paths["x_parquet"].exists() and paths["y_parquet"].exists() and paths["ctx_parquet"].exists():
            x_df = pd.read_parquet(paths["x_parquet"])
            y_df = pd.read_parquet(paths["y_parquet"])
            ctx_df = pd.read_parquet(paths["ctx_parquet"])
            y = y_df["label"].astype(int)
            return x_df, y, ctx_df
    except Exception:
        pass

    try:
        if paths["x_pickle"].exists() and paths["y_pickle"].exists() and paths["ctx_pickle"].exists():
            x_df = pd.read_pickle(paths["x_pickle"])
            y_df = pd.read_pickle(paths["y_pickle"])
            ctx_df = pd.read_pickle(paths["ctx_pickle"])
            y = y_df["label"].astype(int)
            return x_df, y, ctx_df
    except Exception:
        return None

    return None


def _write_cached_preprocessed(
    csv_path: str,
    sample_size: int | None,
    target_col: str,
    x_df: pd.DataFrame,
    y: pd.Series,
    context_df: pd.DataFrame,
) -> None:
    paths = _cache_base_paths(csv_path, sample_size, target_col)
    y_df = pd.DataFrame({"label": y.to_numpy()})

    try:
        x_df.to_parquet(paths["x_parquet"], index=False)
        y_df.to_parquet(paths["y_parquet"], index=False)
        context_df.to_parquet(paths["ctx_parquet"], index=False)
        return
    except Exception:
        pass

    try:
        x_df.to_pickle(paths["x_pickle"])
        y_df.to_pickle(paths["y_pickle"])
        context_df.to_pickle(paths["ctx_pickle"])
    except Exception:
        pass


def load_dataset(
    csv_path: str,
    sample_size: int | None = 40000,
    target_col: str = "label",
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if use_cache:
        cached = _read_cached_preprocessed(csv_path, sample_size, target_col)
        if cached is not None:
            return cached

    df = pd.read_csv(csv_path, low_memory=False)
    x_df, y, context_df = prepare_dataset(df, sample_size=sample_size, target_col=target_col)

    if use_cache:
        _write_cached_preprocessed(csv_path, sample_size, target_col, x_df, y, context_df)

    return x_df, y, context_df


def prepare_dataset(
    df: pd.DataFrame, sample_size: int | None = 40000, target_col: str = "label"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).sort_index()

    if target_col not in df.columns:
        raise ValueError(f"Expected a target column named '{target_col}' in the dataset.")

    y_raw = df[target_col]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int)
    else:
        y = pd.Series(pd.factorize(y_raw.astype(str))[0], index=df.index, dtype=int)
    context_df = _build_context_profiles(df)

    drop_cols = [
        c
        for c in [
            target_col,
            "attack_cat",
            "id",
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Timestamp",
            "Source Port",
            "Destination Port",
            "srcip",
            "src_ip",
            "source_ip",
        ]
        if c in df.columns
    ]
    x_df = df.drop(columns=drop_cols)
    x_df = x_df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = x_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            series = x_df[col]
            if series.isna().any():
                fill_value = float(series.median()) if series.notna().any() else 0.0
                x_df[col] = series.fillna(fill_value)

    categorical_cols = x_df.columns.difference(numeric_cols)
    if len(categorical_cols) > 0:
        x_df[categorical_cols] = x_df[categorical_cols].fillna("unknown").astype(str)

    x_df = pd.get_dummies(x_df, drop_first=False)
    x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    x_df = x_df.astype(np.float32, copy=False)
    return x_df, y, context_df


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # Keep ROC arrays compact for UI and exports.
    roc_points = [
        {"fpr": float(fpr[i]), "tpr": float(tpr[i])}
        for i in range(0, len(fpr), max(1, len(fpr) // 100))
    ]

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "classification_report": classification_report(y_true, y_pred, output_dict=False),
        "roc_curve_points": roc_points,
    }


def run_intelligent_ids_pipeline(
    csv_path: str,
    sample_size: int | None = 20000,
    target_col: str = "label",
    fast_mode: bool = False,
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
    llm_provider: str = "auto",
    enable_honeybadger: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    x_df, y, context_df = load_dataset(
        csv_path,
        sample_size=sample_size,
        target_col=target_col,
        use_cache=True,
    )
    return run_intelligent_ids_from_prepared(
        x_df=x_df,
        y=y,
        context_df=context_df,
        fast_mode=fast_mode,
        use_llm=use_llm,
        llm_model=llm_model,
        llm_provider=llm_provider,
        enable_honeybadger=enable_honeybadger,
        threshold=threshold,
    )


def run_intelligent_ids_from_dataframe(
    df: pd.DataFrame,
    sample_size: int | None = 20000,
    target_col: str = "label",
    fast_mode: bool = False,
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
    llm_provider: str = "auto",
    enable_honeybadger: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    x_df, y, context_df = prepare_dataset(df, sample_size=sample_size, target_col=target_col)
    return run_intelligent_ids_from_prepared(
        x_df=x_df,
        y=y,
        context_df=context_df,
        fast_mode=fast_mode,
        use_llm=use_llm,
        llm_model=llm_model,
        llm_provider=llm_provider,
        enable_honeybadger=enable_honeybadger,
        threshold=threshold,
    )


def run_live_network_pipeline(
    live_df: pd.DataFrame,
    baseline_csv_path: str,
    baseline_sample_size: int | None = 12000,
    fast_mode: bool = False,
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
    llm_provider: str = "auto",
    enable_honeybadger: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    # Train on historical baseline, infer on live rows.
    x_base, y_base, _ctx_base = load_dataset(baseline_csv_path, sample_size=baseline_sample_size)
    x_live, _y_live, ctx_live = prepare_dataset(live_df, sample_size=None)

    # Align live columns to baseline training schema.
    x_live = x_live.reindex(columns=x_base.columns, fill_value=0)

    detector = HybridDetector(fast_mode=fast_mode)
    detector.fit(x_base.to_numpy(dtype=np.float32), y_base.to_numpy())

    p_attack, anomaly_scores, confidence_scores = detector.predict_components(
        x_live.to_numpy(dtype=np.float32)
    )
    y_pred = (confidence_scores >= threshold).astype(int)

    # In live streams, strict thresholds can produce zero alerts; promote top anomalies for triage.
    if int(np.sum(y_pred)) == 0 and len(y_pred) >= 40:
        k = max(5, int(0.03 * len(y_pred)))
        top_idx = np.argsort(anomaly_scores)[-k:]
        y_pred[top_idx] = 1
        confidence_scores[top_idx] = np.maximum(confidence_scores[top_idx], 0.56)

    detections: List[DetectionOutput] = []
    for i in range(len(y_pred)):
        attack_type = _attack_type_from_hint(
            attack_hint=str(ctx_live.iloc[i]["attack_hint"]),
            service=str(ctx_live.iloc[i]["service"]),
            state=str(ctx_live.iloc[i]["state"]),
        )
        detections.append(
            DetectionOutput(
                sample_index=i,
                probability_attack=float(p_attack[i]),
                anomaly_score=float(anomaly_scores[i]),
                confidence_score=float(confidence_scores[i]),
                prediction=int(y_pred[i]),
                attack_type=attack_type,
                source_ip=str(ctx_live.iloc[i]["source_ip"]),
                proto=str(ctx_live.iloc[i]["proto"]),
                service=str(ctx_live.iloc[i]["service"]),
                state=str(ctx_live.iloc[i]["state"]),
                source_profile=str(ctx_live.iloc[i]["source_profile"]),
                destination_profile=str(ctx_live.iloc[i]["destination_profile"]),
            )
        )

    correlation = CorrelationEngine(window_size=8, similarity_threshold=0.58)
    events = correlation.correlate(detections)

    reasoning = ReasoningLayer(use_llm=use_llm, model=llm_model, provider=llm_provider)
    honeybadger = HoneybadgerLayer()

    final_outputs: List[Dict[str, Any]] = []
    for event in events[:10]:
        item = reasoning.explain_event(event)
        item["source_ip"] = event.source_ip
        item["incident_statement"] = event.incident_statement
        item["model_classification"] = "Attack"
        item["anomaly_score"] = f"{event.avg_anomaly_score:.3f}"
        if enable_honeybadger:
            item.update(honeybadger.build_playbook(event))
        final_outputs.append(item)

    metrics = {
        "accuracy": "n/a-live",
        "precision": "n/a-live",
        "recall": "n/a-live",
        "f1_score": "n/a-live",
        "roc_auc": "n/a-live",
        "classification_report": "Live mode does not have ground-truth labels.",
        "roc_curve_points": [],
    }

    detections_preview = [
        {
            "sample_index": d.sample_index,
            "classification": "Attack" if d.prediction == 1 else "Normal",
            "attack_type": d.attack_type,
            "confidence_score": round(d.confidence_score, 3),
            "anomaly_score": round(d.anomaly_score, 3),
            "source_ip": d.source_ip,
            "proto": d.proto,
            "service": d.service,
        }
        for d in detections[:50]
    ]

    return {
        "metrics": metrics,
        "total_samples_tested": int(len(detections)),
        "attack_predictions": int(np.sum(y_pred)),
        "correlated_events": int(len(events)),
        "honeybadger_enabled": bool(enable_honeybadger),
        "events_table": _events_to_dicts(events),
        "detections_preview": detections_preview,
        "final_outputs": final_outputs,
    }


def run_intelligent_ids_from_prepared(
    x_df: pd.DataFrame,
    y: pd.Series,
    context_df: pd.DataFrame,
    fast_mode: bool = False,
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
    llm_provider: str = "auto",
    enable_honeybadger: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    x = x_df.to_numpy(dtype=np.float32)

    x_train, x_test, y_train, y_test, _ctx_train, ctx_test = train_test_split(
        x,
        y.to_numpy(),
        context_df.reset_index(drop=True),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    detector = HybridDetector(fast_mode=fast_mode)
    detector.fit(x_train, y_train)

    x_train_scaled = detector.scaler.transform(x_train)
    x_test_scaled = detector.scaler.transform(x_test)
    x_train_temporal = detector._add_temporal_features(x_train_scaled)
    x_test_temporal = detector._add_temporal_features(x_test_scaled)

    train_probs = {
        "LSTM proxy": detector.lstm_proxy.predict_proba(x_train_temporal)[:, 1],
        "CNN proxy": detector.cnn_proxy.predict_proba(x_train_scaled)[:, 1],
        "Random Forest": detector.rf.predict_proba(x_train)[:, 1],
    }
    test_probs = {
        "LSTM proxy": detector.lstm_proxy.predict_proba(x_test_temporal)[:, 1],
        "CNN proxy": detector.cnn_proxy.predict_proba(x_test_scaled)[:, 1],
        "Random Forest": detector.rf.predict_proba(x_test)[:, 1],
    }

    train_hybrid_probs, _, train_confidence_scores = detector.predict_components(x_train)

    p_attack, anomaly_scores, confidence_scores = detector.predict_components(x_test)
    y_pred = (confidence_scores >= threshold).astype(int)

    train_probs["Hybrid Ensemble"] = train_confidence_scores
    test_probs["Hybrid Ensemble"] = confidence_scores

    model_accuracy = []
    for model_name in ["LSTM proxy", "CNN proxy", "Random Forest", "Hybrid Ensemble"]:
        train_pred = (train_probs[model_name] >= 0.5).astype(int)
        test_pred = (test_probs[model_name] >= 0.5).astype(int)
        model_accuracy.append(
            {
                "model": model_name,
                "train_accuracy": round(float(accuracy_score(y_train, train_pred)), 4),
                "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
            }
        )

    detections: List[DetectionOutput] = []
    for i in range(len(y_pred)):
        attack_type = _attack_type_from_hint(
            attack_hint=str(ctx_test.iloc[i]["attack_hint"]),
            service=str(ctx_test.iloc[i]["service"]),
            state=str(ctx_test.iloc[i]["state"]),
        )

        detections.append(
            DetectionOutput(
                sample_index=i,
                probability_attack=float(p_attack[i]),
                anomaly_score=float(anomaly_scores[i]),
                confidence_score=float(confidence_scores[i]),
                prediction=int(y_pred[i]),
                attack_type=attack_type,
                source_ip=str(ctx_test.iloc[i]["source_ip"]),
                proto=str(ctx_test.iloc[i]["proto"]),
                service=str(ctx_test.iloc[i]["service"]),
                state=str(ctx_test.iloc[i]["state"]),
                source_profile=str(ctx_test.iloc[i]["source_profile"]),
                destination_profile=str(ctx_test.iloc[i]["destination_profile"]),
            )
        )

    correlation = CorrelationEngine(window_size=8, similarity_threshold=0.58)
    events = correlation.correlate(detections)

    reasoning = ReasoningLayer(use_llm=use_llm, model=llm_model, provider=llm_provider)
    honeybadger = HoneybadgerLayer()

    final_outputs: List[Dict[str, Any]] = []
    for event in events[:10]:
        item = reasoning.explain_event(event)
        item["source_ip"] = event.source_ip
        item["incident_statement"] = event.incident_statement
        item["model_classification"] = "Attack"
        item["anomaly_score"] = f"{event.avg_anomaly_score:.3f}"

        if enable_honeybadger:
            item.update(honeybadger.build_playbook(event))

        final_outputs.append(item)

    metrics = evaluate_model(y_test, y_pred, confidence_scores)

    detections_preview = [
        {
            "sample_index": d.sample_index,
            "classification": "Attack" if d.prediction == 1 else "Normal",
            "attack_type": d.attack_type,
            "confidence_score": round(d.confidence_score, 3),
            "anomaly_score": round(d.anomaly_score, 3),
            "source_ip": d.source_ip,
            "proto": d.proto,
            "service": d.service,
        }
        for d in detections[:50]
    ]

    return {
        "metrics": metrics,
        "total_samples_tested": int(len(y_test)),
        "attack_predictions": int(np.sum(y_pred)),
        "correlated_events": int(len(events)),
        "honeybadger_enabled": bool(enable_honeybadger),
        "model_accuracy": model_accuracy,
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "confidence_scores": confidence_scores.tolist(),
        "attack_probabilities": p_attack.tolist(),
        "events_table": _events_to_dicts(events),
        "detections_preview": detections_preview,
        "final_outputs": final_outputs,
    }


if __name__ == "__main__":
    preferred_data_path = "data/processed/multi_dataset_combined.csv"
    fallback_data_path = "data/processed/clean_data.csv"
    data_path = preferred_data_path if os.path.exists(preferred_data_path) else fallback_data_path
    result = run_intelligent_ids_pipeline(
        csv_path=data_path,
        sample_size=12000,
        use_llm=True,
        llm_model=os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        llm_provider=os.getenv("LLM_PROVIDER", "auto"),
        enable_honeybadger=True,
    )

    print("=== AI Intrusion Detection Dashboard Backend ===")
    print(f"Samples tested: {result['total_samples_tested']}")
    print(f"Attack predictions: {result['attack_predictions']}")
    print(f"Correlated incidents: {result['correlated_events']}")
    print(f"Accuracy: {result['metrics']['accuracy']}")
    print(f"Precision: {result['metrics']['precision']}")
    print(f"Recall: {result['metrics']['recall']}")
    print(f"F1: {result['metrics']['f1_score']}")
    print(f"ROC-AUC: {result['metrics']['roc_auc']}")

    if result["final_outputs"]:
        first = result["final_outputs"][0]
        print("\n--- Sample Incident ---")
        print(f"Type: {first['attack_type']}")
        print(f"Source IP: {first['source_ip']}")
        print(f"Severity: {first['severity']}")
        print(f"Classification: {first['model_classification']}")
        print(f"Anomaly Score: {first['anomaly_score']}")
        print(f"Explanation: {first['reason']}")
        print(f"Recommended Action: {first['recommended_action']}")
