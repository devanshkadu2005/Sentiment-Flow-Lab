"use client";

import { useEffect, useMemo, useState } from "react";

type SentimentLabel = "positive" | "negative";
type ConsensusLabel = SentimentLabel | "indeterminate";

type ModelResult = {
  name: string;
  family: string;
  label: SentimentLabel;
  posProb: number;
  confidence: number;
  notes: string;
  latencyMs: number;
  testAccuracy?: number;
  tunedThreshold: number;
};

type AnalysisResponse = {
  input: string;
  models: ModelResult[];
  consensus: {
    label: ConsensusLabel;
    posVotes: number;
    total: number;
    agreement: string;
    confidence: number;
    reason?: string;
  };
  meta: {
    latencyMs: number;
    backendLatencyMs?: number;
    tokenCount?: number;
    shortTextGuard?: boolean;
    guardMessage?: string;
  };
  error?: string;
};

const EXAMPLES = [
  {
    label: "Positive example 1",
    expected: "positive",
    text: "Great movie with excellent acting and a satisfying ending.",
  },
  {
    label: "Negative example 1",
    expected: "negative",
    text: "Boring movie with weak writing and poor acting.",
  },
  {
    label: "Negative example 2",
    expected: "negative",
    text: "Disappointing film with a slow plot and dull dialogue.",
  },
  {
    label: "Positive example 2",
    expected: "positive",
    text: "Enjoyable movie with great performances and a clear story.",
  },
] as const;

function formatPct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function clamp01(value: number) {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function effectiveThreshold(model: ModelResult, sensitivityOffset: number) {
  return clamp01(model.tunedThreshold + sensitivityOffset);
}

function deriveConsensus(models: ModelResult[], sensitivityOffset: number, shortTextGuard: boolean) {
  if (models.length === 0) {
    return { label: "indeterminate" as ConsensusLabel, posVotes: 0, total: 0, confidence: 0, reason: "No model outputs." };
  }

  if (shortTextGuard) {
    return {
      label: "indeterminate" as ConsensusLabel,
      posVotes: 0,
      total: models.length,
      confidence: 0,
      reason: "Input is too short for reliable inference.",
    };
  }

  const posVotes = models.filter((item) => item.posProb >= effectiveThreshold(item, sensitivityOffset)).length;

  if (models.length === 4 && posVotes === 2) {
    return {
      label: "indeterminate" as const,
      posVotes,
      total: models.length,
      confidence: 0,
      reason: "Exact 2/4 split across calibrated model thresholds.",
    };
  }

  const label: SentimentLabel = posVotes > models.length / 2 ? "positive" : "negative";
  const confidence =
    models.reduce((sum, item) => sum + Math.abs(item.posProb - effectiveThreshold(item, sensitivityOffset)), 0) / models.length;
  return { label, posVotes, total: models.length, confidence, reason: "Majority decision using calibrated thresholds." };
}

async function analyzeText(text: string): Promise<AnalysisResponse> {
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return response.json();
}

function ModelCard({ model, sensitivityOffset }: { model: ModelResult; sensitivityOffset: number }) {
  const appliedThreshold = effectiveThreshold(model, sensitivityOffset);
  const label = model.posProb >= appliedThreshold ? "positive" : "negative";
  return (
    <article className="model-card">
      <div className="model-head">
        <strong>{model.name}</strong>
        <span className="badge">{model.family}</span>
      </div>
      <div className={`label ${label}`}>{label === "positive" ? "Positive" : "Negative"}</div>
      <div className="mini-stat">
        Confidence {formatPct(model.confidence)} | Latency {model.latencyMs}ms
      </div>
      <div className="mini-stat">Calibrated threshold: {model.tunedThreshold.toFixed(2)} | Applied: {appliedThreshold.toFixed(2)}</div>
      {typeof model.testAccuracy === "number" ? (
        <div className="mini-stat">Test accuracy: {formatPct(model.testAccuracy)}</div>
      ) : null}
      <div className="bar">
        <span style={{ width: `${Math.round(model.posProb * 100)}%` }} />
      </div>
      <div className="mini-stat">Positive probability: {formatPct(model.posProb)}</div>
      <div className="hint">{model.notes}</div>
    </article>
  );
}

export default function Page() {
  const [textA, setTextA] = useState("");
  const [sensitivityOffset, setSensitivityOffset] = useState(0);
  const [liveMode, setLiveMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [singleResult, setSingleResult] = useState<AnalysisResponse | null>(null);

  const activeLength = useMemo(() => textA.length, [textA]);

  useEffect(() => {
    if (!liveMode) return;
    if (!textA.trim()) {
      setSingleResult(null);
      return;
    }

    const timer = setTimeout(async () => {
      setLoading(true);
      setError("");
      try {
        const result = await analyzeText(textA);
        if (result.error) {
          setError(result.error);
          setSingleResult(null);
        } else {
          setSingleResult(result);
        }
      } catch {
        setError("Network issue while running live analysis.");
      } finally {
        setLoading(false);
      }
    }, 380);

    return () => clearTimeout(timer);
  }, [liveMode, textA]);

  async function runAnalysis() {
    setError("");
    setLoading(true);
    setSingleResult(null);

    try {
      if (!textA.trim()) throw new Error("Please enter some text.");
      const result = await analyzeText(textA);
      if (result.error) throw new Error(result.error);
      setSingleResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  const singleConsensus = singleResult
    ? deriveConsensus(singleResult.models, sensitivityOffset, Boolean(singleResult.meta.shortTextGuard))
    : null;

  return (
    <main className="page-shell">
      <section className="hero">
        <h1>Sentiment Flow Lab</h1>
        <p>A clean sentiment workspace for quick single-input analysis with live feedback and adjustable decision threshold.</p>
      </section>

      <section className="panel">
        <h2>Input</h2>

        <h3 style={{ marginTop: "1rem", marginBottom: "0.5rem" }}>Quick examples</h3>
        <p className="hint" style={{ marginTop: "0", marginBottom: "0.55rem" }}>
          Presets are tuned for default sensitivity offset 0.00 and show an expected majority outcome.
        </p>
        <div className="samples">
          {EXAMPLES.map((example) => (
            <button
              className="chip"
              type="button"
              key={example.label}
              onClick={() => {
                setTextA(example.text);
                setSensitivityOffset(0);
              }}
            >
              {example.label} ({example.expected})
            </button>
          ))}
        </div>

        <div className="input-grid" style={{ marginTop: "0.9rem" }}>
          <div className="input-card">
            <div className="input-title">Input text</div>
            <textarea
              value={textA}
              onChange={(event) => setTextA(event.target.value)}
              placeholder="Paste a review, tweet, feedback note, or any sentence..."
            />
          </div>
        </div>

        <div className="toolbar">
          <button className="action-btn primary" type="button" onClick={runAnalysis} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze"}
          </button>

          <button
            className="action-btn"
            type="button"
            onClick={() => {
              setTextA("");
              setSingleResult(null);
            }}
          >
            Reset
          </button>

          <label className="toggle">
            <input type="checkbox" checked={liveMode} onChange={(event) => setLiveMode(event.target.checked)} />
            Live mode (single)
          </label>

          <label className="toggle">
            Sensitivity offset {sensitivityOffset >= 0 ? "+" : ""}
            {sensitivityOffset.toFixed(2)}
            <input
              type="range"
              min={-0.1}
              max={0.1}
              step={0.01}
              value={sensitivityOffset}
              onChange={(event) => setSensitivityOffset(Number(event.target.value))}
            />
          </label>

          <span className="meta-note">Characters: {activeLength}</span>
        </div>

        <p className="threshold-help">
          Each model has its own calibrated threshold learned during evaluation. Sensitivity offset shifts those thresholds
          globally. In live mode, every pause while typing re-runs analysis. Positive offset makes positive classification
          harder; negative offset makes it easier.
        </p>

        {singleResult?.meta.shortTextGuard ? (
          <p style={{ color: "#b7791f", marginTop: "0.5rem" }}>
            {singleResult.meta.guardMessage || "Short-text guard is active. Use at least 3 tokens for reliable inference."}
          </p>
        ) : null}

        {error ? <p style={{ color: "#cd4f2f", marginTop: "0.8rem" }}>{error}</p> : null}
      </section>

      {singleResult && (
        <section className="panel">
          <h2>Model Results</h2>
          <div className="result-grid">
            {singleResult.models.map((model) => (
              <ModelCard key={model.name} model={model} sensitivityOffset={sensitivityOffset} />
            ))}
          </div>

          {singleConsensus ? (
            <div className={`consensus ${singleConsensus.label}`}>
              <h3>
                {singleConsensus.label === "positive"
                  ? "Overall: Positive"
                  : singleConsensus.label === "negative"
                    ? "Overall: Negative"
                    : "Overall: Non-deterministic"}
              </h3>
              <p className="mini-stat">
                {singleConsensus.posVotes}/{singleConsensus.total} models above threshold
                {singleConsensus.label === "indeterminate"
                  ? ` | ${singleConsensus.reason || "No deterministic direction"}`
                  : ` | decision confidence ${formatPct(singleConsensus.confidence)}`}
              </p>
              <p className="mini-stat">API latency: {singleResult.meta.latencyMs}ms</p>
            </div>
          ) : null}
        </section>
      )}
    </main>
  );
}