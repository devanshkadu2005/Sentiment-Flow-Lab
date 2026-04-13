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
};

type AnalysisResponse = {
  input: string;
  models: ModelResult[];
  consensus: {
    label: SentimentLabel;
    posVotes: number;
    total: number;
    agreement: string;
    confidence: number;
  };
  meta: {
    usedTransformer: boolean;
    latencyMs: number;
  };
  error?: string;
};

const EXAMPLES = [
  {
    label: "Cinematic high",
    text: "This was deeply moving and unexpectedly brilliant. I loved how every scene built tension with purpose.",
  },
  {
    label: "Harsh review",
    text: "A complete mess. Flat writing, weak acting, and a finale that felt rushed and careless.",
  },
  {
    label: "Sarcastic",
    text: "Yeah right, because two hours of predictable dialogue is exactly what everyone dreams of.",
  },
  {
    label: "Mixed",
    text: "The visuals were beautiful, but the pacing dragged in the second half.",
  },
];

function formatPct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function deriveConsensus(models: ModelResult[], threshold: number) {
  if (models.length === 0) {
    return { label: "indeterminate" as ConsensusLabel, posVotes: 0, total: 0, confidence: 0 };
  }

  const posVotes = models.filter((item) => item.posProb >= threshold).length;

  if (models.length === 4 && posVotes === 2) {
    return {
      label: "indeterminate" as const,
      posVotes,
      total: models.length,
      confidence: 0,
    };
  }

  const label: SentimentLabel = posVotes > models.length / 2 ? "positive" : "negative";
  const confidence = models.reduce((sum, item) => sum + Math.abs(item.posProb - threshold), 0) / models.length;
  return { label, posVotes, total: models.length, confidence };
}

async function analyzeText(text: string): Promise<AnalysisResponse> {
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return response.json();
}

function ModelCard({ model, threshold }: { model: ModelResult; threshold: number }) {
  const label = model.posProb >= threshold ? "positive" : "negative";
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
  const [threshold, setThreshold] = useState(0.5);
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

  const singleConsensus = singleResult ? deriveConsensus(singleResult.models, threshold) : null;

  return (
    <main className="page-shell">
      <section className="hero">
        <h1>Sentiment Flow Lab</h1>
        <p>A clean sentiment workspace for quick single-input analysis with live feedback and adjustable decision threshold.</p>
      </section>

      <section className="panel">
        <h2>Input</h2>

        <h3 style={{ marginTop: "1rem", marginBottom: "0.5rem" }}>Quick examples</h3>
        <div className="samples">
          {EXAMPLES.map((example) => (
            <button
              className="chip"
              type="button"
              key={example.label}
              onClick={() => setTextA(example.text)}
            >
              {example.label}
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
            Threshold {threshold.toFixed(2)}
            <input
              type="range"
              min={0.35}
              max={0.65}
              step={0.01}
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />
          </label>

          <span className="meta-note">Characters: {activeLength}</span>
        </div>

        <p className="threshold-help">
          Threshold controls the positive/negative cutoff for each model. In live mode, every pause while typing re-runs
          analysis using this cutoff. Example: threshold 0.60 means a model must output at least 60% positive probability
          to count as positive.
        </p>

        {error ? <p style={{ color: "#cd4f2f", marginTop: "0.8rem" }}>{error}</p> : null}
      </section>

      {singleResult && (
        <section className="panel">
          <h2>Model Results</h2>
          <div className="result-grid">
            {singleResult.models.map((model) => (
              <ModelCard key={model.name} model={model} threshold={threshold} />
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
                  ? " | exactly 2/4 split, so no deterministic direction"
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