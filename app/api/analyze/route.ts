import { NextResponse } from "next/server";

export const runtime = "nodejs";

type ModelResult = {
  name: string;
  family: string;
  label: "positive" | "negative";
  posProb: number;
  confidence: number;
  notes: string;
  latencyMs: number;
  testAccuracy?: number;
  tunedThreshold: number;
};

type BackendModel = {
  key: string;
  name: string;
  family: string;
  pos_prob: number;
  latency_ms: number;
  test_accuracy?: number | null;
  threshold?: number;
  label?: "positive" | "negative";
  confidence?: number;
  note?: string;
};

function clamp01(value: number) {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function buildResult(model: BackendModel): ModelResult {
  const posProb = clamp01(Number(model.pos_prob));
  const tunedThreshold = clamp01(typeof model.threshold === "number" ? model.threshold : 0.5);
  const confidence = clamp01(
    typeof model.confidence === "number" ? model.confidence : Math.abs(posProb - tunedThreshold) * 2,
  );
  const acc = typeof model.test_accuracy === "number" ? clamp01(model.test_accuracy) : undefined;

  return {
    name: model.name,
    family: model.family,
    label: model.label === "positive" || model.label === "negative" ? model.label : posProb >= tunedThreshold ? "positive" : "negative",
    posProb,
    confidence,
    notes:
      model.note || (acc !== undefined ? `Offline test accuracy ${(acc * 100).toFixed(1)}%.` : "Prediction from trained model."),
    latencyMs: Math.max(0, Number(model.latency_ms) || 0),
    testAccuracy: acc,
    tunedThreshold,
  };
}

function deriveConsensus(models: ModelResult[]) {
  const posVotes = models.filter((item) => item.label === "positive").length;
  const total = models.length;
  const label = total === 4 && posVotes === 2 ? "indeterminate" : posVotes > total / 2 ? "positive" : "negative";
  const confidence = models.reduce((sum, item) => sum + item.confidence, 0) / Math.max(total, 1);
  return {
    label,
    posVotes,
    total,
    agreement: `${posVotes}/${total} models positive`,
    confidence: clamp01(confidence),
    reason: label === "indeterminate" ? "Exact split across models." : "Majority from model outputs.",
  };
}

export async function POST(request: Request) {
  const t0 = Date.now();

  try {
    const body = await request.json();
    const text = String(body.text ?? "").trim();

    if (!text) {
      return NextResponse.json({ error: "Text is required." }, { status: 400 });
    }

    if (text.length > 4000) {
      return NextResponse.json({ error: "Text too long. Limit to 4000 characters." }, { status: 400 });
    }

    const backendUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
    let response: Response;
    try {
      response = await fetch(`${backendUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
        cache: "no-store",
      });
    } catch {
      return NextResponse.json(
        {
          error: "ML backend is unreachable.",
          detail: `Start backend/main.py and verify NEXT_PUBLIC_API_URL (${backendUrl}).`,
        },
        { status: 502 },
      );
    }

    if (!response.ok) {
      const raw = await response.text();
      const detail = raw.slice(0, 300);
      return NextResponse.json(
        {
          error: "ML backend request failed.",
          detail: detail || `Status ${response.status}`,
        },
        { status: 502 },
      );
    }

    const payload = await response.json();
    const incoming = Array.isArray(payload?.models) ? (payload.models as BackendModel[]) : [];
    if (incoming.length === 0) {
      return NextResponse.json(
        {
          error: "ML backend returned no model outputs.",
        },
        { status: 502 },
      );
    }

    const models = incoming.map((item) => buildResult(item));
    const fallbackConsensus = deriveConsensus(models);

    const backendConsensus = payload?.consensus;
    const consensus = {
      label:
        backendConsensus?.label === "positive" ||
        backendConsensus?.label === "negative" ||
        backendConsensus?.label === "indeterminate"
          ? backendConsensus.label
          : fallbackConsensus.label,
      posVotes: Number(backendConsensus?.pos_votes ?? fallbackConsensus.posVotes),
      total: Number(backendConsensus?.total ?? fallbackConsensus.total),
      agreement: `${Number(backendConsensus?.pos_votes ?? fallbackConsensus.posVotes)}/${Number(
        backendConsensus?.total ?? fallbackConsensus.total,
      )} models positive`,
      confidence: clamp01(Number(backendConsensus?.confidence ?? fallbackConsensus.confidence)),
      reason: String(backendConsensus?.reason ?? fallbackConsensus.reason ?? ""),
    };

    return NextResponse.json({
      input: text,
      models,
      consensus,
      meta: {
        latencyMs: Date.now() - t0,
        backendLatencyMs: Number(payload?.meta?.latency_ms ?? 0),
        tokenCount: Number(payload?.meta?.token_count ?? 0),
        shortTextGuard: Boolean(payload?.meta?.short_text_guard),
        guardMessage: String(payload?.meta?.guard_message ?? ""),
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Unable to analyze text.",
        detail: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}