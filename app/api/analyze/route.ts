import { NextResponse } from "next/server";
import Sentiment from "sentiment";

const vader = require("vader-sentiment");
const sentiment = new Sentiment();

export const runtime = "nodejs";

type SentimentLabel = "positive" | "negative";

type ModelResult = {
  name: string;
  family: string;
  label: SentimentLabel;
  posProb: number;
  confidence: number;
  notes: string;
  latencyMs: number;
};

const POSITIVE_TERMS = new Set([
  "excellent",
  "amazing",
  "great",
  "beautiful",
  "love",
  "best",
  "enjoyed",
  "wonderful",
  "fantastic",
  "moving",
]);

const NEGATIVE_TERMS = new Set([
  "awful",
  "terrible",
  "worst",
  "boring",
  "hate",
  "waste",
  "poor",
  "mess",
  "disappointing",
  "flat",
]);

const NEGATORS = new Set(["not", "never", "hardly", "barely", "no"]);

function clamp01(value: number) {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function scoreToProbability(score: number, scale = 2.6) {
  return 1 / (1 + Math.exp(-score / scale));
}

function buildResult(name: string, family: string, posProb: number, confidence: number, notes: string, latencyMs: number): ModelResult {
  return {
    name,
    family,
    label: posProb >= 0.5 ? "positive" : "negative",
    posProb: clamp01(posProb),
    confidence: clamp01(confidence),
    notes,
    latencyMs,
  };
}

function lexiconClassic(text: string): Omit<ModelResult, "latencyMs"> {
  const tokens = text.toLowerCase().match(/[a-z']+/g) ?? [];
  let score = 0;
  let matched = 0;

  for (let i = 0; i < tokens.length; i += 1) {
    const current = tokens[i];
    const previous = i > 0 ? tokens[i - 1] : "";
    const inverted = NEGATORS.has(previous);

    if (POSITIVE_TERMS.has(current)) {
      score += inverted ? -1 : 1;
      matched += 1;
    }
    if (NEGATIVE_TERMS.has(current)) {
      score += inverted ? 1 : -1;
      matched += 1;
    }
  }

  const posProb = scoreToProbability(score, 2.2);
  const confidence = clamp01(Math.min(1, Math.abs(score) / Math.max(2, matched + 1)) + 0.1);

  return {
    name: "Lexicon Classic",
    family: "Rule-based",
    label: posProb >= 0.5 ? "positive" : "negative",
    posProb,
    confidence,
    notes: matched > 0 ? `Matched ${matched} weighted sentiment terms.` : "No strong keywords; treated as mild sentiment.",
  };
}

function sentimentJsModel(text: string): Omit<ModelResult, "latencyMs"> {
  const result = sentiment.analyze(text);
  const comparative = Number.isFinite(result.comparative) ? result.comparative : 0;
  const posProb = scoreToProbability(comparative, 0.8);
  const confidence = clamp01(Math.abs(comparative) / 1.8 + 0.15);

  return {
    name: "Sentiment.js",
    family: "Bag-of-words",
    label: posProb >= 0.5 ? "positive" : "negative",
    posProb,
    confidence,
    notes: `Comparative score: ${comparative.toFixed(3)}.`,
  };
}

function vaderModel(text: string): Omit<ModelResult, "latencyMs"> {
  const scores = vader.SentimentIntensityAnalyzer.polarity_scores(text);
  const compound = Number(scores.compound ?? 0);
  const posProb = clamp01((compound + 1) / 2);
  const confidence = clamp01(Math.abs(compound));

  return {
    name: "VADER",
    family: "Social text",
    label: posProb >= 0.5 ? "positive" : "negative",
    posProb,
    confidence,
    notes: `Compound score: ${compound.toFixed(3)}.`,
  };
}

function contextAwareModel(text: string): Omit<ModelResult, "latencyMs"> {
  const lc = text.toLowerCase();
  const base = sentiment.analyze(text).comparative || 0;
  let score = base;
  const notes: string[] = [];

  if (/(yeah right|as if|sure because)/.test(lc)) {
    score -= 1.2;
    notes.push("Sarcasm cue found.");
  }

  const pivotParts = lc.split(/\b(?:but|however|although)\b/);
  if (pivotParts.length > 1) {
    const tail = pivotParts[pivotParts.length - 1] || "";
    score = score * 0.45 + (sentiment.analyze(tail).comparative || 0) * 0.9;
    notes.push("Weighted latter clause after contrast cue.");
  }

  const shoutCount = (text.match(/[A-Z]{2,}/g) ?? []).length;
  if (shoutCount > 1) {
    score += score >= 0 ? 0.2 : -0.2;
    notes.push("All-caps intensity adjusted.");
  }

  const posProb = scoreToProbability(score, 0.9);
  const confidence = clamp01(Math.abs(score) / 1.5 + 0.2);

  return {
    name: "Context Lens",
    family: "Heuristic",
    label: posProb >= 0.5 ? "positive" : "negative",
    posProb,
    confidence,
    notes: notes.length ? notes.join(" ") : "No special context cues detected.",
  };
}

async function transformerModel(text: string): Promise<Omit<ModelResult, "latencyMs"> | null> {
  const token = process.env.HF_TOKEN;
  if (!token) return null;

  const response = await fetch(
    "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: text,
        options: { wait_for_model: true },
      }),
      cache: "no-store",
    },
  );

  if (!response.ok) return null;

  const raw = await response.json();
  const list = Array.isArray(raw) ? (Array.isArray(raw[0]) ? raw[0] : raw) : [];
  if (!Array.isArray(list) || list.length === 0) return null;

  let pos = 0.5;
  let neg = 0.5;
  for (const item of list) {
    const label = String(item.label ?? "").toLowerCase();
    const score = Number(item.score ?? 0);
    if (label.includes("positive") || label === "label_2") pos = score;
    if (label.includes("negative") || label === "label_0") neg = score;
  }

  const total = Math.max(pos + neg, 1e-6);
  const posProb = clamp01(pos / total);

  return {
    name: "RoBERTa (HF)",
    family: "Transformer",
    label: posProb >= 0.5 ? "positive" : "negative",
    posProb,
    confidence: clamp01(Math.max(pos, neg)),
    notes: "API-backed transformer analysis.",
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

    const localModels = [lexiconClassic, sentimentJsModel, vaderModel, contextAwareModel];
    const localResults: ModelResult[] = localModels.map((fn) => {
      const start = Date.now();
      const result = fn(text);
      return buildResult(result.name, result.family, result.posProb, result.confidence, result.notes, Date.now() - start);
    });

    const tfStart = Date.now();
    const transformer = await transformerModel(text);
    if (transformer) {
      localResults.push(
        buildResult(
          transformer.name,
          transformer.family,
          transformer.posProb,
          transformer.confidence,
          transformer.notes,
          Date.now() - tfStart,
        ),
      );
    }

    const posVotes = localResults.filter((item) => item.label === "positive").length;
    const total = localResults.length;
    const consensusLabel: SentimentLabel = posVotes >= Math.ceil(total / 2) ? "positive" : "negative";
    const avgConfidence = localResults.reduce((sum, item) => sum + item.confidence, 0) / total;

    return NextResponse.json({
      input: text,
      models: localResults,
      consensus: {
        label: consensusLabel,
        posVotes,
        total,
        agreement: `${posVotes}/${total} models positive`,
        confidence: clamp01(avgConfidence),
      },
      meta: {
        usedTransformer: Boolean(transformer),
        latencyMs: Date.now() - t0,
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