import { useEffect, useState, type ChangeEvent, type FormEvent, type ReactNode } from "react";
import axios from "axios";
import "./result-states.css";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Clock3,
  FileText,
  Gauge,
  Globe2,
  Layers3,
  LoaderCircle,
  Search,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
type Contribution = { term: string; contribution: number; direction: "FAKE" | "REAL" };
type EvidenceMatch = {
  title: string;
  snippet: string;
  dataset_label: string;
  similarity: number;
  evidence_status: string;
  source: string;
  date?: string | null;
};
type SourceCredibility = {
  available: boolean;
  status?: string;
  message: string;
  publisher?: string;
  domain?: string;
  real_articles?: number;
  fake_articles?: number;
  total_articles?: number;
  minimum_sample_size?: number;
  credibility_score?: number;
  method?: string;
};
type TransformerPrediction = {
  label: string;
  fake_probability: number | null;
  real_probability: number | null;
  confidence: number | null;
  error?: string;
  disclaimer?: string;
  model: string;
};
type Prediction = {
  label: string;
  fake_probability: number;
  real_probability: number;
  confidence: number;
  explanation?: Contribution[];
  explanation_available?: boolean;
  explanation_message?: string;
  explanation_method?: string;
  evidence?: EvidenceMatch[];
  source_credibility?: SourceCredibility;
  disclaimer: string;
  title?: string;
  url?: string | null;
  analyzed_text?: string;
  transformer_prediction?: TransformerPrediction;
};
type Analytics = {
  available: boolean;
  message?: string;
  total_articles?: number;
  duplicate_rows_removed?: number;
  conflicting_duplicate_rows_excluded?: number;
  class_distribution: { label: string; count: number }[];
  subjects: { subject: string; fake: number; real: number }[];
  topics?: { category: string; fake: number; real: number; total: number; fake_share: number | null }[];
  monthly_trends: { month: string; fake: number; real: number }[];
  linguistic_comparison: {
    label: string;
    avg_words: number;
    avg_exclamations: number;
    avg_capital_ratio: number;
    avg_flesch_reading_ease?: number | null;
  }[];
  emotional_language_test?: {
    metric: string;
    method: string;
    real_sample_size: number;
    fake_sample_size: number;
    real_median: number | null;
    fake_median: number | null;
    p_value: number | null;
    rank_biserial_effect_size: number | null;
    effect_magnitude: string;
    interpretation: string;
    assumptions: string;
  } | null;
};
type Sources = {
  available: boolean;
  message: string;
  minimum_sample_size?: number;
  sources: {
    publisher: string;
    domain?: string;
    real_articles: number;
    fake_articles: number;
    total_articles: number;
    credibility_score: number;
    method?: string;
    minimum_sample_size?: number;
  }[];
};
const fmt = (n: number) => new Intl.NumberFormat("en-US").format(n);

function predictionClass(label: string): "FAKE" | "REAL" | "UNKNOWN" {
  const normalized = label.trim().toUpperCase().replace(/[\s_]+/g, "-");
  if (["FAKE", "FAKE-LIKE", "FAKE-NEWS", "MISINFORMATION"].includes(normalized)) return "FAKE";
  if (["REAL", "REAL-LIKE", "REAL-NEWS"].includes(normalized)) return "REAL";
  return "UNKNOWN";
}

function HighlightedArticle({ text, contributions }: { text: string; contributions: Contribution[] }) {
  const matches: { start: number; end: number; direction: "FAKE" | "REAL"; term: string }[] = [];
  for (const item of contributions) {
    const term = item.term.trim();
    if (!term || (item.direction !== "FAKE" && item.direction !== "REAL")) continue;
    const pattern = term.split(/\s+/).map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s+");
    const expression = new RegExp(`(?<![\\p{L}\\p{N}_])(${pattern})(?![\\p{L}\\p{N}_])`, "giu");
    for (const match of text.matchAll(expression)) {
      if (match.index !== undefined) {
        matches.push({ start: match.index, end: match.index + match[0].length, direction: item.direction, term: match[0] });
      }
    }
  }
  matches.sort((a, b) => a.start - b.start || b.end - a.end);
  const selected: typeof matches = [];
  let end = -1;
  for (const match of matches) {
    if (match.start < end) continue;
    selected.push(match);
    end = match.end;
  }
  const content: ReactNode[] = [];
  let cursor = 0;
  for (const match of selected) {
    if (cursor < match.start) content.push(text.slice(cursor, match.start));
    content.push(
      <mark className={`article-highlight ${match.direction.toLowerCase()}`} key={`${match.start}-${match.end}`}>
        {match.term}
      </mark>,
    );
    cursor = match.end;
  }
  if (cursor < text.length) content.push(text.slice(cursor));
  return <div className="highlighted-article">{content}</div>;
}

export default function App() {
  const [page, setPage] = useState("Overview");
  const [apiReady, setApiReady] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [articleTitle, setArticleTitle] = useState("");
  const [articleText, setArticleText] = useState("");
  const [articleUrl, setArticleUrl] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [sources, setSources] = useState<Sources | null>(null);
  const [batchBusy, setBatchBusy] = useState(false);
  const [transformerReady, setTransformerReady] = useState(false);
  useEffect(() => {
    function checkHealth() {
      axios
        .get(`${API}/health`)
        .then((r) => {
          setApiReady(true);
          setModelReady(Boolean(r.data.model_ready));
          setTransformerReady(Boolean(r.data.transformer_ready));
        })
        .catch(() => {
          setApiReady(false);
          setModelReady(false);
          setTransformerReady(false);
        });
    }

    checkHealth();
    const interval = setInterval(checkHealth, 5000);

    axios
      .get(`${API}/analytics`)
      .then((r) => setAnalytics(r.data))
      .catch(() => {});
    axios
      .get(`${API}/sources`)
      .then((r) => setSources(r.data))
      .catch(() => {});

    return () => clearInterval(interval);
  }, []);
  async function analyze(e: FormEvent) {
    e.preventDefault();
    setError("");
    setPrediction(null);
    setBusy(true);
    try {
      const r = await axios.post(`${API}/predict`, {
        title: articleTitle,
        text: articleText,
        url: articleUrl || null,
      });
      const result = r.data as Prediction;
      if (
        predictionClass(result.label) === "UNKNOWN" ||
        !Number.isFinite(result.confidence) ||
        !Number.isFinite(result.fake_probability) ||
        !Number.isFinite(result.real_probability)
      ) {
        throw new Error("The API returned an unavailable prediction.");
      }
      setPrediction(result);
    } catch (err: any) {
      setError(
        err?.response?.data?.detail || err?.message ||
          "Unable to analyze. Check that the API is running and the model is trained.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function batch(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setError("");
    setBatchBusy(true);
    try {
      const form = new FormData();
      form.append("file", f);
      const r = await axios.post(`${API}/batch`, form, {
        responseType: "blob",
      });
      const url = URL.createObjectURL(r.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = "misinformation_batch_report.csv";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: any) {
      setError(
        err?.response?.data?.detail ||
          "Batch processing failed. Ensure the model is trained and the CSV has a title/text column.",
      );
    } finally {
      setBatchBusy(false);
      e.target.value = "";
    }
  }
  const nav = [
    { name: "Overview", icon: Layers3 },
    { name: "Analyze article", icon: Search },
    { name: "Misinformation trends", icon: BarChart3 },
    { name: "Source credibility", icon: ShieldCheck },
    { name: "Batch reports", icon: FileText },
  ];
  const total = analytics?.total_articles || 0;
  const fake =
    analytics?.class_distribution.find((x) => x.label === "Fake")?.count || 0;
  const real =
    analytics?.class_distribution.find((x) => x.label === "Real")?.count || 0;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Activity size={19} />
          </div>
          <div>
            <b>
              signal<span>/</span>
            </b>
            <small>INTELLIGENCE PLATFORM</small>
          </div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav>
          {nav.map((n) => (
            <button
              key={n.name}
              onClick={() => setPage(n.name)}
              className={`nav-item ${page === n.name ? "active" : ""}`}
            >
              <n.icon size={17} />
              <span>{n.name}</span>
              {page === n.name && (
                <ChevronRight className="nav-arrow" size={15} />
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="status-card">
            <div className="status-head">
              <span className={`status-dot ${apiReady ? "online" : ""}`}></span>
              <b>{apiReady ? "API connected" : "API unavailable"}</b>
            </div>
            <p>
              {modelReady
                ? "Baseline model loaded"
                : apiReady
                  ? "Model training required"
                  : "Start the FastAPI server to connect."}
            </p>
            {transformerReady && (
              <p style={{ color: "#6ee7b7", marginTop: 4, fontSize: 11 }}>
                DistilBERT loaded ✓
              </p>
            )}
            <div className="status-foot">
              <span>LOCAL ENVIRONMENT</span>
              <span className="mono">v1.0.0</span>
            </div>
          </div>
          <div className="user-chip">
            <div className="avatar">MI</div>
            <div>
              <b>Research workspace</b>
              <small>Data intelligence</small>
            </div>
            <CircleHelp size={17} className="muted" />
          </div>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> <b>{page}</b>
          </div>
          <div className="top-actions">
            <span className="live-pill">
              <span className="status-dot online"></span> LOCAL ANALYSIS
            </span>
            <div className="top-avatar">S</div>
          </div>
        </header>
        <div className="content">
          <div className="page-intro">
            <div>
              <div className="eyebrow">
                <Sparkles size={13} /> MISINFORMATION INTELLIGENCE
              </div>
              <h1>{page === "Overview" ? "See beyond the headline." : page}</h1>
              <p>
                {page === "Overview"
                  ? "A structured workspace for analyzing news patterns, model predictions, and source history."
                  : page === "Analyze article"
                    ? "Evaluate an article with the baseline model and inspect the terms influencing its prediction."
                    : page === "Misinformation trends"
                      ? "Explore patterns across the supplied labeled dataset."
                      : page === "Source credibility"
                        ? "Review publisher history from an explicitly supplied and auditable source registry."
                        : "Run multiple articles through the same baseline and export a reviewable report."}
              </p>
            </div>
            <div className="date-chip">
              <Clock3 size={14} /> RESEARCH CONSOLE{" "}
              <span className="divider-v" /> <span>v1.0</span>
            </div>
          </div>
          {error && (
            <div className="error-banner">
              <AlertTriangle size={17} />
              <span>{error}</span>
              <button onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {page === "Overview" && (
            <>
              <div className="hero-panel">
                <div className="hero-copy">
                  <div className="hero-kicker">
                    <span className="hero-kicker-dot" /> BASELINE MODEL
                    WORKSPACE
                  </div>
                  <h2>
                    Context is everything.
                    <br />
                    <em>Evidence is the next step.</em>
                  </h2>
                  <p>
                    Analyze news text, understand model signals, and explore
                    patterns across the dataset — all in one place.
                  </p>
                  <button
                    className="primary-btn"
                    onClick={() => setPage("Analyze article")}
                  >
                    Analyze an article <ChevronRight size={16} />
                  </button>
                </div>
                <div className="hero-graphic">
                  <div className="orbit orbit-one"></div>
                  <div className="orbit orbit-two"></div>
                  <div className="orbit-core">
                    <Activity size={34} />
                  </div>
                  <div className="float-tag tag-a">
                    <span className="tag-dot orange"></span> NLP SIGNALS
                  </div>
                  <div className="float-tag tag-b">
                    <ShieldCheck size={14} /> SOURCE HISTORY
                  </div>
                  <div className="float-tag tag-c">
                    <span className="pulse-line"></span> EVIDENCE
                  </div>
                  <div className="hero-coordinate">MODEL / 001</div>
                </div>
              </div>
              <div className="metric-grid">
                <Metric
                  label="ARTICLES IN CORPUS"
                  value={total ? fmt(total) : "—"}
                  detail={
                    total ? "Labeled dataset records" : "Dataset not connected"
                  }
                  icon={<FileText size={18} />}
                  accent="blue"
                />
                <Metric
                  label="FAKE-LABELED"
                  value={total ? fmt(fake) : "—"}
                  detail={
                    total
                      ? `${((fake / total) * 100).toFixed(1)}% of corpus`
                      : "Awaiting dataset"
                  }
                  icon={<AlertTriangle size={18} />}
                  accent="orange"
                />
                <Metric
                  label="REAL-LABELED"
                  value={total ? fmt(real) : "—"}
                  detail={
                    total
                      ? `${((real / total) * 100).toFixed(1)}% of corpus`
                      : "Awaiting dataset"
                  }
                  icon={<CheckCircle2 size={18} />}
                  accent="green"
                />
                <Metric
                  label="MODEL STATUS"
                  value={modelReady ? "Ready" : "Pending"}
                  detail={
                    modelReady
                      ? "TF-IDF + Logistic Regression"
                      : "Train baseline to activate"
                  }
                  icon={<Gauge size={18} />}
                  accent="violet"
                />
              </div>
              <div className="dashboard-grid">
                <section className="panel chart-panel">
                  <div className="panel-heading">
                    <div>
                      <div className="panel-overline">CORPUS OVERVIEW</div>
                      <h3>Label distribution</h3>
                    </div>
                    <span className="subtle-badge">TRAINING DATA</span>
                  </div>
                  {analytics?.available ? (
                    <>
                      <div className="chart-legend">
                        <span>
                          <i className="legend-dot blue-dot"></i> Real
                        </span>
                        <span>
                          <i className="legend-dot orange-dot"></i> Fake
                        </span>
                      </div>
                      <div className="donut-wrap">
                        <ResponsiveContainer width="100%" height={205}>
                          <PieChart>
                            <Pie
                              data={analytics.class_distribution}
                              dataKey="count"
                              nameKey="label"
                              innerRadius={62}
                              outerRadius={86}
                              paddingAngle={4}
                              stroke="none"
                            >
                              {analytics.class_distribution.map((d, i) => (
                                <Cell
                                  key={d.label}
                                  fill={
                                    d.label === "Fake" ? "#ff865e" : "#6b9bff"
                                  }
                                />
                              ))}
                            </Pie>
                            <Tooltip
                              contentStyle={{
                                background: "#171d29",
                                border: "1px solid #30394a",
                                borderRadius: 10,
                                color: "#fff",
                              }}
                              formatter={(v: any) => fmt(Number(v))}
                            />
                          </PieChart>
                        </ResponsiveContainer>
                        <div className="donut-center">
                          <b>{fmt(total)}</b>
                          <span>ARTICLES</span>
                        </div>
                      </div>
                    </>
                  ) : (
                    <EmptyState text="Connect Fake.csv and True.csv, then train the model to load dataset analytics." />
                  )}
                  <div className="panel-foot">
                    <span>Source: supplied labeled corpus</span>
                    <span className="muted">
                      Descriptive, not ground truth beyond labels
                    </span>
                  </div>
                </section>
                <section className="panel trend-panel">
                  <div className="panel-heading">
                    <div>
                      <div className="panel-overline">TEMPORAL SIGNALS</div>
                      <h3>Article volume over time</h3>
                    </div>
                    <span className="subtle-badge">MONTHLY</span>
                  </div>
                  {analytics?.monthly_trends?.length ? (
                    <ResponsiveContainer width="100%" height={240}>
                      <AreaChart data={analytics.monthly_trends}>
                        <defs>
                          <linearGradient
                            id="realFill"
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                          >
                            <stop
                              offset="0%"
                              stopColor="#6b9bff"
                              stopOpacity={0.25}
                            />
                            <stop
                              offset="100%"
                              stopColor="#6b9bff"
                              stopOpacity={0}
                            />
                          </linearGradient>
                          <linearGradient
                            id="fakeFill"
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                          >
                            <stop
                              offset="0%"
                              stopColor="#ff865e"
                              stopOpacity={0.2}
                            />
                            <stop
                              offset="100%"
                              stopColor="#ff865e"
                              stopOpacity={0}
                            />
                          </linearGradient>
                        </defs>
                        <CartesianGrid
                          stroke="#252d3a"
                          strokeDasharray="3 4"
                          vertical={false}
                        />
                        <XAxis
                          dataKey="month"
                          tick={{ fill: "#7e8a9e", fontSize: 10 }}
                          tickLine={false}
                          axisLine={false}
                          minTickGap={22}
                        />
                        <YAxis
                          tick={{ fill: "#7e8a9e", fontSize: 10 }}
                          tickLine={false}
                          axisLine={false}
                          width={38}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#171d29",
                            border: "1px solid #30394a",
                            borderRadius: 10,
                            color: "#fff",
                          }}
                        />
                        <Area
                          type="monotone"
                          dataKey="real"
                          name="Real-labeled"
                          stroke="#6b9bff"
                          fill="url(#realFill)"
                          strokeWidth={2}
                        />
                        <Area
                          type="monotone"
                          dataKey="fake"
                          name="Fake-labeled"
                          stroke="#ff865e"
                          fill="url(#fakeFill)"
                          strokeWidth={2}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <EmptyState text="Monthly trends appear when the dataset includes parseable publication dates." />
                  )}
                  <div className="panel-foot">
                    <span>Based on dates in the dataset</span>
                    <span className="muted">Date quality may vary</span>
                  </div>
                </section>
              </div>
              <div className="lower-grid">
                <section className="panel subject-panel">
                  <div className="panel-heading">
                    <div>
                      <div className="panel-overline">TOPIC BREAKDOWN</div>
                      <h3>Subjects in the corpus</h3>
                    </div>
                    <button
                      className="text-btn"
                      onClick={() => setPage("Misinformation trends")}
                    >
                      Explore <ChevronRight size={14} />
                    </button>
                  </div>
                  {analytics?.subjects?.length ? (
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart
                        data={analytics.subjects.slice(0, 6)}
                        layout="vertical"
                        margin={{ left: 4, right: 14 }}
                      >
                        <CartesianGrid
                          stroke="#252d3a"
                          strokeDasharray="3 4"
                          horizontal={false}
                        />
                        <XAxis
                          type="number"
                          tick={{ fill: "#7e8a9e", fontSize: 10 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          type="category"
                          dataKey="subject"
                          width={100}
                          tick={{ fill: "#a9b3c3", fontSize: 11 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#171d29",
                            border: "1px solid #30394a",
                            borderRadius: 10,
                            color: "#fff",
                          }}
                        />
                        <Bar
                          dataKey="real"
                          name="Real-labeled"
                          stackId="a"
                          fill="#6b9bff"
                          radius={[0, 0, 0, 0]}
                        />
                        <Bar
                          dataKey="fake"
                          name="Fake-labeled"
                          stackId="a"
                          fill="#ff865e"
                          radius={[0, 4, 4, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <EmptyState text="Subject distribution will appear after the data files are added." />
                  )}
                </section>
                <section className="panel quick-panel">
                  <div className="panel-heading">
                    <div>
                      <div className="panel-overline">QUICK ACTIONS</div>
                      <h3>Continue your analysis</h3>
                    </div>
                  </div>
                  <QuickAction
                    icon={<Search size={17} />}
                    title="Analyze a news article"
                    desc="Run a single article through the baseline."
                    onClick={() => setPage("Analyze article")}
                  />
                  <QuickAction
                    icon={<BarChart3 size={17} />}
                    title="Explore dataset trends"
                    desc="Compare topics and language patterns."
                    onClick={() => setPage("Misinformation trends")}
                  />
                  <QuickAction
                    icon={<Upload size={17} />}
                    title="Batch prediction"
                    desc="Upload a CSV and export model outputs."
                    onClick={() => setPage("Batch reports")}
                  />
                </section>
              </div>
              <div className="notice">
                <ShieldCheck size={17} />
                <div>
                  <b>Research tool, not a truth oracle</b>
                  <p>
                    Model predictions are learned patterns from a labeled
                    dataset. Related articles are not necessarily verified
                    evidence, and historical source scores require independently
                    sourced publisher records.
                  </p>
                </div>
              </div>
            </>
          )}
          {page === "Analyze article" && (
            <div className="analysis-layout">
              <section className="panel input-panel">
                <div className="panel-heading">
                  <div>
                    <div className="panel-overline">INPUT WORKSPACE</div>
                    <h3>Submit article for analysis</h3>
                  </div>
                  <span className="subtle-badge">TEXT CLASSIFICATION</span>
                </div>
                <form onSubmit={analyze}>
                  <label>
                    ARTICLE URL <span>OPTIONAL</span>
                  </label>
                  <div className="input-with-icon">
                    <Globe2 size={16} />
                    <input
                      value={articleUrl}
                      onChange={(e) => setArticleUrl(e.target.value)}
                      placeholder="https://publisher.com/article..."
                    />
                  </div>
                  <div className="field-note">
                    If article text is empty, the server attempts to extract
                    readable page content. Some sites block automated
                    extraction.
                  </div>
                  <div className="field-divider">
                    <span>OR ENTER TEXT DIRECTLY</span>
                  </div>
                  <label>HEADLINE</label>
                  <input
                    className="plain-input"
                    value={articleTitle}
                    onChange={(e) => setArticleTitle(e.target.value)}
                    placeholder="Paste the article headline"
                  />
                  <label>
                    ARTICLE BODY <span>REQUIRED IF NO URL</span>
                  </label>
                  <textarea
                    value={articleText}
                    onChange={(e) => setArticleText(e.target.value)}
                    placeholder="Paste the full article or claim here..."
                    rows={11}
                  />
                  <div className="textarea-foot">
                    <span>
                      {articleText.length.toLocaleString()} characters
                    </span>
                    <button
                      type="button"
                      className="text-btn"
                      onClick={() => {
                        setArticleTitle("");
                        setArticleText("");
                        setArticleUrl("");
                        setPrediction(null);
                      }}
                    >
                      Clear input
                    </button>
                  </div>
                  <button
                    disabled={busy || !modelReady}
                    className="primary-btn full-btn"
                  >
                    {busy ? (
                      <>
                        <LoaderCircle className="spin" size={16} /> Analyzing
                        article...
                      </>
                    ) : (
                      <>
                        <Zap size={16} /> Analyze article{" "}
                        <ChevronRight size={16} />
                      </>
                    )}
                  </button>
                  {!modelReady && (
                    <p className="inline-warning">
                      <AlertTriangle size={14} /> Train the baseline model
                      before running predictions.
                    </p>
                  )}
                </form>
              </section>
              <section className="results-column">
                {prediction ? (
                  <>
                    <div
                      className={`verdict-card ${predictionClass(prediction.label) === "FAKE" ? "verdict-fake" : "verdict-real"}`}
                    >
                      <div className="verdict-top">
                        <span className="verdict-icon">
                          {predictionClass(prediction.label) === "FAKE" ? (
                            <AlertTriangle size={20} />
                          ) : (
                            <CheckCircle2 size={20} />
                          )}
                        </span>
                        <span className="subtle-badge">MODEL ESTIMATE</span>
                      </div>
                      <div className="verdict-label">
                        {prediction.label.replace("-", " ")}
                      </div>
                      <p>Predicted class based on learned text patterns</p>
                      <div className="confidence-row">
                        <span>Model confidence</span>
                        <b>{(prediction.confidence * 100).toFixed(1)}%</b>
                      </div>
                      <div className="confidence-track">
                        <span
                          style={{ width: `${prediction.confidence * 100}%` }}
                        />
                      </div>
                      <div className="probability-grid">
                        <div>
                          <span>Fake probability</span>
                          <b>
                            {(prediction.fake_probability * 100).toFixed(1)}%
                          </b>
                        </div>
                        <div>
                          <span>Real probability</span>
                          <b>
                            {(prediction.real_probability * 100).toFixed(1)}%
                          </b>
                        </div>
                      </div>
                    </div>
                    {predictionClass(prediction.label) === "FAKE" && (
                      <div className="fake-alert" role="alert" aria-live="assertive">
                        <div className="fake-alert-message">
                          🚨 FAKE NEWS ALERT! This article has been identified as potentially fake by our AI model. Please verify the information before sharing.
                        </div>
                        <span>Model confidence: {(prediction.confidence * 100).toFixed(1)}%</span>
                      </div>
                    )}
                    {/* ── Dual-model comparison card ── */}
                    <section className="panel" style={{ marginBottom: 18 }}>
                      <div className="panel-heading">
                        <div>
                          <div className="panel-overline">MODEL COMPARISON</div>
                          <h3>TF-IDF + LR vs DistilBERT</h3>
                        </div>
                        <span className="subtle-badge">DUAL MODEL</span>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 8 }}>
                        {/* Baseline LR card */}
                        <div style={{
                          background: predictionClass(prediction.label) === "FAKE" ? "rgba(255,120,60,0.08)" : "rgba(107,155,255,0.08)",
                          border: `1.5px solid ${predictionClass(prediction.label) === "FAKE" ? "#ff7c3c55" : "#6b9bff55"}`,
                          borderRadius: 12, padding: "16px 18px",
                        }}>
                          <div style={{ fontSize: 10, letterSpacing: "0.08em", color: "#7e8a9e", marginBottom: 6 }}>TF-IDF + LOGISTIC REGRESSION</div>
                          <div style={{
                            fontSize: 22, fontWeight: 700,
                            color: predictionClass(prediction.label) === "FAKE" ? "#ff865e" : "#6b9bff",
                          }}>
                            {prediction.label.replace("-", " ")}
                          </div>
                          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 4 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                              <span style={{ color: "#7e8a9e" }}>Confidence</span>
                              <b>{(prediction.confidence * 100).toFixed(1)}%</b>
                            </div>
                            <div style={{ height: 5, borderRadius: 3, background: "#1e2636", overflow: "hidden" }}>
                              <div style={{
                                height: "100%", width: `${prediction.confidence * 100}%`,
                                background: predictionClass(prediction.label) === "FAKE" ? "#ff865e" : "#6b9bff",
                                borderRadius: 3, transition: "width 0.6s ease"
                              }} />
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#7e8a9e", marginTop: 2 }}>
                              <span>Fake {(prediction.fake_probability * 100).toFixed(1)}%</span>
                              <span>Real {(prediction.real_probability * 100).toFixed(1)}%</span>
                            </div>
                          </div>
                        </div>
                        {/* DistilBERT card */}
                        {prediction.transformer_prediction ? (
                          prediction.transformer_prediction.error ? (
                            <div style={{
                              background: "rgba(126,138,158,0.07)", border: "1.5px solid #30394a",
                              borderRadius: 12, padding: "16px 18px", display: "flex", flexDirection: "column", justifyContent: "center"
                            }}>
                              <div style={{ fontSize: 10, letterSpacing: "0.08em", color: "#7e8a9e", marginBottom: 6 }}>DISTILBERT</div>
                              <div style={{ color: "#7e8a9e", fontSize: 13 }}>Unavailable</div>
                              <div style={{ fontSize: 11, color: "#5a6478", marginTop: 4 }}>{prediction.transformer_prediction.error}</div>
                            </div>
                          ) : (
                            <div style={{
                              background: predictionClass(prediction.transformer_prediction.label) === "FAKE" ? "rgba(255,120,60,0.08)" : "rgba(107,155,255,0.08)",
                              border: `1.5px solid ${predictionClass(prediction.transformer_prediction.label) === "FAKE" ? "#ff7c3c55" : "#6b9bff55"}`,
                              borderRadius: 12, padding: "16px 18px",
                            }}>
                              <div style={{ fontSize: 10, letterSpacing: "0.08em", color: "#7e8a9e", marginBottom: 6 }}>DISTILBERT (FINE-TUNED)</div>
                              <div style={{
                                fontSize: 22, fontWeight: 700,
                                color: predictionClass(prediction.transformer_prediction.label) === "FAKE" ? "#ff865e" : "#6b9bff",
                              }}>
                                {prediction.transformer_prediction.label.replace("-", " ")}
                              </div>
                              <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 4 }}>
                                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                                  <span style={{ color: "#7e8a9e" }}>Confidence</span>
                                  <b>{prediction.transformer_prediction.confidence !== null ? `${((prediction.transformer_prediction.confidence ?? 0) * 100).toFixed(1)}%` : "—"}</b>
                                </div>
                                <div style={{ height: 5, borderRadius: 3, background: "#1e2636", overflow: "hidden" }}>
                                  <div style={{
                                    height: "100%",
                                    width: `${((prediction.transformer_prediction.confidence ?? 0) * 100)}%`,
                                    background: predictionClass(prediction.transformer_prediction.label) === "FAKE" ? "#ff865e" : "#6b9bff",
                                    borderRadius: 3, transition: "width 0.6s ease"
                                  }} />
                                </div>
                                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#7e8a9e", marginTop: 2 }}>
                                  <span>Fake {prediction.transformer_prediction.fake_probability !== null ? `${((prediction.transformer_prediction.fake_probability ?? 0) * 100).toFixed(1)}%` : "—"}</span>
                                  <span>Real {prediction.transformer_prediction.real_probability !== null ? `${((prediction.transformer_prediction.real_probability ?? 0) * 100).toFixed(1)}%` : "—"}</span>
                                </div>
                              </div>
                              {predictionClass(prediction.label) !== predictionClass(prediction.transformer_prediction.label) && (
                                <div style={{
                                  marginTop: 10, padding: "6px 10px", borderRadius: 6,
                                  background: "rgba(251,191,36,0.1)", border: "1px solid rgba(251,191,36,0.3)",
                                  fontSize: 11, color: "#fbbf24", display: "flex", gap: 6, alignItems: "center"
                                }}>
                                  ⚠ Models disagree — review carefully.
                                </div>
                              )}
                            </div>
                          )
                        ) : (
                          <div style={{
                            background: "rgba(126,138,158,0.07)", border: "1.5px solid #30394a",
                            borderRadius: 12, padding: "16px 18px", display: "flex", alignItems: "center", justifyContent: "center"
                          }}>
                            <span style={{ color: "#5a6478", fontSize: 13 }}>DistilBERT not available</span>
                          </div>
                        )}
                      </div>
                      <p style={{ fontSize: 11, color: "#5a6478", marginTop: 10 }}>
                        Predictions are independent model estimates. Agreement increases confidence; disagreement warrants deeper review.
                      </p>
                    </section>
                    <section className="panel">
                      <div className="panel-heading">
                        <div>
                          <div className="panel-overline">
                            MODEL INTERPRETATION
                          </div>
                          <h3>Influential terms</h3>
                        </div>
                        <span className="subtle-badge">MODEL CONTRIBUTIONS</span>
                      </div>
                      <p className="panel-description">
                        Contributions are calculated from this prediction's TF-IDF values and fitted Logistic Regression coefficients.
                      </p>
                      {(prediction.explanation ?? []).length > 0 && (
                        <>
                          <div className="explanation-legend" aria-label="Contribution highlight legend">
                            <span><i className="legend-square fake-legend" /> Pushes toward FAKE</span>
                            <span><i className="legend-square real-legend" /> Pushes toward REAL</span>
                          </div>
                          <HighlightedArticle
                            text={prediction.analyzed_text || `${prediction.title || articleTitle} ${articleText}`.trim()}
                            contributions={prediction.explanation ?? []}
                          />
                        </>
                      )}
                      {(prediction.explanation ?? []).length ? (
                        (prediction.explanation ?? []).map((t) => (
                          <div className="term-row" key={t.term}>
                            <span
                              className={`term-chip ${t.direction === "FAKE" ? "fake-chip" : "real-chip"}`}
                            >
                              {t.term}
                            </span>
                            <div className="term-bar">
                              <span
                                className={
                                  t.direction === "FAKE"
                                    ? "fake-bar"
                                    : "real-bar"
                                }
                                style={{
                                  width: `${Math.min(100, Math.abs(t.contribution) * 100)}%`,
                                }}
                              />
                            </div>
                            <span
                              className={`term-direction ${t.direction === "FAKE" ? "orange-text" : "blue-text"}`}
                            >
                              {t.direction}
                            </span>
                          </div>
                        ))
                      ) : (
                        <p className="explanation-unavailable">
                          {prediction.explanation_message || "Explanation unavailable for this model output."}
                        </p>
                      )}
                    </section>
                    <SourceCredibilityCard source={prediction.source_credibility} />
                    <section className="panel">
                      <div className="panel-heading">
                        <div>
                          <div className="panel-overline">
                            RELATED CORPUS ITEMS
                          </div>
                          <h3>Evidence discovery</h3>
                        </div>
                        <span className="subtle-badge">TOP 3 MATCHES</span>
                      </div>
                      <div className="evidence-disclaimer">
                        <AlertTriangle size={15} /> Similarity is not
                        verification. These are related labeled dataset
                        articles, not independently fact-checked evidence.
                      </div>
                      {(prediction.evidence ?? []).length ? (
                        (prediction.evidence ?? []).map((ev, i) => (
                          <div
                            className="evidence-item"
                            key={`${ev.title}-${i}`}
                          >
                            <div className="evidence-index">0{i + 1}</div>
                            <div className="evidence-body">
                              <div className="evidence-meta">
                                <span
                                  className={`dataset-tag ${ev.dataset_label === "FAKE" ? "orange-tag" : "blue-tag"}`}
                                >
                                  {ev.dataset_label}-LABELED
                                </span>
                                <span>
                                  SIMILARITY {(ev.similarity * 100).toFixed(1)}%
                                </span>
                                {ev.date && <span>{ev.date}</span>}
                              </div>
                              <b>{ev.title}</b>
                              <p>
                                {ev.snippet}
                                {ev.snippet.length >= 360 ? "…" : ""}
                              </p>
                              <span className="source-note">
                                {ev.source || "Local training corpus"} · source not independently verified
                              </span>
                            </div>
                          </div>
                        ))
                      ) : (
                        <EmptyState text="No sufficiently similar corpus articles found." />
                      )}
                    </section>
                    <div className="notice">
                      <CircleHelp size={17} />
                      <div>
                        <b>Interpretation note</b>
                        <p>
                          {prediction.disclaimer} The model does not browse
                          fact-checking databases or establish article truth.
                        </p>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="empty-results">
                    <div className="empty-orbit">
                      <Search size={30} />
                    </div>
                    <h3>Your analysis will appear here</h3>
                    <p>
                      Submit a headline and article text to see the model
                      estimate, term-level signals, and related corpus items.
                    </p>
                    <div className="empty-checks">
                      <span>
                        <CheckCircle2 size={15} /> Probability estimates
                      </span>
                      <span>
                        <CheckCircle2 size={15} /> Influential terms
                      </span>
                      <span>
                        <CheckCircle2 size={15} /> Related articles
                      </span>
                    </div>
                  </div>
                )}
              </section>
            </div>
          )}
          {page === "Misinformation trends" && (
            <>
              <div className="metric-grid">
                <Metric
                  label="CORPUS SIZE"
                  value={total ? fmt(total) : "—"}
                  detail="All loaded labeled records"
                  icon={<Layers3 size={18} />}
                  accent="blue"
                />
                <Metric
                  label="SUBJECT CATEGORIES"
                  value={
                    analytics?.subjects?.length
                      ? String(analytics.subjects.length)
                      : "—"
                  }
                  detail="Top categories shown below"
                  icon={<BarChart3 size={18} />}
                  accent="violet"
                />
                <Metric
                  label="DATE COVERAGE"
                  value={
                    analytics?.monthly_trends?.length
                      ? String(analytics.monthly_trends.length)
                      : "—"
                  }
                  detail="Months with parseable dates"
                  icon={<Clock3 size={18} />}
                  accent="green"
                />
                <Metric
                  label="LINGUISTIC FEATURES"
                  value={analytics?.linguistic_comparison?.length ? "3" : "—"}
                  detail="Word count, exclamations, capitals"
                  icon={<Activity size={18} />}
                  accent="orange"
                />
              </div>
              <div className="dashboard-grid">
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <div className="panel-overline">TIME SERIES</div>
                      <h3>Fake vs real-labeled articles</h3>
                    </div>
                  </div>
                  {analytics?.monthly_trends?.length ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={analytics.monthly_trends}>
                        <CartesianGrid
                          stroke="#252d3a"
                          strokeDasharray="3 4"
                          vertical={false}
                        />
                        <XAxis
                          dataKey="month"
                          tick={{ fill: "#7e8a9e", fontSize: 10 }}
                          tickLine={false}
                          axisLine={false}
                        />
                        <YAxis
                          tick={{ fill: "#7e8a9e", fontSize: 10 }}
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#171d29",
                            border: "1px solid #30394a",
                            borderRadius: 10,
                            color: "#fff",
                          }}
                        />
                        <Area
                          dataKey="real"
                          name="Real-labeled"
                          stroke="#6b9bff"
                          fill="#6b9bff"
                          fillOpacity={0.1}
                        />
                        <Area
                          dataKey="fake"
                          name="Fake-labeled"
                          stroke="#ff865e"
                          fill="#ff865e"
                          fillOpacity={0.1}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <EmptyState text="No parseable date series available." />
                  )}
                </section>
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <div className="panel-overline">
                        LINGUISTIC COMPARISON
                      </div>
                      <h3>Average article signals</h3>
                    </div>
                  </div>
                  {analytics?.linguistic_comparison?.length ? (
                    <div className="ling-list">
                      {analytics.linguistic_comparison.map((d) => (
                        <div className="ling-block" key={d.label}>
                          <div className="ling-title">
                            <b>{d.label}-labeled</b>
                            <span
                              className={`dataset-tag ${d.label === "Fake" ? "orange-tag" : "blue-tag"}`}
                            >
                              {d.label.toUpperCase()}
                            </span>
                          </div>
                          <div className="ling-values">
                            <div>
                              <b>{fmt(Math.round(d.avg_words))}</b>
                              <span>Avg. words</span>
                            </div>
                            <div>
                              <b>{d.avg_exclamations.toFixed(2)}</b>
                              <span>Exclamations</span>
                            </div>
                            <div>
                              <b>{d.avg_capital_ratio.toFixed(1)}%</b>
                              <span>Capital ratio</span>
                            </div>
                            <div>
                              <b>{d.avg_flesch_reading_ease == null ? "—" : d.avg_flesch_reading_ease.toFixed(1)}</b>
                              <span>Flesch reading ease</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <EmptyState text="Linguistic comparison appears after loading data." />
                  )}
                  <div className="field-note">
                    Descriptive comparisons do not establish causation. Flesch Reading Ease is a readability estimate, not a writing-quality score.
                  </div>
                </section>
              </div>
              {analytics?.topics?.length ? (
                <section className="panel full-panel">
                  <div className="panel-heading">
                    <div>
                      <div className="panel-overline">TOPIC-LEVEL LABEL MIX</div>
                      <h3>Fake-labeled share by topic</h3>
                    </div>
                  </div>
                  <div className="topic-analysis-list">
                    {analytics.topics.map((topic) => (
                      <div className="topic-analysis-row" key={topic.category}>
                        <div className="topic-analysis-label">
                          <b>{topic.category}</b>
                          <span>{fmt(topic.total)} articles · {fmt(topic.fake)} fake-labeled</span>
                        </div>
                        <div className="topic-analysis-track" aria-label={`${topic.category}: ${((topic.fake_share ?? 0) * 100).toFixed(1)}% fake-labeled`}>
                          <span style={{ width: `${(topic.fake_share ?? 0) * 100}%` }} />
                        </div>
                        <b className="topic-analysis-value">{topic.fake_share == null ? "—" : `${(topic.fake_share * 100).toFixed(1)}%`}</b>
                      </div>
                    ))}
                  </div>
                </section>
              ) : null}
              <section className="panel full-panel">
                <div className="panel-heading">
                  <div>
                    <div className="panel-overline">EMOTIONAL-LANGUAGE COMPARISON</div>
                    <h3>Corpus-level statistical test</h3>
                  </div>
                  <span className="subtle-badge">MANN–WHITNEY U</span>
                </div>
                {analytics?.emotional_language_test ? (
                  <>
                    <p className="panel-description">{analytics.emotional_language_test.metric}</p>
                    <div className="statistics-grid">
                      <div><span>REAL N</span><b>{fmt(analytics.emotional_language_test.real_sample_size)}</b><small>Median {analytics.emotional_language_test.real_median?.toFixed(3) ?? "—"}</small></div>
                      <div><span>FAKE N</span><b>{fmt(analytics.emotional_language_test.fake_sample_size)}</b><small>Median {analytics.emotional_language_test.fake_median?.toFixed(3) ?? "—"}</small></div>
                      <div><span>P-VALUE</span><b>{analytics.emotional_language_test.p_value == null ? "—" : analytics.emotional_language_test.p_value.toExponential(2)}</b><small>Two-sided</small></div>
                      <div><span>RANK-BISERIAL EFFECT</span><b>{analytics.emotional_language_test.rank_biserial_effect_size?.toFixed(3) ?? "—"}</b><small>{analytics.emotional_language_test.effect_magnitude}</small></div>
                    </div>
                    <p className="statistics-interpretation">{analytics.emotional_language_test.interpretation}</p>
                    <p className="field-note">{analytics.emotional_language_test.method}. {analytics.emotional_language_test.assumptions} Statistical significance is not practical importance, and the score is not a truth indicator.</p>
                  </>
                ) : (
                  <EmptyState text="A statistical comparison is unavailable because the labeled corpus is incomplete." />
                )}
                {analytics?.available && (
                  <p className="field-note">
                    Exact duplicate rows removed: {fmt(analytics.duplicate_rows_removed ?? 0)}. Conflicting-label duplicate rows excluded: {fmt(analytics.conflicting_duplicate_rows_excluded ?? 0)}.
                  </p>
                )}
              </section>
              <section className="panel full-panel">
                <div className="panel-heading">
                  <div>
                    <div className="panel-overline">SUBJECT DISTRIBUTION</div>
                    <h3>Article counts by subject</h3>
                  </div>
                </div>
                {analytics?.subjects?.length ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={analytics.subjects}
                      margin={{ left: 8, right: 16, bottom: 30 }}
                    >
                      <CartesianGrid
                        stroke="#252d3a"
                        strokeDasharray="3 4"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="subject"
                        angle={-20}
                        textAnchor="end"
                        interval={0}
                        tick={{ fill: "#a9b3c3", fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        tick={{ fill: "#7e8a9e", fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#171d29",
                          border: "1px solid #30394a",
                          borderRadius: 10,
                          color: "#fff",
                        }}
                      />
                      <Bar
                        dataKey="real"
                        name="Real-labeled"
                        fill="#6b9bff"
                        radius={[3, 3, 0, 0]}
                      />
                      <Bar
                        dataKey="fake"
                        name="Fake-labeled"
                        fill="#ff865e"
                        radius={[3, 3, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyState text="No subject data available." />
                )}
              </section>
            </>
          )}
          {page === "Source credibility" && (
            <section className="panel full-panel">
              <div className="panel-heading">
                <div>
                  <div className="panel-overline">PUBLISHER HISTORY</div>
                  <h3>Source credibility leaderboard</h3>
                </div>
                <span className="subtle-badge">AUDITABLE INPUT REQUIRED</span>
              </div>
              <div className="source-explainer">
                <div className="source-explainer-icon">
                  <ShieldCheck size={22} />
                </div>
                <div>
                  <b>Credibility must be earned from evidence.</b>
                  <p>
                    The supplied news dataset has no publisher identity field.
                    To calculate a leaderboard, provide
                    `backend/data/publisher_history.csv` with `publisher`,
                    `real_articles`, and `fake_articles` columns. Counts should
                    come from a documented, consistent labeling process.
                  </p>
                </div>
              </div>
              {sources?.available ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>RANK</th>
                        <th>PUBLISHER</th>
                        <th>HISTORICAL SCORE</th>
                        <th>REAL</th>
                        <th>FAKE</th>
                        <th>ARTICLES</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sources.sources.map((s, i) => (
                        <tr key={s.publisher}>
                          <td>
                            <span className="rank-num">
                              {String(i + 1).padStart(2, "0")}
                            </span>
                          </td>
                          <td>
                                <b>{s.publisher}</b>
                                {s.domain && <small className="publisher-domain">{s.domain}</small>}
                          </td>
                          <td>
                            <div className="score-cell">
                              <div className="score-track">
                                <span
                                  style={{ width: `${s.credibility_score}%` }}
                                />
                              </div>
                              <b>{s.credibility_score.toFixed(1)}</b>
                            </div>
                          </td>
                          <td>{fmt(s.real_articles)}</td>
                          <td>{fmt(s.fake_articles)}</td>
                          <td>{fmt(s.total_articles)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty-source">
                  <ShieldCheck size={28} />
                  <h3>Publisher registry not configured</h3>
                  <p>
                    {sources?.message ||
                      "Connect the backend API to check whether a publisher history file is available."}
                  </p>
                  <div className="code-sample">
                    publisher,real_articles,fake_articles
                    <br />
                    Publisher name,120,8
                  </div>
                  <span className="field-note">
                    Example values are illustrative only. Replace them with
                    verified records before use.
                  </span>
                </div>
              )}
              <div className="notice">
                <AlertTriangle size={17} />
                <div>
                  <b>Methodology</b>
                  <p>
                    Scores use ((REAL + 2) / (TOTAL + 4)) × 100 and require at least {sources?.minimum_sample_size ?? 30} records. Registry provenance is not independently verified. This is a historical indicator, not a guarantee of an individual article's accuracy.
                  </p>
                </div>
              </div>
            </section>
          )}
          {page === "Batch reports" && (
            <section className="panel batch-panel">
              <div className="panel-heading">
                <div>
                  <div className="panel-overline">BULK CLASSIFICATION</div>
                  <h3>Batch analysis & reporting</h3>
                </div>
                <span className="subtle-badge">CSV INPUT / CSV OUTPUT</span>
              </div>
              <div className="upload-zone">
                <div className="upload-icon">
                  <Upload size={24} />
                </div>
                <h3>Upload your article dataset</h3>
                <p>
                  Run the baseline model on multiple headlines or article texts,
                  then download a CSV report with class predictions and
                  probability estimates.
                </p>
                <label
                  className={`upload-btn ${batchBusy || !modelReady ? "disabled" : ""}`}
                >
                  <input
                    type="file"
                    accept=".csv,text/csv"
                    onChange={batch}
                    disabled={batchBusy || !modelReady}
                  />
                  {batchBusy ? (
                    <>
                      <LoaderCircle size={16} className="spin" /> Processing
                      CSV...
                    </>
                  ) : (
                    <>
                      <Upload size={16} /> Choose CSV file
                    </>
                  )}
                </label>
                <span className="field-note">
                  Maximum recommended: 5,000 rows per upload for local
                  development.
                </span>
              </div>
              <div className="batch-guides">
                <div>
                  <div className="guide-num">01</div>
                  <b>Prepare your CSV</b>
                  <p>
                    Use a `title` or `headline` column and/or a `text`,
                    `article`, or `content` column.
                  </p>
                </div>
                <div>
                  <div className="guide-num">02</div>
                  <b>Run predictions</b>
                  <p>
                    Each row receives a predicted label plus fake and real
                    probability estimates.
                  </p>
                </div>
                <div>
                  <div className="guide-num">03</div>
                  <b>Review the report</b>
                  <p>
                    Download the generated CSV and manually review high-impact
                    or uncertain items.
                  </p>
                </div>
              </div>
              {!modelReady && (
                <div className="inline-warning">
                  <AlertTriangle size={14} /> The model must be trained before
                  batch processing is available.
                </div>
              )}
              <div className="notice">
                <CircleHelp size={17} />
                <div>
                  <b>CSV privacy</b>
                  <p>
                    In this local development setup, the uploaded file is
                    processed by your local API. Do not deploy this endpoint
                    publicly without authentication, upload limits, and privacy
                    controls.
                  </p>
                </div>
              </div>
            </section>
          )}
          <footer className="footer">
            <span> SIGNAL / MISINFORMATION INTELLIGENCE</span>
            <span>Built for research · Model outputs require human review</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
function Metric({
  label,
  value,
  detail,
  icon,
  accent,
}: {
  label: string;
  value: string;
  detail: string;
  icon: any;
  accent: string;
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{label}</span>
        <div className={`metric-icon ${accent}`}>{icon}</div>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </div>
  );
}
function EmptyState({ text }: { text: string }) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">
        <Activity size={19} />
      </div>
      <p>{text}</p>
    </div>
  );
}
function QuickAction({
  icon,
  title,
  desc,
  onClick,
}: {
  icon: any;
  title: string;
  desc: string;
  onClick: () => void;
}) {
  return (
    <button className="quick-action" onClick={onClick}>
      <span className="quick-icon">{icon}</span>
      <span className="quick-copy">
        <b>{title}</b>
        <small>{desc}</small>
      </span>
      <ChevronRight size={16} />
    </button>
  );
}

function SourceCredibilityCard({ source }: { source?: SourceCredibility }) {
  return (
    <section className="panel source-score-panel" aria-labelledby="source-score-title">
      <div className="panel-heading">
        <div>
          <div className="panel-overline">PUBLISHER HISTORY</div>
          <h3 id="source-score-title">Source credibility</h3>
        </div>
        <ShieldCheck size={18} aria-hidden="true" />
      </div>
      {source?.available ? (
        <>
          <div className="source-score-topline">
            <div>
              <b>{source.publisher}</b>
              <span>{source.domain}</span>
            </div>
            <strong>{source.credibility_score?.toFixed(1)}<small>/100</small></strong>
          </div>
          <div className="source-counts" aria-label="Historical article counts">
            <span>REAL <b>{fmt(source.real_articles ?? 0)}</b></span>
            <span>FAKE <b>{fmt(source.fake_articles ?? 0)}</b></span>
            <span>TOTAL <b>{fmt(source.total_articles ?? 0)}</b></span>
          </div>
          <p className="source-method">
            {source.method || "Beta(2,2)-smoothed historical REAL-label share"}: (REAL + 2) / (TOTAL + 4) × 100.
            {source.minimum_sample_size ? ` Minimum history: ${source.minimum_sample_size} records.` : ""}
          </p>
        </>
      ) : (
        <>
          <p className="source-unavailable">Credibility score unavailable — insufficient source data.</p>
          {source?.domain && <p className="source-domain">Identified domain: {source.domain}</p>}
          {source?.publisher && (
            <div className="source-counts">
              <span>REAL <b>{fmt(source.real_articles ?? 0)}</b></span>
              <span>FAKE <b>{fmt(source.fake_articles ?? 0)}</b></span>
              <span>TOTAL <b>{fmt(source.total_articles ?? 0)}</b></span>
            </div>
          )}
          {source?.minimum_sample_size && (
            <p className="source-method">Required minimum history: {source.minimum_sample_size} records.</p>
          )}
          <span className="field-note">{source?.message || "No recognized source history is available for this article."}</span>
        </>
      )}
      <p className="source-independence">Historical labels are not independently verified here and never change the model prediction.</p>
    </section>
  );
}
