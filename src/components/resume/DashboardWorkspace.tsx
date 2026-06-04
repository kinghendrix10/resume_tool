"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ParseItem } from "@/lib/resume/api-types";
import type { ResumePayload } from "@/lib/resume/schema";
import {
  educationHeatmapRows,
  experienceTimelineRows,
  keywordDensityRows,
  projectBubbleRows,
  skillsChartRows,
  wordFrequencyFromResumes,
  type TimelineRow,
} from "@/lib/resume/chart-data";
import { radarDataset } from "@/lib/resume/radar-data";
import { fitScoresForChart, stackedScoresForChart } from "@/lib/resume/scoring";
import { normalizeResumeData, toCsv } from "@/lib/resume/normalization";
import { EducationMiniFlow } from "./EducationMiniFlow";

const REQ_KEY = "resume_tool_requisitions_v1";

type Requisition = { id: string; name: string; jd: string; keywords: string };

function loadReqs(): Requisition[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(REQ_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as Requisition[];
  } catch {
    return [];
  }
}

function saveReqs(reqs: Requisition[]) {
  localStorage.setItem(REQ_KEY, JSON.stringify(reqs));
}

const COLORS = ["#0d9488", "#2563eb", "#7c3aed", "#ea580c", "#0891b2"];

/** Keep in sync with `MAX_FILES` in `@/lib/resume/extract-text`. */
const PARSE_MAX_FILES = 12;

function fileDedupeKey(f: File) {
  return `${f.name}\0${f.size}\0${f.lastModified}`;
}

function mergePickedFiles(existing: File[], picked: File[]) {
  const seen = new Set(existing.map(fileDedupeKey));
  const next = [...existing];
  for (const f of picked) {
    const k = fileDedupeKey(f);
    if (seen.has(k)) continue;
    seen.add(k);
    next.push(f);
    if (next.length >= PARSE_MAX_FILES) break;
  }
  return next;
}

export function DashboardWorkspace() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<ParseItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [jobDescription, setJobDescription] = useState("");
  const [keywordsLine, setKeywordsLine] = useState("Python, SQL, machine learning, AWS");
  const [requisitions, setRequisitions] = useState<Requisition[]>([]);
  const [reqName, setReqName] = useState("");
  const [compareIdx, setCompareIdx] = useState<number[]>([]);
  const [activeTab, setActiveTab] = useState<"overview" | "charts" | "compare">("overview");

  useEffect(() => {
    setRequisitions(loadReqs());
  }, []);

  const okPayloads = useMemo(() => {
    const out: { index: number; filename: string; data: ResumePayload }[] = [];
    results.forEach((r, i) => {
      if (r.ok) out.push({ index: i, filename: r.filename, data: r.data });
    });
    return out;
  }, [results]);

  const resumes = useMemo(() => okPayloads.map((x) => x.data), [okPayloads]);
  const keywords = useMemo(
    () => keywordsLine.split(",").map((k) => k.trim()).filter(Boolean),
    [keywordsLine]
  );

  const hydrateReqs = useCallback(() => setRequisitions(loadReqs()), []);

  const parseFiles = async () => {
    if (!files.length) return;
    setLoading(true);
    try {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      const res = await fetch("/api/parse", { method: "POST", body: fd });
      const json = await res.json();
      if (!res.ok) {
        setResults([{ filename: "_", ok: false, error: json.error || "Request failed" }]);
        return;
      }
      setResults(json.results as ParseItem[]);
    } finally {
      setLoading(false);
    }
  };

  const saveRequisition = () => {
    if (!reqName.trim()) return;
    const next: Requisition = {
      id: crypto.randomUUID(),
      name: reqName.trim(),
      jd: jobDescription,
      keywords: keywordsLine,
    };
    const list = [...requisitions, next];
    setRequisitions(list);
    saveReqs(list);
    setReqName("");
  };

  const applyReq = (r: Requisition) => {
    setJobDescription(r.jd);
    setKeywordsLine(r.keywords);
  };

  const deleteReq = (id: string) => {
    const list = requisitions.filter((x) => x.id !== id);
    setRequisitions(list);
    saveReqs(list);
  };

  const toggleCompare = (index: number) => {
    setCompareIdx((prev) => {
      if (prev.includes(index)) return prev.filter((x) => x !== index);
      if (prev.length >= 3) return [...prev.slice(1), index];
      return [...prev, index];
    });
  };

  const comparePayloads = useMemo(
    () =>
      compareIdx
        .map((i) => {
          const r = results[i];
          return r && r.ok ? { filename: r.filename, data: r.data } : null;
        })
        .filter((x): x is { filename: string; data: ResumePayload } => Boolean(x)),
    [compareIdx, results]
  );

  const skillsRows = useMemo(() => skillsChartRows(resumes), [resumes]);
  const timelineRows = useMemo(() => experienceTimelineRows(resumes), [resumes]);
  const heatRows = useMemo(() => educationHeatmapRows(resumes), [resumes]);
  const projRows = useMemo(() => projectBubbleRows(resumes), [resumes]);
  const fitRows = useMemo(() => fitScoresForChart(resumes, jobDescription), [resumes, jobDescription]);
  const stacked = useMemo(() => stackedScoresForChart(resumes, jobDescription), [resumes, jobDescription]);
  const kwRows = useMemo(() => keywordDensityRows(resumes, keywords), [resumes, keywords]);
  const wordFreq = useMemo(() => wordFrequencyFromResumes(resumes, keywords), [resumes, keywords]);
  const { rows: radarRows, keys: radarKeys } = useMemo(() => radarDataset(resumes), [resumes]);

  const kwAgg = useMemo(() => {
    const m = new Map<string, number>();
    for (const r of kwRows) {
      m.set(r.keyword, (m.get(r.keyword) ?? 0) + r.count);
    }
    return [...m.entries()].map(([keyword, count]) => ({ keyword, count }));
  }, [kwRows]);

  const skillsRowsDisplay = useMemo(
    () => skillsRows.map((r) => ({ ...r, label: `${r.skill} (${r.name})` })),
    [skillsRows]
  );

  const csvBlob = useMemo(() => {
    if (!resumes.length) return "";
    return toCsv(normalizeResumeData(resumes));
  }, [resumes]);

  return (
    <div className="flex min-h-screen">
      <aside className="no-print w-64 shrink-0 border-r border-[var(--color-border)] bg-[var(--color-surface-elevated)] p-4">
        <div className="mb-6">
          <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-muted)]">
            Workspace
          </p>
          <h1 className="mt-1 text-lg font-bold text-[var(--color-foreground)]">Resume intelligence</h1>
        </div>
        <nav className="flex flex-col gap-1 text-sm">
          {(["overview", "charts", "compare"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setActiveTab(t)}
              className={`rounded-lg px-3 py-2 text-left font-medium capitalize ${
                activeTab === t
                  ? "bg-teal-50 text-[var(--color-primary)]"
                  : "text-[var(--color-muted)] hover:bg-slate-50"
              }`}
            >
              {t === "compare" ? "Compare" : t}
            </button>
          ))}
        </nav>
        <div className="mt-8 space-y-3 text-sm">
          <label className="block font-medium text-[var(--color-foreground)]" htmlFor="hiring-roles-text">
            Role(s) you&apos;re hiring for
          </label>
          <textarea
            id="hiring-roles-text"
            className="w-full rounded-lg border border-[var(--color-border)] bg-white p-2 text-xs"
            rows={6}
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste one job description or several postings (stacked in this box). Used for fit scoring across uploaded resumes."
          />
          <label className="block font-medium">Keywords</label>
          <input
            className="w-full rounded-lg border border-[var(--color-border)] bg-white p-2 text-xs"
            value={keywordsLine}
            onChange={(e) => setKeywordsLine(e.target.value)}
          />
          <div className="rounded-lg border border-dashed border-teal-200 bg-teal-50/50 p-2">
            <p className="text-xs font-medium text-teal-900">Requisitions</p>
            <button
              type="button"
              className="mt-1 text-xs text-teal-700 underline"
              onClick={hydrateReqs}
            >
              Load from browser
            </button>
            <div className="mt-2 flex gap-1">
              <input
                className="min-w-0 flex-1 rounded border bg-white px-1 py-0.5 text-xs"
                placeholder="Name to save"
                value={reqName}
                onChange={(e) => setReqName(e.target.value)}
              />
              <button
                type="button"
                className="rounded bg-[var(--color-primary)] px-2 py-1 text-xs text-white"
                onClick={saveRequisition}
              >
                Save
              </button>
            </div>
            <ul className="mt-2 max-h-32 space-y-1 overflow-y-auto text-xs">
              {requisitions.map((r) => (
                <li key={r.id} className="flex items-center justify-between gap-1">
                  <button type="button" className="truncate text-left underline" onClick={() => applyReq(r)}>
                    {r.name}
                  </button>
                  <button type="button" className="text-red-600" onClick={() => deleteReq(r.id)}>
                    ×
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </aside>

      <main className="flex-1 p-6">
        <header className="no-print mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="text-sm text-[var(--color-muted)]">Bright, business-ready hiring workspace</p>
            <h2 className="text-2xl font-semibold tracking-tight">Candidate analysis</h2>
          </div>
          <div className="flex max-w-xl flex-col items-stretch gap-2 sm:items-end">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx"
              multiple
              className="sr-only"
              aria-label="Upload resume PDF or Word files"
              onChange={(e) => {
                const picked = e.target.files ? [...e.target.files] : [];
                setFiles((prev) => mergePickedFiles(prev, picked));
                e.target.value = "";
              }}
            />
            <div className="flex flex-wrap justify-end gap-2">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="rounded-lg border border-[var(--color-border)] bg-white px-4 py-2 text-sm font-medium shadow-sm hover:border-teal-300"
              >
                Upload resume(s)
              </button>
              <button
                type="button"
                disabled={loading || !files.length}
                onClick={parseFiles}
                className="rounded-lg bg-[var(--color-primary)] px-5 py-2 text-sm font-semibold text-white shadow-sm disabled:opacity-50"
              >
                {loading ? "Parsing…" : "Parse with Gemini"}
              </button>
              <button
                type="button"
                className="rounded-lg border px-4 py-2 text-sm"
                onClick={() => {
                  setResults([]);
                  setFiles([]);
                  setCompareIdx([]);
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
              >
                Clear
              </button>
              <button
                type="button"
                className="rounded-lg border border-slate-300 px-4 py-2 text-sm"
                onClick={() => window.print()}
              >
                Print briefing
              </button>
              {csvBlob ? (
                <a
                  href={`data:text/csv;charset=utf-8,${encodeURIComponent(csvBlob)}`}
                  download="candidates.csv"
                  className="rounded-lg border border-[var(--color-accent)] px-4 py-2 text-sm font-medium text-[var(--color-accent)]"
                >
                  Download CSV
                </a>
              ) : null}
            </div>
            <p className="text-right text-xs text-[var(--color-muted)]">
              PDF or Word (.docx). In the file dialog, pick one file or several (Ctrl+click or Shift+click on
              Windows). You can upload again to add more, up to {PARSE_MAX_FILES} total.
            </p>
            {files.length > 0 ? (
              <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-3 py-2 text-left text-xs text-[var(--color-foreground)]">
                <p className="font-medium text-[var(--color-muted)]">
                  {files.length} file{files.length === 1 ? "" : "s"} queued
                </p>
                <ul className="mt-1 max-h-24 list-inside list-disc space-y-0.5 overflow-y-auto text-[var(--color-foreground)]">
                  {files.map((f) => (
                    <li key={fileDedupeKey(f)} className="truncate" title={f.name}>
                      {f.name}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        </header>

        {activeTab === "overview" && (
          <section className="space-y-6">
            <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-elevated)] p-4 shadow-sm">
              <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Parse results</h3>
              <ul className="mt-3 divide-y text-sm">
                {results.length === 0 && <li className="py-2 text-[var(--color-muted)]">No runs yet.</li>}
                {results.map((r, i) => (
                  <li key={`${r.filename}-${i}`} className="flex flex-wrap items-center gap-2 py-2">
                    <span className="font-medium">{r.filename}</span>
                    {r.ok ? (
                      <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-800">
                        Parsed · {r.data.Name || "Unnamed"}
                      </span>
                    ) : (
                      <span className="rounded-full bg-red-50 px-2 py-0.5 text-xs text-red-800">
                        {r.error}
                      </span>
                    )}
                    {r.ok && (
                      <label className="ml-auto flex items-center gap-1 text-xs text-[var(--color-muted)]">
                        <input
                          type="checkbox"
                          checked={compareIdx.includes(i)}
                          onChange={() => toggleCompare(i)}
                        />
                        Compare
                      </label>
                    )}
                  </li>
                ))}
              </ul>
            </div>

            <div id="briefing-print" className="rounded-xl border border-[var(--color-border)] bg-white p-4 print:border-0 print:shadow-none">
              <h3 className="text-sm font-semibold">Briefing summary</h3>
              <p className="mt-2 text-xs text-[var(--color-muted)]">
                {resumes.length} candidate(s). JD fit scores when description is non-empty.
              </p>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[640px] text-left text-xs">
                  <thead>
                    <tr className="border-b text-[var(--color-muted)]">
                      <th className="py-2 pr-2">Name</th>
                      <th className="py-2 pr-2">Fit score</th>
                      <th className="py-2">Summary</th>
                    </tr>
                  </thead>
                  <tbody>
                    {resumes.map((r) => (
                      <tr key={r.Name + r.Summary} className="border-b border-slate-100">
                        <td className="py-2 font-medium">{r.Name}</td>
                        <td className="py-2">{fitRows.find((x) => x.name === (r.Name || "Unknown"))?.score ?? "—"}</td>
                        <td className="max-w-md truncate py-2 text-[var(--color-muted)]">{r.Summary}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}

        {activeTab === "charts" && resumes.length === 0 && (
          <p className="text-sm text-[var(--color-muted)]">Parse at least one resume to unlock charts.</p>
        )}

        {activeTab === "charts" && resumes.length > 0 && (
          <section className="space-y-8">
            <div className="grid gap-6 lg:grid-cols-2">
              <ChartCard title="Skills comparison">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={skillsRowsDisplay}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="label" tick={{ fontSize: 9 }} interval={0} angle={-25} height={70} />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="proficiencyNumeric" name="Level" fill="#0d9488" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
              <ChartCard title="JD fit score">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={fitRows}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="score" fill="#2563eb" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>
            <ChartCard title="Stacked dimensions">
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={stacked}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="Name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Skills" stackId="a" fill="#0d9488" name="Skills" />
                  <Bar dataKey="Experience" stackId="a" fill="#2563eb" name="Experience" />
                  <Bar dataKey="Education" stackId="a" fill="#7c3aed" name="Education" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
            <div className="grid gap-6 lg:grid-cols-2">
              <ChartCard title="Keyword density">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={kwAgg}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="keyword" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#7c3aed">
                      {kwAgg.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
              <ChartCard title="Projects (bubble size ~ impact %)">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={projRows} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="Project" type="category" width={120} tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Bar dataKey="Impact" fill="#ea580c" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>
            <ChartCard title="Experience timeline (approx.)">
              <SimpleTimeline rows={timelineRows} />
            </ChartCard>
            <div className="grid gap-6 lg:grid-cols-2">
              <ChartCard title="Education heatmap">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={heatRows}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis dataKey="degree" type="category" width={100} />
                    <Tooltip />
                    <Bar dataKey="year" fill="#0891b2" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
              <ChartCard title="Skills radar">
                {radarRows.length === 0 ? (
                  <p className="text-sm text-[var(--color-muted)]">No overlapping skills to plot.</p>
                ) : (
                <ResponsiveContainer width="100%" height={360}>
                  <RadarChart data={radarRows}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="skill" tick={{ fontSize: 9 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 4]} />
                    {radarKeys.map((k, i) => (
                      <Radar
                        key={k}
                        name={okPayloads[i]?.data.Name || k}
                        dataKey={k}
                        stroke={COLORS[i % COLORS.length]}
                        fill={COLORS[i % COLORS.length]}
                        fillOpacity={0.2}
                      />
                    ))}
                    <Legend />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
                )}
              </ChartCard>
            </div>
            <ChartCard title="Word emphasis">
              <div className="flex flex-wrap gap-2">
                {wordFreq.slice(0, 40).map((w) => (
                  <span
                    key={w.text}
                    className="rounded-md bg-slate-100 px-2 py-1 text-slate-800"
                    style={{ fontSize: 10 + Math.min(w.value, 8) }}
                  >
                    {w.text}
                  </span>
                ))}
              </div>
            </ChartCard>
            {resumes[0] && (
              <ChartCard title="Education network (first candidate)">
                <div className="h-[360px] rounded-lg border bg-slate-50">
                  <EducationMiniFlow resume={resumes[0]} />
                </div>
              </ChartCard>
            )}
          </section>
        )}

        {activeTab === "compare" && (
          <section>
            <p className="mb-4 text-sm text-[var(--color-muted)]">
              Select up to three candidates from the overview list, then compare side by side.
            </p>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {comparePayloads.map((p) => (
                <div
                  key={p.filename}
                  className="rounded-xl border border-[var(--color-border)] bg-white p-4 shadow-sm"
                >
                  <h3 className="text-lg font-semibold">{p.data.Name}</h3>
                  <p className="text-xs text-[var(--color-muted)]">{p.filename}</p>
                  <p className="mt-3 text-sm">{p.data.Summary}</p>
                  <h4 className="mt-4 text-xs font-bold uppercase text-teal-800">Skills</h4>
                  <ul className="mt-1 text-xs">
                    {p.data.Skills.slice(0, 12).map((s) => (
                      <li key={s.Skill}>
                        {s.Skill} — {s.Proficiency}
                      </li>
                    ))}
                  </ul>
                  <h4 className="mt-4 text-xs font-bold uppercase text-teal-800">Experience</h4>
                  <ul className="mt-1 text-xs">
                    {p.data.Work_Experience.map((e) => (
                      <li key={e.Company + e.Duration}>
                        {e.Job_Title} @ {e.Company} ({e.Duration})
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-elevated)] p-4 shadow-sm">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      {children}
    </div>
  );
}

function SimpleTimeline({ rows }: { rows: TimelineRow[] }) {
  if (!rows.length) return <p className="text-sm text-[var(--color-muted)]">No timeline data.</p>;
  const min = Math.min(...rows.map((r) => r.start));
  const max = Math.max(...rows.map((r) => r.end));
  const span = Math.max(max - min, 1);
  return (
    <div className="space-y-2">
      {rows.map((r, i) => {
        const left = ((r.start - min) / span) * 100;
        const width = ((r.end - r.start) / span) * 100;
        return (
          <div key={i} className="text-xs">
            <div className="flex justify-between text-[var(--color-muted)]">
              <span>
                {r.name} — {r.title}
              </span>
              <span>{r.company}</span>
            </div>
            <div className="mt-1 h-3 w-full rounded bg-slate-100">
              <div
                className="h-3 rounded bg-[var(--color-primary)]"
                style={{ marginLeft: `${left}%`, width: `${Math.max(width, 2)}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

