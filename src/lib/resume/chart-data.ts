import type { ResumePayload } from "./schema";

export type SkillRow = { name: string; skill: string; proficiency: string; proficiencyNumeric: number };

const PROF_MAP: Record<string, number> = {
  Expert: 4,
  Advanced: 3,
  Intermediate: 2,
  Basic: 1,
  Unspecified: 0,
};

export function skillsChartRows(resumes: ResumePayload[]): SkillRow[] {
  const rows: SkillRow[] = [];
  for (const r of resumes) {
    const nm = r.Name || "Unknown";
    for (const s of r.Skills) {
      const pr = s.Proficiency || "Unspecified";
      rows.push({
        name: nm,
        skill: s.Skill,
        proficiency: pr,
        proficiencyNumeric: PROF_MAP[pr] ?? 0,
      });
    }
  }
  return rows;
}

export type TimelineRow = {
  name: string;
  company: string;
  title: string;
  start: number;
  end: number;
};

/** Parse duration to epoch ms range (approximate for chart). */
export function parseDurationMs(duration: string): { start: number; end: number } | null {
  if (!duration?.trim()) return null;
  const parts = duration.split(/\s+to\s+|\s*-\s*/i).map((s) => s.trim());
  const startStr = parts[0];
  const endStr = parts.length > 1 ? parts[1] : "Present";
  const start = Date.parse(startStr);
  let end: number;
  if (endStr.toLowerCase() === "present") {
    end = Date.now();
  } else {
    end = Date.parse(endStr);
  }
  if (Number.isNaN(start) || Number.isNaN(end)) return null;
  return { start, end };
}

export function experienceTimelineRows(resumes: ResumePayload[]): TimelineRow[] {
  const rows: TimelineRow[] = [];
  for (const r of resumes) {
    const nm = r.Name || "Unknown";
    for (const ex of r.Work_Experience) {
      const parsed = parseDurationMs(ex.Duration);
      if (!parsed) continue;
      rows.push({
        name: nm,
        company: ex.Company,
        title: ex.Job_Title,
        start: parsed.start,
        end: parsed.end,
      });
    }
  }
  return rows;
}

export type HeatmapRow = { name: string; degree: string; institution: string; year: string };

export function educationHeatmapRows(resumes: ResumePayload[]): HeatmapRow[] {
  const rows: HeatmapRow[] = [];
  for (const r of resumes) {
    const nm = r.Name || "Unknown";
    for (const e of r.Education) {
      rows.push({
        name: nm,
        degree: e.Degree,
        institution: e.Institution,
        year: e.Year,
      });
    }
  }
  return rows;
}

export type ProjectBubble = {
  Name: string;
  Project: string;
  Impact: number;
  Description: string;
};

export function projectBubbleRows(resumes: ResumePayload[]): ProjectBubble[] {
  const rows: ProjectBubble[] = [];
  for (const r of resumes) {
    const nm = r.Name || "Unknown";
    for (const p of r.Projects) {
      let impact = 0;
      const desc = p.Description || "";
      if (desc.includes("%")) {
        const before = desc.split("%")[0] ?? "";
        const digits = before.replace(/\D/g, "");
        if (digits) impact = Math.min(parseInt(digits, 10) || 0, 100);
      }
      rows.push({
        Name: nm,
        Project: p.Name || "Project",
        Impact: impact,
        Description: desc || "—",
      });
    }
  }
  return rows;
}

export type EducationEdge = { source: string; target: string };

export function educationGraphEdges(resume: ResumePayload): { nodes: { id: string; label: string; type: string }[]; edges: EducationEdge[] } {
  const name = resume.Name || "Candidate";
  const nodes: { id: string; label: string; type: string }[] = [
    { id: `p-${name}`, label: name, type: "person" },
  ];
  const edges: EducationEdge[] = [];
  for (const e of resume.Education) {
    const inst = e.Institution || "School";
    const deg = e.Degree || "Degree";
    const nidI = `i-${inst}`;
    const nidD = `d-${deg}`;
    if (!nodes.find((n) => n.id === nidI)) nodes.push({ id: nidI, label: inst, type: "institution" });
    if (!nodes.find((n) => n.id === nidD)) nodes.push({ id: nidD, label: deg, type: "degree" });
    edges.push({ source: `p-${name}`, target: nidI });
    edges.push({ source: `p-${name}`, target: nidD });
  }
  return { nodes, edges };
}

export function keywordDensityRows(
  resumes: ResumePayload[],
  keywords: string[]
): { name: string; keyword: string; count: number }[] {
  const rows: { name: string; keyword: string; count: number }[] = [];
  for (const r of resumes) {
    const blob = JSON.stringify(r).toLowerCase();
    const nm = r.Name || "Unknown";
    for (const kw of keywords) {
      const k = kw.trim();
      if (!k) continue;
      const re = new RegExp(k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
      const matches = blob.match(re);
      rows.push({ name: nm, keyword: k, count: matches?.length ?? 0 });
    }
  }
  return rows;
}

export function wordFrequencyFromResumes(resumes: ResumePayload[], extra: string[]): { text: string; value: number }[] {
  const blob = resumes
    .map((r) => JSON.stringify(r).toLowerCase())
    .join(" ");
  const extraBlob = extra.join(" ").toLowerCase();
  const combined = `${blob} ${extraBlob}`;
  const words = combined.match(/[a-z]{4,}/g) ?? [];
  const counts = new Map<string, number>();
  for (const w of words) {
    counts.set(w, (counts.get(w) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 60)
    .map(([text, value]) => ({ text, value }));
}
