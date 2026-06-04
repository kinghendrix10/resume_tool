import type { ResumePayload } from "./schema";

/** Rows for Recharts Radar: `skill` + `c0`, `c1`, ... numeric levels 0–4. */
export function radarDataset(resumes: ResumePayload[]): {
  rows: Record<string, string | number>[];
  keys: string[];
} {
  const skillSet = new Set<string>();
  for (const r of resumes) {
    for (const s of r.Skills) {
      if (s.Skill) skillSet.add(s.Skill);
    }
  }
  const skills = [...skillSet].sort();
  const keys = resumes.map((_, i) => `c${i}`);
  const level = (p: string) => {
    const m: Record<string, number> = { Expert: 4, Advanced: 3, Intermediate: 2, Basic: 1 };
    return m[p] ?? 0;
  };
  const rows = skills.map((skill) => {
    const row: Record<string, string | number> = { skill };
    resumes.forEach((r, i) => {
      const found = r.Skills.find((s) => s.Skill === skill);
      row[`c${i}`] = found ? level(found.Proficiency) : 0;
    });
    return row;
  });
  return { rows, keys };
}
