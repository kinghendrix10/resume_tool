import type { ResumePayload } from "./schema";

function proficiencyWeight(level: string, scale: "full" | "stacked"): number {
  const l = (level || "").trim();
  if (scale === "full") {
    const m: Record<string, number> = { Expert: 4, Advanced: 3, Intermediate: 2, Basic: 1 };
    return m[l] ?? 1;
  }
  const m: Record<string, number> = { Expert: 3, Advanced: 2, Intermediate: 1, Basic: 0 };
  return m[l] ?? 0;
}

function skillsMatchScore(
  skills: ResumePayload["Skills"] | Record<string, string> | unknown,
  jdLower: string,
  scale: "full" | "stacked"
): number {
  let score = 0;
  if (skills && typeof skills === "object" && !Array.isArray(skills)) {
    for (const [skill, proficiency] of Object.entries(skills as Record<string, string>)) {
      if (skill && jdLower.includes(skill.toLowerCase())) {
        score += proficiencyWeight(String(proficiency), scale);
      }
    }
    return score;
  }
  if (Array.isArray(skills)) {
    for (const item of skills) {
      if (item && typeof item === "object" && "Skill" in item) {
        const sk = String((item as { Skill?: string }).Skill ?? "");
        const pr = String((item as { Proficiency?: string }).Proficiency ?? "Intermediate");
        if (sk && jdLower.includes(sk.toLowerCase())) {
          score += proficiencyWeight(pr, scale);
        }
      }
    }
  }
  return score;
}

export function calculateResumeScore(resume: ResumePayload, jobDescription: string): number {
  const jdLower = (jobDescription || "").toLowerCase();
  return skillsMatchScore(resume.Skills, jdLower, "full");
}

export function categoryScore(
  resume: ResumePayload,
  category: "Skills" | "Experience" | "Education",
  jdLower: string
): number {
  if (category === "Skills") {
    return skillsMatchScore(resume.Skills, jdLower, "stacked");
  }
  if (category === "Experience") {
    return resume.Work_Experience?.length ?? 0;
  }
  if (category === "Education") {
    return resume.Education?.length ?? 0;
  }
  return 0;
}

export type StackedRow = { Name: string; Skills: number; Experience: number; Education: number };

export function stackedScoresForChart(
  resumes: ResumePayload[],
  jobDescription: string
): StackedRow[] {
  const jd = (jobDescription || "").toLowerCase();
  return resumes.map((r) => ({
    Name: r.Name || "Unknown",
    Skills: categoryScore(r, "Skills", jd),
    Experience: categoryScore(r, "Experience", jd),
    Education: categoryScore(r, "Education", jd),
  }));
}

export type FitScoreRow = { name: string; score: number };

export function fitScoresForChart(
  resumes: ResumePayload[],
  jobDescription: string
): FitScoreRow[] {
  return resumes.map((r) => ({
    name: r.Name || "Unknown",
    score: calculateResumeScore(r, jobDescription),
  }));
}
