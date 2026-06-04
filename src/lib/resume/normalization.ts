import type { ResumePayload } from "./schema";

export type NormalizedRow = {
  name: string;
  email: string;
  phone: string;
  summary: string;
  skills: string;
  experience: string;
  education: string;
  projects: string;
  certifications: string;
  core_competencies: string;
  key_achievements: string;
};

function normalizeSkills(skills: ResumePayload["Skills"]): string {
  return skills
    .map((s) => `${s.Skill} (${s.Proficiency})`)
    .filter((x) => x.trim())
    .join(", ");
}

function normalizeExperience(experience: ResumePayload["Work_Experience"]): string {
  return experience
    .map(
      (exp) =>
        `${exp.Company} (${exp.Job_Title}, ${exp.Duration})`
    )
    .join("; ");
}

function normalizeEducation(education: ResumePayload["Education"]): string {
  return education
    .map((e) => `${e.Degree} from ${e.Institution} (${e.Year})`)
    .join("; ");
}

function normalizeProjects(projects: ResumePayload["Projects"]): string {
  return projects.map((p) => `${p.Name}: ${p.Description}`).join("; ");
}

function normalizeAchievements(achievements: ResumePayload["Key_Achievements"]): string {
  return achievements
    .map((a) => `${a.Description} (${a.Metrics || "N/A"})`)
    .join("; ");
}

export function normalizeSingleResume(resume: ResumePayload): NormalizedRow {
  return {
    name: resume.Name,
    email: resume.Contact.Email,
    phone: resume.Contact.Phone,
    summary: resume.Summary,
    skills: normalizeSkills(resume.Skills),
    experience: normalizeExperience(resume.Work_Experience),
    education: normalizeEducation(resume.Education),
    projects: normalizeProjects(resume.Projects),
    certifications: resume.Certifications.join(", "),
    core_competencies: resume.Core_Competencies.join(", "),
    key_achievements: normalizeAchievements(resume.Key_Achievements),
  };
}

export function normalizeResumeData(resumes: ResumePayload[]): NormalizedRow[] {
  return resumes.map(normalizeSingleResume);
}

export function toCsv(rows: NormalizedRow[]): string {
  if (rows.length === 0) return "";
  const headers = Object.keys(rows[0]) as (keyof NormalizedRow)[];
  const esc = (v: string) => `"${String(v).replace(/"/g, '""')}"`;
  const line = (r: NormalizedRow) => headers.map((h) => esc(r[h])).join(",");
  return [headers.map((h) => esc(String(h))).join(","), ...rows.map(line)].join("\n");
}
