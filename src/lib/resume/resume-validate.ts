import type { ResumePayload } from "./schema";
import { ResumePayloadSchema } from "./schema";

function coerceContact(val: unknown): { Email: string; Phone: string } {
  if (val && typeof val === "object" && !Array.isArray(val)) {
    const o = val as Record<string, unknown>;
    return {
      Email: String(o.Email ?? ""),
      Phone: String(o.Phone ?? ""),
    };
  }
  return { Email: "", Phone: "" };
}

function ensureListOfDicts(val: unknown): Record<string, unknown>[] {
  if (!val) return [];
  if (Array.isArray(val)) {
    return val.filter(
      (x): x is Record<string, unknown> =>
        x !== null && typeof x === "object" && !Array.isArray(x)
    );
  }
  if (typeof val === "object" && !Array.isArray(val)) {
    return [val as Record<string, unknown>];
  }
  return [];
}

function coerceWork(row: Record<string, unknown>) {
  return {
    Company: String(row.Company ?? ""),
    Job_Title: String(row.Job_Title ?? row["Job Title"] ?? row.Title ?? ""),
    Duration: String(row.Duration ?? ""),
    Location: String(row.Location ?? ""),
    Achievements: ensureListOfDicts(row.Achievements).map((a) => ({
      Description: String(a.Description ?? ""),
      Metrics: String(a.Metrics ?? ""),
    })),
  };
}

function coerceEdu(row: Record<string, unknown>) {
  return {
    Degree: String(row.Degree ?? ""),
    Institution: String(row.Institution ?? ""),
    Year: String(row.Year ?? ""),
  };
}

function coerceProject(row: Record<string, unknown>) {
  return {
    Name: String(row.Name ?? row.Title ?? ""),
    Description: String(row.Description ?? ""),
    Impact: String(row.Impact ?? ""),
  };
}

function coerceAchievement(row: Record<string, unknown>) {
  return {
    Description: String(row.Description ?? row.Achievement ?? ""),
    Metrics: String(
      row.Metrics ?? row["Quantifiable Metrics"] ?? row.KeyMetric ?? ""
    ),
  };
}

/**
 * Normalize arbitrary LLM JSON into the canonical resume shape (mirrors Python validate_and_fill_resume).
 */
export function validateAndFillResume(data: unknown): ResumePayload {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return ResumePayloadSchema.parse({});
  }
  const d = data as Record<string, unknown>;

  const skillsRaw = d.Skills;
  let skills: { Skill: string; Proficiency: string }[] = [];
  if (skillsRaw && typeof skillsRaw === "object" && !Array.isArray(skillsRaw)) {
    skills = Object.entries(skillsRaw as Record<string, unknown>).map(([k, v]) => ({
      Skill: String(k),
      Proficiency: v != null && v !== "" ? String(v) : "Unspecified",
    }));
  } else {
    skills = ensureListOfDicts(skillsRaw).map((row) => ({
      Skill: String(row.Skill ?? ""),
      Proficiency: String(row.Proficiency ?? "Unspecified"),
    }));
  }

  const work = d.Work_Experience ?? d["Work Experience"] ?? d.Experience;
  const keyAch = d.Key_Achievements ?? d["Key Achievements"] ?? d.Achievements;

  const certs = d.Certifications;
  const certifications: string[] = [];
  if (Array.isArray(certs)) {
    for (const c of certs) {
      if (c == null) continue;
      if (typeof c === "string") certifications.push(c);
      else if (typeof c === "object" && c !== null && "Name" in c)
        certifications.push(String((c as { Name?: string }).Name ?? ""));
      else certifications.push(String(c));
    }
  }

  const cc = d.Core_Competencies;
  const coreCompetencies = Array.isArray(cc) ? cc.map((x) => String(x)).filter(Boolean) : [];

  const merged = {
    Name: String(d.Name ?? ""),
    Contact: coerceContact(d.Contact),
    Summary: String(d.Summary ?? ""),
    Skills: skills,
    Work_Experience: ensureListOfDicts(work).map(coerceWork),
    Education: ensureListOfDicts(d.Education).map(coerceEdu),
    Projects: ensureListOfDicts(d.Projects).map(coerceProject),
    Certifications: certifications,
    Core_Competencies: coreCompetencies,
    Key_Achievements: ensureListOfDicts(keyAch).map(coerceAchievement),
  };

  const parsed = ResumePayloadSchema.safeParse(merged);
  if (parsed.success) return parsed.data;
  return ResumePayloadSchema.parse({});
}
