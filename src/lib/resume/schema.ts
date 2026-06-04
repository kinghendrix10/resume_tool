import { z } from "zod";

export const ContactSchema = z.object({
  Email: z.string().optional().default(""),
  Phone: z.string().optional().default(""),
});

export const SkillEntrySchema = z.object({
  Skill: z.string().optional().default(""),
  Proficiency: z.string().optional().default("Unspecified"),
});

export const AchievementSchema = z.object({
  Description: z.string().optional().default(""),
  Metrics: z.string().optional().default(""),
});

export const WorkExperienceSchema = z.object({
  Company: z.string().optional().default(""),
  Job_Title: z.string().optional().default(""),
  Duration: z.string().optional().default(""),
  Location: z.string().optional().default(""),
  Achievements: z.array(AchievementSchema).optional().default([]),
});

export const EducationSchema = z.object({
  Degree: z.string().optional().default(""),
  Institution: z.string().optional().default(""),
  Year: z.string().optional().default(""),
});

export const ProjectSchema = z.object({
  Name: z.string().optional().default(""),
  Description: z.string().optional().default(""),
  Impact: z.string().optional().default(""),
});

export const ResumePayloadSchema = z.object({
  Name: z.string().default(""),
  Contact: ContactSchema.default({ Email: "", Phone: "" }),
  Summary: z.string().default(""),
  Skills: z.array(SkillEntrySchema).default([]),
  Work_Experience: z.array(WorkExperienceSchema).default([]),
  Education: z.array(EducationSchema).default([]),
  Projects: z.array(ProjectSchema).default([]),
  Certifications: z.array(z.string()).default([]),
  Core_Competencies: z.array(z.string()).default([]),
  Key_Achievements: z.array(AchievementSchema).default([]),
});

export type ResumePayload = z.infer<typeof ResumePayloadSchema>;
