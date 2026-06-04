import { describe, expect, it } from "vitest";
import { ResumePayloadSchema } from "./schema";
import { calculateResumeScore } from "./scoring";

const sample = ResumePayloadSchema.parse({
  Name: "Dev",
  Contact: { Email: "", Phone: "" },
  Summary: "",
  Skills: [{ Skill: "Python", Proficiency: "Expert" }],
  Work_Experience: [],
  Education: [],
  Projects: [],
  Certifications: [],
  Core_Competencies: [],
  Key_Achievements: [],
});

describe("calculateResumeScore", () => {
  it("scores when jd contains skill", () => {
    const jd = "We need Python expertise for our team.";
    expect(calculateResumeScore(sample, jd)).toBeGreaterThan(0);
  });

  it("zero when no overlap", () => {
    expect(calculateResumeScore(sample, "Only COBOL here")).toBe(0);
  });
});
