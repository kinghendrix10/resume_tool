import { describe, expect, it } from "vitest";
import { validateAndFillResume } from "./resume-validate";

describe("validateAndFillResume", () => {
  it("fills defaults", () => {
    const r = validateAndFillResume({});
    expect(r.Name).toBe("");
    expect(r.Skills).toEqual([]);
  });

  it("normalizes skills dict", () => {
    const r = validateAndFillResume({
      Name: "T",
      Skills: { Python: "Expert" },
    });
    expect(r.Skills[0]).toEqual({ Skill: "Python", Proficiency: "Expert" });
  });
});
