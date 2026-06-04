import { describe, expect, it } from "vitest";
import { extractJsonString, hasError, postProcessLlmResponse } from "./extract-json";

describe("extractJsonString", () => {
  it("parses fenced json", () => {
    const raw = '```json\n{"Name":"A"}\n```';
    expect(extractJsonString(raw)).toBe('{"Name":"A"}');
  });

  it("parses brace slice", () => {
    const raw = 'prefix {"Name":"Bob"} suffix';
    expect(extractJsonString(raw)).toBe('{"Name":"Bob"}');
  });
});

describe("postProcessLlmResponse", () => {
  it("returns object on valid json", () => {
    const r = postProcessLlmResponse('{"Name":"X"}');
    expect(hasError(r)).toBe(false);
    expect((r as Record<string, unknown>).Name).toBe("X");
  });

  it("returns error on invalid", () => {
    const r = postProcessLlmResponse("not json");
    expect(hasError(r)).toBe(true);
  });
});
