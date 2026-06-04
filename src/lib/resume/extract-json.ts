/**
 * Extract JSON substring from LLM output (fenced or brace slice).
 */
export function extractJsonString(raw: string | null | undefined): string | null {
  if (!raw || !raw.trim()) return null;
  const fence = raw.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fence) return fence[1].trim();
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start !== -1 && end !== -1 && end > start) {
    return raw.slice(start, end + 1).trim();
  }
  return raw.trim();
}

export function postProcessLlmResponse(response: string): Record<string, unknown> | ParseErrorShape {
  const candidate = extractJsonString(response);
  if (!candidate) {
    return { error: "Empty or unparseable LLM response.", response: response.slice(0, 2000) };
  }
  try {
    const data = JSON.parse(candidate) as unknown;
    if (data !== null && typeof data === "object" && !Array.isArray(data)) {
      return data as Record<string, unknown>;
    }
    return { error: "JSON root was not an object.", response: response.slice(0, 2000) };
  } catch {
    return { error: "Failed to parse JSON from LLM response.", response: candidate.slice(0, 2000) };
  }
}

export type ParseErrorShape = { error: string; response?: string };

export function hasError(
  v: Record<string, unknown> | ParseErrorShape
): v is ParseErrorShape {
  return "error" in v && typeof (v as ParseErrorShape).error === "string";
}
