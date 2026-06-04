import { GoogleGenAI } from "@google/genai";
import { createParsePrompt, createRepairPrompt } from "./prompts";
import { hasError, postProcessLlmResponse } from "./extract-json";
import { validateAndFillResume } from "./resume-validate";
import type { ResumePayload } from "./schema";

export type ParseResult = ResumePayload | { error: string; response?: string };

export async function parseResumeWithGemini(resumeText: string): Promise<ParseResult> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return { error: "GEMINI_API_KEY is not configured on the server." };
  }

  const model = process.env.GEMINI_MODEL ?? "gemini-2.0-flash";
  const ai = new GoogleGenAI({ apiKey });

  const runPrompt = async (prompt: string) => {
    const res = await ai.models.generateContent({
      model,
      contents: prompt,
      config: {
        maxOutputTokens: 8192,
        temperature: 0,
      },
    });
    const text = typeof res.text === "string" ? res.text : undefined;
    if (text?.trim()) return text;
    const cand = res.candidates?.[0];
    const parts = cand?.content?.parts;
    if (Array.isArray(parts)) {
      const joined = parts.map((p) => ("text" in p ? String(p.text ?? "") : "")).join("");
      if (joined.trim()) return joined;
    }
    return null;
  };

  const prompt = createParsePrompt(resumeText);
  let response = await runPrompt(prompt);
  if (!response) {
    return { error: "Failed to get response from LLM." };
  }

  let data = postProcessLlmResponse(response);
  if (hasError(data)) {
    const repair = await runPrompt(createRepairPrompt(resumeText, response));
    if (repair) {
      response = repair;
      data = postProcessLlmResponse(repair);
    }
  }
  if (hasError(data)) {
    return { error: data.error, response: data.response };
  }

  return validateAndFillResume(data);
}
