import type { ResumePayload } from "./schema";

export type ParseOk = { filename: string; ok: true; data: ResumePayload };
export type ParseFail = { filename: string; ok: false; error: string; detail?: string };
export type ParseItem = ParseOk | ParseFail;
