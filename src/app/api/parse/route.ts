import { NextRequest, NextResponse } from "next/server";
import type { ParseItem } from "@/lib/resume/api-types";
import { extractTextFromBuffer, MAX_FILES, MAX_UPLOAD_BYTES } from "@/lib/resume/extract-text";
import { parseResumeWithGemini } from "@/lib/resume/gemini-parse";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const files = form.getAll("files") as File[];
    if (!files.length) {
      return NextResponse.json({ error: "No files uploaded." }, { status: 400 });
    }
    if (files.length > MAX_FILES) {
      return NextResponse.json(
        { error: `Maximum ${MAX_FILES} files per request.` },
        { status: 400 }
      );
    }

    const results: ParseItem[] = [];

    for (const file of files) {
      if (!(file instanceof File)) continue;
      const buf = Buffer.from(await file.arrayBuffer());
      if (buf.length > MAX_UPLOAD_BYTES) {
        results.push({
          filename: file.name,
          ok: false,
          error: "File too large",
          detail: `Max ${MAX_UPLOAD_BYTES} bytes`,
        });
        continue;
      }
      const lower = file.name.toLowerCase();
      if (!lower.endsWith(".pdf") && !lower.endsWith(".docx")) {
        results.push({
          filename: file.name,
          ok: false,
          error: "Unsupported type",
          detail: "Use PDF or DOCX",
        });
        continue;
      }

      let text: string;
      try {
        text = await extractTextFromBuffer(buf, file.name);
      } catch (e) {
        results.push({
          filename: file.name,
          ok: false,
          error: "Extraction failed",
          detail: e instanceof Error ? e.message : String(e),
        });
        continue;
      }

      if (!text.trim()) {
        results.push({
          filename: file.name,
          ok: false,
          error: "No text extracted",
        });
        continue;
      }

      const parsed = await parseResumeWithGemini(text);
      if ("error" in parsed) {
        results.push({
          filename: file.name,
          ok: false,
          error: parsed.error,
          detail: parsed.response,
        });
        continue;
      }

      results.push({ filename: file.name, ok: true, data: parsed });
    }

    return NextResponse.json({ results });
  } catch (e) {
    console.error(e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Parse failed" },
      { status: 500 }
    );
  }
}

