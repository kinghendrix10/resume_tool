import mammoth from "mammoth";

export const MAX_UPLOAD_BYTES = 8 * 1024 * 1024; // 8 MB per file
export const MAX_FILES = 12;

export async function extractTextFromBuffer(
  buffer: Buffer,
  filename: string
): Promise<string> {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".docx")) {
    const result = await mammoth.extractRawText({ buffer });
    return result.value ?? "";
  }
  if (lower.endsWith(".pdf")) {
    const pdfParse = (await import("pdf-parse")).default;
    const data = await pdfParse(buffer);
    return data.text ?? "";
  }
  throw new Error(`Unsupported file type: ${filename}`);
}
