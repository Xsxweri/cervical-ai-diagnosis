export interface ReportSection {
  /** null 表示标题之前的引言 */
  title: string | null;
  body: string;
}

function cleanBody(body: string): string {
  return body
    .split(/\r?\n/)
    .filter((l) => !/^\s*-{3,}\s*$/.test(l))
    .join('\n')
    .trim();
}

/**
 * 把 LLM 报告按 Markdown 标题（#/##/###）切成章节。
 *
 * 设计前提：后端目前返回自由文本（`response.content`），并不保证有标题。
 * 因此本函数在**没有标题时返回 null**，调用方应降级为整段渲染——
 * 绝不臆造章节，避免把正文切碎。
 */
export function parseReportSections(text: string): ReportSection[] | null {
  const lines = text.split(/\r?\n/);
  const headingRe = /^\s{0,3}#{1,3}\s+(.+?)\s*#*\s*$/;

  if (!lines.some((l) => headingRe.test(l))) return null;

  const out: ReportSection[] = [];
  let cur: ReportSection = { title: null, body: '' };

  for (const line of lines) {
    const m = headingRe.exec(line);
    if (m) {
      if (cur.title || cur.body.trim()) {
        out.push({ ...cur, body: cleanBody(cur.body) });
      }
      cur = { title: m[1], body: '' };
    } else {
      cur.body += (cur.body ? '\n' : '') + line;
    }
  }
  if (cur.title || cur.body.trim()) {
    out.push({ ...cur, body: cleanBody(cur.body) });
  }
  return out;
}