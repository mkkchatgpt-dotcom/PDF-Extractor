import { useState } from "react";
import type { ChangeEvent } from "react";
import axios from "axios";

type RecordItem = {
  record_id: number;
  page_number: number;
  object_type: string;
  annotation_type: string;
  author: string | null;
  modified_time: string | null;
  text: string;
};
type FileResult = {
  filename: string;
  summary: { pages: number; annotations: number };
  records: RecordItem[];
  success: boolean;
  error: string | null;
  download_url?: string;
};
type BatchResult = {
  total_files: number;
  successful_files: number;
  failed_files: number;
  results: FileResult[];
  download_url: string;
  download_type: "xlsx" | "zip";
};
const HEADERS = ["File Name", "Pg. No", "Type", "Author", "Modified Time", "Comments"];
const fileKey = (file: File) => JSON.stringify([file.name, file.size, file.lastModified]);

export default function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<BatchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function handleFileSelection(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files || []);
    event.target.value = "";
    if (!selected.length || busy) return;
    const pdfs = selected.filter(file => file.name.toLowerCase().endsWith(".pdf"));
    setError(pdfs.length !== selected.length ? "Only PDF files can be added. Other files were skipped." : "");
    if (!pdfs.length) return;
    setFiles(current => {
      const known = new Set(current.map(fileKey));
      const added = pdfs.filter(file => {
        const key = fileKey(file);
        if (known.has(key)) return false;
        known.add(key);
        return true;
      });
      return [...current, ...added];
    });
    setResult(null);
  }
  function removeFile(index: number) {
    if (busy) return;
    setFiles(current => current.filter((_, i) => i !== index));
    setResult(null);
    setError("");
  }
  function clearFiles() {
    if (busy) return;
    setFiles([]);
    setResult(null);
    setError("");
  }
  async function extract() {
    if (busy) return;
    if (!files.length) { setError("Please select at least one PDF file."); return; }
    setBusy(true);
    setError("");
    setResult(null);
    const form = new FormData();
    files.forEach(file => form.append("files", file));
    try {
      const response = await axios.post<BatchResult>("/api/extract", form);
      setResult(response.data);
    } catch (err: unknown) {
      const detail = axios.isAxiosError(err) ? err.response?.data?.detail : undefined;
      setError(typeof detail === "string" ? detail : "Could not process the selected PDF files.");
    } finally { setBusy(false); }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="eyebrow">DOCUMENT REVIEW TOOL</div>
        <h1>PDF Review Extractor</h1>
        <p>Extract PDF annotations and comments into Excel.</p>
      </header>
      <main className="container">
        <section className="upload-card">
          <h2>Upload PDFs</h2>
          <p className="muted">Select several PDFs together, or add them one at a time. New selections are added to your list.</p>
          <label className={`dropzone${busy ? " disabled" : ""}`}>
            <input type="file" accept=".pdf,application/pdf" multiple disabled={busy} onChange={handleFileSelection} />
            <strong>{files.length ? `${files.length} PDF file(s) selected — Add more PDFs` : "Choose PDF files"}</strong>
            <span>{busy ? "Processing selected files…" : "Click to select PDFs. Use Ctrl or Shift to select multiple files."}</span>
          </label>
          {!!files.length && (
            <div className="selected-files">
              <div className="selected-files-header"><strong>Selected PDFs</strong><button type="button" className="clear-button" disabled={busy} onClick={clearFiles}>Clear All</button></div>
              <div className="file-list">{files.map((file, index) => (
                <div className="file-item" key={fileKey(file)}>
                  <div className="file-number">{index + 1}</div>
                  <div className="file-info"><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB</span></div>
                  <div className="file-actions">
                    {result?.results[index]?.success && result.results[index].download_url && (
                      <a className="file-download" href={result.results[index].download_url} aria-label={`Download Excel for ${file.name}`}>Download Excel</a>
                    )}
                    <button type="button" className="remove-button" disabled={busy} onClick={() => removeFile(index)} aria-label={`Remove ${file.name}`}>Remove</button>
                  </div>
                </div>
              ))}</div>
            </div>
          )}
          <button type="button" className="primary" disabled={!files.length || busy} onClick={extract}>{busy ? `Extracting ${files.length} PDF(s)…` : files.length > 1 ? `Extract All ${files.length} PDFs` : "Extract PDF"}</button>
          {busy && <div className="processing-message" role="status">Processing PDFs and generating Excel files…</div>}
          {error && <div className="error" role="alert">{error}</div>}
        </section>
        {result && (
          <section className="results">
            <div className="batch-title"><h2>Extraction Complete</h2></div>
            <div className="summary">
              <div><span>Total PDFs</span><strong>{result.total_files}</strong></div>
              <div><span>Successful</span><strong>{result.successful_files}</strong></div>
              <div><span>Failed</span><strong>{result.failed_files}</strong></div>
              <a className="download" href={result.download_url}>{result.download_type === "xlsx" ? "Download Excel" : "Download All as ZIP"}</a>
            </div>
            <div className="download-info">{result.download_type === "xlsx" ? "Download the extracted comments as an Excel file (.xlsx)." : "Download the successfully generated Excel files together as a ZIP. Use Download Excel beside each selected PDF for individual downloads."}</div>
            {result.results.map((pdf, pdfIndex) => (
              <div className="pdf-result" key={`${pdf.filename}-${pdfIndex}`}>
                <div className="pdf-result-header">
                  <div><span className="pdf-number">PDF {pdfIndex + 1}</span><h3>{pdf.filename}</h3></div>
                  <div className={pdf.success ? "status-success" : "status-failed"}>{pdf.success ? "Processed" : "Failed"}</div>
                </div>
                {pdf.error && <div className="error">{pdf.error}</div>}
                {pdf.success && <>
                  <div className="pdf-summary">
                    <div><span>Pages</span><strong>{pdf.summary.pages}</strong></div>
                    <div><span>Annotations</span><strong>{pdf.summary.annotations}</strong></div>

                  </div>
                  <div className="table-wrap"><table>
                    <thead><tr>{HEADERS.map(header => <th key={header} scope="col">{header}</th>)}</tr></thead>
                    <tbody>{!pdf.records.length ? <tr><td colSpan={HEADERS.length} className="empty">No PDF annotations were found.</td></tr> : pdf.records.map((item, index) => (
                      <tr key={`${item.record_id}-${index}`}>
                        <td>{pdf.filename}</td><td>{item.page_number}</td><td>{item.annotation_type || item.object_type}</td><td>{item.author || ""}</td><td>{item.modified_time || ""}</td><td className="comment-cell">{item.text}</td>
                      </tr>
                    ))}</tbody>
                  </table></div>
                </>}
              </div>
            ))}
          </section>
        )}
      </main>
    </div>
  );
}
