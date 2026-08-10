import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = process.cwd();
const outRoot = path.join(repoRoot, "output", "ieee_innovation_v1");
const phase3Root = path.join(outRoot, "phase3");
const previewRoot = path.join(repoRoot, ".artifact_phase3_previews");
const csvText = await fs.readFile(path.join(outRoot, "innovation_decision_matrix.csv"), "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Decision Matrix" });
const matrix = workbook.worksheets.getItem("Decision Matrix");
matrix.showGridLines = false;
matrix.freezePanes.freezeRows(1);
matrix.getUsedRange().format = { font: { name: "Aptos", size: 10 }, verticalAlignment: "top", wrapText: true };
matrix.getUsedRange().getRow(0).format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" }, wrapText: true };
matrix.getRange("A1:A4").format.columnWidth = 9;
matrix.getRange("B1:B4").format.columnWidth = 14;
matrix.getRange("C1:C4").format.columnWidth = 44;
matrix.getRange("D1:D4").format.columnWidth = 58;
matrix.getRange("E1:J4").format.columnWidth = 18;
matrix.getRange("K1:K4").format.columnWidth = 13;
matrix.getRange("L1:L4").format.columnWidth = 24;
matrix.getRange("M1:M4").format.columnWidth = 52;
matrix.getRange("A2:M2").format = { fill: "#E2F0D9", font: { bold: true, color: "#375623" }, wrapText: true };
matrix.getRange("A2:M4").format.rowHeight = 88;

const summary = workbook.worksheets.add("Recommendation");
summary.showGridLines = false;
summary.getRange("A1:F1").merge();
summary.getRange("A1").values = [["IEEE SOH Innovation Decision — ARS Phase 3"]];
summary.getRange("A1:F1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF", size: 16 } };
summary.getRange("A3:B8").values = [
  ["Decision", "Value"],
  ["Recommended candidate", "CILA"],
  ["Score", 92],
  ["Checkpoint 2", "PASS WITH MANDATORY SAFEGUARDS"],
  ["Model code modified", "No"],
  ["User confirmation", "Pending"],
];
summary.getRange("A3:B3").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
summary.getRange("B4:B8").format.wrapText = true;
summary.getRange("B4").format = { fill: "#E2F0D9", font: { bold: true, color: "#375623" } };
summary.getRange("D3:F7").values = [
  ["Boundary", "Status", "Interpretation"],
  ["Paper numeric matched", "No", "Frozen baseline remains below paper reference"],
  ["Transformer best observed", "No", "No superiority claim"],
  ["CILA RMSE improved", "Unknown", "No CILA experiment has run"],
  ["Outer test selected", "No", "Must be frozen before implementation"],
];
summary.getRange("D3:F3").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
summary.getRange("A10:F12").merge(true);
summary.getRange("A10:F12").values = [["Score 92/100 is a multi-criteria implementation decision, not an estimated probability of RMSE improvement. NASA condition and battery identity are confounded, so the future claim must be cross-condition battery generalization, not causal temperature robustness."], [null], [null]];
summary.getRange("A10:F12").format = { fill: "#FFF2CC", font: { color: "#7F6000" }, wrapText: true, verticalAlignment: "center" };
summary.getRange("A1:A12").format.columnWidth = 27;
summary.getRange("B1:B12").format.columnWidth = 33;
summary.getRange("C1:C12").format.columnWidth = 4;
summary.getRange("D1:D12").format.columnWidth = 27;
summary.getRange("E1:E12").format.columnWidth = 14;
summary.getRange("F1:F12").format.columnWidth = 45;

const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "Phase 3 formula error scan" });
console.log(errors.ndjson);
await fs.mkdir(previewRoot, { recursive: true });
for (const [sheetName, range] of [["Recommendation", "A1:F12"], ["Decision Matrix", "A1:M4"]]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewRoot, `${sheetName.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(phase3Root, "innovation_decision_evidence.xlsx"));
console.log(JSON.stringify({ workbook: path.join(phase3Root, "innovation_decision_evidence.xlsx") }));
