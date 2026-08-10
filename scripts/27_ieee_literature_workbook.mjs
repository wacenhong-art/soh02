import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = process.cwd();
const evidenceRoot = path.join(repoRoot, "output", "ieee_innovation_v1");
const phase2Root = path.join(evidenceRoot, "phase2");
const previewRoot = path.join(repoRoot, ".artifact_phase2_previews");

const sources = [
  ["Search Log", path.join(evidenceRoot, "literature_search_log.csv")],
  ["Candidates", path.join(evidenceRoot, "ieee_literature_candidates.csv")],
  ["Deep Read", path.join(evidenceRoot, "ieee_literature_matrix.csv")],
  ["Verification", path.join(phase2Root, "source_verification_matrix.csv")],
];

const firstCsv = await fs.readFile(sources[0][1], "utf8");
const workbook = await Workbook.fromCSV(firstCsv, { sheetName: sources[0][0] });
for (const [sheetName, csvPath] of sources.slice(1)) {
  const csvText = await fs.readFile(csvPath, "utf8");
  await workbook.fromCSV(csvText, { sheetName });
}

const summary = workbook.worksheets.add("Summary");
summary.showGridLines = false;
summary.getRange("A1:F1").merge();
summary.getRange("A1").values = [["IEEE SOH Literature Evidence — ARS Phase 2"]];
summary.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
summary.getRange("A1:F1").format.rowHeight = 30;
summary.getRange("A3:B8").values = [
  ["Metric", "Value"],
  ["Search/verification calls logged", null],
  ["IEEE candidates screened", null],
  ["Deep-read sources", null],
  ["Verified deep reads", null],
  ["Phase 2 gate", "PASS"],
];
summary.getRange("B4").formulas = [["=COUNTA('Search Log'!A2:A200)"]];
summary.getRange("B5").formulas = [["=COUNTA('Candidates'!F2:F500)"]];
summary.getRange("B6").formulas = [["=COUNTA('Deep Read'!A2:A100)"]];
summary.getRange("B7").formulas = [["=COUNTIF('Verification'!K2:K100,\"VERIFIED\")"]];
summary.getRange("A3:B3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  borders: { preset: "outside", style: "thin", color: "#9FBAD0" },
};
summary.getRange("A4:B8").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
  top: { style: "thin", color: "#9FBAD0" },
  bottom: { style: "thin", color: "#9FBAD0" },
  left: { style: "thin", color: "#9FBAD0" },
  right: { style: "thin", color: "#9FBAD0" },
};
summary.getRange("B8").format = { fill: "#E2F0D9", font: { bold: true, color: "#375623" } };
summary.getRange("D3:F7").values = [
  ["Audit boundary", "Status", "Meaning"],
  ["Model code modified", "No", "Literature-only phase"],
  ["Test battery selected", "No", "Untouched pool remains unassigned"],
  ["Full-text human read", "No", "Abstract-level verified corpus"],
  ["Innovation selected", "No", "Requires Phase 3 synthesis + DA checkpoint"],
];
summary.getRange("D3:F3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
};
summary.getRange("D4:F7").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
};
summary.getRange("A10:F12").merge(true);
summary.getRange("A10:F12").values = [[
  "Interpretation: the Phase 2 gate confirms a reproducible and independently verified evidence corpus. It does not claim that any candidate method is already suitable for G-MTC, nor that the original baseline has reached the paper's numerical results.",
], [null], [null]];
summary.getRange("A10:F12").format = {
  fill: "#FFF2CC",
  font: { color: "#7F6000" },
  wrapText: true,
  verticalAlignment: "center",
};
summary.getRange("A1:A12").format.columnWidth = 29;
summary.getRange("B1:B12").format.columnWidth = 18;
summary.getRange("C1:C12").format.columnWidth = 4;
summary.getRange("D1:D12").format.columnWidth = 24;
summary.getRange("E1:E12").format.columnWidth = 14;
summary.getRange("F1:F12").format.columnWidth = 38;

const deepView = workbook.worksheets.add("Deep Read View");
deepView.showGridLines = false;
deepView.freezePanes.freezeRows(1);
deepView.getRange("A1:G1").values = [["DOI", "Title", "Publication date", "Real-world problem", "Method", "G-MTC boundary", "Status"]];
deepView.getRange("A2:G2").formulas = [[
  "='Deep Read'!A2",
  "='Deep Read'!B2",
  "='Deep Read'!D2",
  "='Deep Read'!I2",
  "='Deep Read'!J2",
  "='Deep Read'!P2",
  "='Deep Read'!T2",
]];
deepView.getRange("A2:G11").fillDown();
deepView.getRange("A1:G1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
deepView.getRange("A2:G11").format = { wrapText: true, verticalAlignment: "top" };
deepView.getRange("A1:A11").format.columnWidth = 34;
deepView.getRange("B1:B11").format.columnWidth = 48;
deepView.getRange("C1:C11").format.columnWidth = 17;
deepView.getRange("D1:F11").format.columnWidth = 42;
deepView.getRange("G1:G11").format.columnWidth = 16;
deepView.getRange("A2:G11").format.rowHeight = 88;

for (const [sheetName] of sources) {
  const sheet = workbook.worksheets.getItem(sheetName);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getUsedRange();
  const header = used.getRow(0);
  header.format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  header.format.rowHeight = 32;
  used.format.font = { name: "Aptos", size: 10 };
  used.format.verticalAlignment = "top";
  used.format.autofitColumns();
  used.format.columnWidth = 18;
  if (sheetName === "Search Log") {
    sheet.getRange("B1").values = [["executed_at (UTC)"]];
    sheet.getRange("B2:B20").setNumberFormat("yyyy-mm-dd hh:mm");
    sheet.getRange("C1:C20").format.columnWidth = 30;
    sheet.getRange("D1:D20").format.columnWidth = 42;
    sheet.getRange("E1:E20").format.columnWidth = 34;
    sheet.getRange("F1:F20").format.columnWidth = 38;
    sheet.getRange("I1:I20").format.columnWidth = 55;
  } else if (sheetName === "Candidates") {
    sheet.getRange("A1:A150").format.columnWidth = 55;
    sheet.getRange("B1:B150").format.columnWidth = 35;
    sheet.getRange("D1:D150").format.columnWidth = 34;
    sheet.getRange("L1:L150").format.columnWidth = 50;
  } else if (sheetName === "Deep Read") {
    sheet.getRange("B1:B20").format.columnWidth = 55;
    sheet.getRange("I1:P20").format.columnWidth = 44;
    sheet.getRange("W1:W20").format.columnWidth = 70;
  } else if (sheetName === "Verification") {
    sheet.getRange("A1:A20").format.columnWidth = 34;
    sheet.getRange("G1:G20").format.columnWidth = 42;
    sheet.getRange("J1:J20").format.columnWidth = 34;
    sheet.getRange("K1:K20").format.columnWidth = 20;
    sheet.getRange("L1:L20").format.columnWidth = 48;
  }
}

await fs.mkdir(previewRoot, { recursive: true });
for (const [sheetName, range] of [
  ["Summary", "A1:F12"],
  ["Search Log", "A1:I11"],
  ["Candidates", "A1:L12"],
  ["Deep Read", "A1:W6"],
  ["Deep Read View", "A1:G11"],
  ["Verification", "A1:L11"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewRoot, `${sheetName.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const summaryInspect = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:F12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 6,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "Phase 2 workbook formula error scan",
});
console.log(summaryInspect.ndjson);
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(phase2Root, "ieee_literature_evidence.xlsx"));
console.log(JSON.stringify({ workbook: path.join(phase2Root, "ieee_literature_evidence.xlsx"), previewRoot }));
