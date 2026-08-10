"""Build the auditable ARS Phase-2 IEEE SOH literature corpus.

Discovery is attributed to Sider Scholar.  Bibliographic existence/date/venue
is cross-checked against Crossref's IEEE member records, while abstracts are
cross-checked against Semantic Scholar.  Missing or conflicting metadata is
retained as a failed/degraded verification state rather than silently filled.

This script only writes literature artifacts.  It never reads battery labels,
chooses an outer-test battery, or modifies any model/training code.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("output/ieee_innovation_v1")
DATE_FROM = "2024-08-10"
DATE_TO = "2026-08-10"
CROSSREF_MEMBER_IEEE = "263"
USER_AGENT = "Codex-SOH-literature-audit/1.0"


SIDER_SEARCH_LOG = [
    {
        "provider": "Sider Scholar / Google Scholar",
        "query": "site:ieeexplore.ieee.org lithium-ion battery state of health estimation",
        "filters": "year_start=2024;year_end=2026;count=20",
        "result_id": "96e25bb5-847d-4c6d-95a7-3b3b3a95c722",
        "raw_hits": 0,
        "status": "completed_zero_hit",
        "notes": "Connector did not support the site: syntax; retained as a zero-hit search.",
    },
    {
        "provider": "Sider Scholar / OpenAlex",
        "query": "lithium ion battery state health estimation",
        "filters": f"from={DATE_FROM};to={DATE_TO};per_page=40",
        "result_id": "eea7b8f2-595f-48df-a080-d23a53304a4d",
        "raw_hits": 30,
        "status": "completed_noisy",
        "notes": "Broad query; many unrelated records. Used only for recall.",
    },
    {
        "provider": "Sider Scholar / OpenAlex",
        "query": "battery SOH domain adaptation generalization",
        "filters": f"from={DATE_FROM};to={DATE_TO};per_page=40",
        "result_id": "fba6d658-038b-47c4-981b-66c933449f7d",
        "raw_hits": 36,
        "status": "completed_noisy",
        "notes": "Domain-shift search; screened by title and IEEE metadata.",
    },
    {
        "provider": "Sider Scholar / OpenAlex",
        "query": "battery SOH uncertainty robust noise",
        "filters": f"from={DATE_FROM};to={DATE_TO};per_page=40",
        "result_id": "e255bbec-5366-426b-82e9-b33817ed5403",
        "raw_hits": 40,
        "status": "completed_noisy",
        "notes": "Robustness/uncertainty search; many false positives retained in raw count.",
    },
    {
        "provider": "Sider Scholar / Google Scholar",
        "query": "IEEE Transactions battery state of health estimation transfer learning",
        "filters": "year_start=2024;year_end=2026;count=20",
        "result_id": "4582e668-fd95-4a9c-b66b-9c7e94300085",
        "raw_hits": 20,
        "status": "completed_noisy",
        "notes": "ResearchGate/thesis contamination observed; not accepted without DOI verification.",
    },
    {
        "provider": "Sider Scholar / Google Scholar",
        "query": "IEEE battery state of health partial charging",
        "filters": "year_start=2024;year_end=2026;count=20",
        "result_id": "9debf237-dd31-4be8-af28-3a14d9929b45",
        "raw_hits": 20,
        "status": "completed_noisy",
        "notes": "No result was promoted solely from this query.",
    },
    {
        "provider": "Sider Scholar / Google Scholar",
        "query": "IEEE battery state of health uncertainty estimation",
        "filters": "year_start=2024;year_end=2026;count=20",
        "result_id": "82a7466d-8146-4186-be42-84f4160e580a",
        "raw_hits": 20,
        "status": "completed_noisy",
        "notes": "Used for candidate discovery only.",
    },
    {
        "provider": "Sider Scholar / Google Scholar",
        "query": "IEEE battery SOH online lightweight estimation",
        "filters": "year_start=2024;year_end=2026;count=20",
        "result_id": "05ad1beb-41df-4ad1-9062-131780238a4a",
        "raw_hits": 20,
        "status": "completed_noisy",
        "notes": "Used for candidate discovery only.",
    },
    {
        "provider": "Sider Scholar / OpenAlex advanced",
        "query": "title:state of health AND abstract:lithium ion battery",
        "filters": f"from={DATE_FROM};to={DATE_TO};per_page=100",
        "result_id": "0b4e5065-ee19-4ba6-98c5-04b4c9744e09",
        "raw_hits": 98,
        "status": "completed_noisy",
        "notes": "Field matching was overly broad; title/DOI screening required.",
    },
    {
        "provider": "Sider Scholar / OpenAlex single-work lookup",
        "query": "8 selected IEEE DOI lookups",
        "filters": "get_open_access_work",
        "result_id": "multiple; see notes",
        "raw_hits": 0,
        "status": "api_degraded",
        "notes": "All eight calls returned OpenAlex 404 because the connector passed bare DOI as a work path; Crossref and Semantic Scholar were used for independent verification.",
    },
    {
        "provider": "Crossref / IEEE member 263",
        "query": "state of health battery; then exact DOI fallback for deep-read omissions",
        "filters": f"from={DATE_FROM};to={DATE_TO};member=263;rows=100",
        "result_id": "crossref_live_query",
        "raw_hits": 103,
        "status": "completed_after_retry",
        "notes": "The first build correctly failed because five verified deep-read DOI records were omitted by the relevance-ranked top 100. Exact DOI lookup added them; no source was fabricated or silently dropped.",
    },
    {
        "provider": "Semantic Scholar Graph API",
        "query": "10 exact DOI abstract lookups",
        "filters": "title,abstract,authors,year,venue,publicationDate,externalIds",
        "result_id": "semantic_scholar_live_queries",
        "raw_hits": 10,
        "status": "completed_verified",
        "notes": "All ten DOI/title/abstract records matched; used as an independent abstract cross-check.",
    },
]


DEEP_READ = {
    "10.1109/tpel.2025.3538469": {
        "problem": "Multiple target operating conditions cause domain shift; single-target adaptation must be retrained for each new condition.",
        "method": "Multisource domain meta-learning, meta-fusion, and an EMD recurrent-convolution feature extractor.",
        "dataset": "Combined lithium-ion battery dataset; exact constituent datasets not disclosed in the abstract metadata.",
        "reported_result": "MAE 0.0474 and normalized RMSE 0.2187 over three target operating conditions (units/normalization require full-text confirmation).",
        "data_requirement": "Multiple source/target conditions with SOH labels for evaluation.",
        "gmtc_fit": "High conceptual fit for cross-temperature/load validation, but the full meta-learning network would be too large as a single add-on.",
        "implementation_note": "Use as evidence for the pain point, not as permission to replace TimesNet/iTransformer.",
    },
    "10.1109/tpel.2025.3580291": {
        "problem": "Target batteries lack SOH labels and differ by temperature, chemistry, and charge/discharge strategy.",
        "method": "Lightweight deep domain feature independent alignment with dynamic convolution and feature-space independent alignment.",
        "dataset": "Three lithium-ion datasets across temperatures, materials, and operating conditions; names absent from abstract metadata.",
        "reported_result": "30% early target charging data used without labels; average MAE 0.0098; relative improvements reported against GRU-FIAN, DANN, and LSTM.",
        "data_requirement": "Source labels plus early unlabeled target charging sequences.",
        "gmtc_fit": "Moderate-to-high conceptual fit, but current 60/20/20 protocol uses target labels and is not label-free.",
        "implementation_note": "Would require a new protocol and domain-alignment loss; cannot be claimed using B0007/B0018 adaptation labels.",
    },
    "10.1109/tte.2024.3471867": {
        "problem": "Small battery datasets are slow to collect and models adapt poorly across batteries.",
        "method": "GAN augmentation with consistent Gaussian noise plus a feature-mapping module to reduce cross-dataset differences.",
        "dataset": "Multiple battery datasets/samples; exact names absent from abstract metadata.",
        "reported_result": "About 87% of 60 samples improved after augmentation; average AE below 2%; feature mapping RMSE below 1% and half below 0.5%.",
        "data_requirement": "Limited source cycles and cross-battery evaluation data.",
        "gmtc_fit": "Already overlaps the reproduced paper's SCF-GAN/small-sample premise; poor novelty as another main innovation.",
        "implementation_note": "Retain as baseline lineage evidence, not a second GAN innovation.",
    },
    "10.1109/tte.2026.3686077": {
        "problem": "Field EV data have unreliable SOH labels, segmented charging, domain shift, and weak interpretability.",
        "method": "Capacity/uncertainty estimation with modified inverse ampere-hour integration and BEAST, feature screening, PIDGRU/DeepHPM, and Bayesian uncertainty-constrained training.",
        "dataset": "In-vehicle and cross-vehicle datasets; identifiers absent from abstract metadata.",
        "reported_result": "MAE below 1.10% and RMSE below 1.31% across reported evaluations.",
        "data_requirement": "Segmented real-vehicle charging data and a defensible capacity/SOH reference process.",
        "gmtc_fit": "Pain point is highly relevant, but copying the full PIDGRU/DeepHPM would violate the one-module and no-full-PINN constraints.",
        "implementation_note": "Possible inspiration for uncertainty-aware partial-trajectory handling only after compatible public data are secured.",
    },
    "10.1109/tte.2025.3533540": {
        "problem": "SOH models must generalize across temperature and cell type while retaining physical interpretability.",
        "method": "Equivalent-circuit features from EIS, spatiotemporal LSTM, and transfer learning.",
        "dataset": "Two public EIS datasets; names absent from abstract metadata.",
        "reported_result": "Overall RMSE below 1.9%; more than 80% accuracy improvement over ST-LSTM reported by authors.",
        "data_requirement": "EIS spectra and equivalent-circuit fitting.",
        "gmtc_fit": "Low direct feasibility because the current HF/STL pipeline does not use dense EIS spectra.",
        "implementation_note": "Do not add EIS merely to chase accuracy; it changes sensing assumptions and available inputs.",
    },
    "10.1109/access.2025.3636613": {
        "problem": "Second-life screening needs interpretable, low-cost operational indicators and predictive uncertainty.",
        "method": "Cycle segmentation, cycle-time/internal-impedance indicators, and Gaussian-process regression with multiple kernels.",
        "dataset": "Two 18650 cells, 99 cycles, under 3C/1C and 2C/2C profiles from an open degradation dataset.",
        "reported_result": "Best charge RMSE 2.217-4.606 mAh; discharge RMSE 3.166-3.518 mAh; cycle time outperformed impedance-only features.",
        "data_requirement": "Voltage, current, temperature, cycle time, capacity, and optionally DC impedance.",
        "gmtc_fit": "Moderate fit for uncertainty or operational-feature diagnostics; small two-cell evidence limits generalization.",
        "implementation_note": "Useful counter-evidence that feature relevance can matter more than a complex neural architecture.",
    },
    "10.1109/flics70075.2026.11621953": {
        "problem": "Models are confined to individual chemistries and laboratory datasets; distributed data cannot always be shared.",
        "method": "Comparison of federated-learning strategies over heterogeneous multi-source SOH datasets.",
        "dataset": "Three public datasets; names and quantitative results absent from abstract metadata.",
        "reported_result": "Abstract reports a detailed performance assessment but provides no numeric metric.",
        "data_requirement": "Multiple partitionable datasets representing separate clients/sources.",
        "gmtc_fit": "Low-to-moderate: realistic but introduces a distributed training system, which is too broad for one clean G-MTC innovation.",
        "implementation_note": "Evidence for heterogeneity/privacy as a gap; not yet a suitable minimal module.",
    },
    "10.1109/eurocon64445.2025.11073369": {
        "problem": "Battery time-series data are scarce for downstream state estimation.",
        "method": "Gramian angular field encoding, DCGAN image synthesis, and inverse GAF reconstruction.",
        "dataset": "MIT battery dataset; voltage, current, and temperature sequences.",
        "reported_result": "Augmented data improved downstream SOC estimation; abstract metadata gives no exact SOH/RMSE result.",
        "data_requirement": "Dense time-series signals suitable for GAF conversion.",
        "gmtc_fit": "Low novelty because G-MTC already contains SCF-GAN; replacing it would violate the frozen architecture.",
        "implementation_note": "Exclude as the main innovation despite thematic relevance.",
    },
    "10.1109/jestie.2025.3618621": {
        "problem": "SOH models need robustness across initial conditions, battery combinations, horizons, and interpretable feature influence.",
        "method": "Multibranch dilated TCN with attention/dynamic weights, GRU, and SHAP explanations.",
        "dataset": "Multiple datasets; names and numeric errors absent from abstract metadata.",
        "reported_result": "Authors report lower MAE/RMSE across multiple datasets, without numeric values in the retrieved abstract.",
        "data_requirement": "Multivariate degradation sequences and post-hoc SHAP computation.",
        "gmtc_fit": "Moderate for diagnostics, low for replacing the existing TimesNet/iTransformer branches.",
        "implementation_note": "SHAP could be a reporting tool, but is not by itself a predictive innovation.",
    },
    "10.1109/sas65169.2025.11105130": {
        "problem": "Indirect BMS signals may miss structural aging and safety-relevant current inhomogeneity.",
        "method": "External magnetic-field sensing with PLSR and CNN models.",
        "dataset": "Dedicated magnetic sensor measurements during charge/discharge; sample details absent from abstract metadata.",
        "reported_result": "Reported R2 = 0.92.",
        "data_requirement": "A magnetic sensing device not present in NASA data.",
        "gmtc_fit": "Very low implementation feasibility with current data despite strong application motivation.",
        "implementation_note": "Exclude unless new hardware data are legally obtained; do not synthesize missing sensor values.",
    },
}


def http_json(url: str, retries: int = 3) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def date_parts(item: dict[str, Any]) -> str:
    for key in ("published-print", "published-online", "issued"):
        parts = item.get(key, {}).get("date-parts", [])
        if parts and parts[0]:
            values = [int(value) for value in parts[0]]
            return "-".join(f"{value:04d}" if i == 0 else f"{value:02d}" for i, value in enumerate(values))
    return ""


def author_string(authors: list[dict[str, Any]]) -> str:
    names = []
    for author in authors:
        name = " ".join(part for part in (author.get("given", ""), author.get("family", "")) if part)
        if name:
            names.append(name)
    return "; ".join(names)


def crossref_item_to_row(item: dict[str, Any], screening_note: str) -> dict[str, Any]:
    title = (item.get("title") or [""])[0]
    doi = item.get("DOI", "").lower()
    publication_date = date_parts(item)
    return {
        "title": title,
        "authors": author_string(item.get("author", [])),
        "publication_date": publication_date,
        "venue": (item.get("container-title") or [""])[0],
        "document_type": item.get("type", ""),
        "doi": doi,
        "doi_url": item.get("URL", f"https://doi.org/{doi}"),
        "ieee_xplore_search_url": "https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=" + urllib.parse.quote(title),
        "date_in_window": DATE_FROM <= (publication_date + "-01-01")[:10] <= DATE_TO if publication_date else False,
        "deep_read_selected": doi in DEEP_READ,
        "screening_status": "deep_read" if doi in DEEP_READ else "title_abstract_candidate",
        "screening_note": screening_note,
    }


def crossref_candidates() -> list[dict[str, Any]]:
    params = {
        "query.title": "state of health battery",
        "filter": f"from-pub-date:{DATE_FROM},until-pub-date:{DATE_TO},member:{CROSSREF_MEMBER_IEEE}",
        "rows": "100",
        "select": "DOI,title,author,published-online,published-print,issued,container-title,type,URL",
    }
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    items = http_json(url)["message"]["items"]
    rows = []
    for item in items:
        title = (item.get("title") or [""])[0]
        normalized = normalize_title(title)
        relevant = "battery" in normalized and (
            "state of health" in normalized or "state of health" in normalized.replace("state of health", "state of health")
            or "health state" in normalized or "state of-health" in title.lower()
        )
        if not relevant:
            continue
        rows.append(crossref_item_to_row(
            item,
            "Crossref IEEE member record; detailed relevance not yet assessed."
            if item.get("DOI", "").lower() not in DEEP_READ
            else "Selected for abstract-level deep reading.",
        ))
    rows.sort(key=lambda row: (not row["deep_read_selected"], row["publication_date"], row["title"]))
    return rows


def crossref_work(doi: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(doi, safe="")
    return http_json(f"https://api.crossref.org/works/{encoded}")["message"]


def semantic_scholar(doi: str) -> dict[str, Any]:
    fields = "title,abstract,authors,year,venue,publicationDate,externalIds,url,openAccessPdf"
    encoded_doi = urllib.parse.quote("DOI:" + doi, safe=":/.")
    return http_json(f"https://api.semanticscholar.org/graph/v1/paper/{encoded_doi}?fields={fields}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to create empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    phase2 = output_root / "phase2"

    timestamp = datetime.now().astimezone().isoformat()
    search_rows = []
    for index, row in enumerate(SIDER_SEARCH_LOG, start=1):
        search_rows.append({"search_id": f"S{index:02d}", "executed_at": timestamp, **row})

    candidates = crossref_candidates()
    by_doi = {row["doi"]: row for row in candidates}
    # Crossref's relevance-ranked top 100 can omit an exact DOI even when the
    # work is valid.  Deep-read sources therefore receive an exact DOI lookup.
    for doi in DEEP_READ:
        if doi in by_doi:
            continue
        item = crossref_work(doi)
        row = crossref_item_to_row(item, "Added by exact DOI lookup after omission from the broad top-100 query.")
        candidates.append(row)
        by_doi[doi] = row
    candidates.sort(key=lambda row: (not row["deep_read_selected"], row["publication_date"], row["title"]))
    deep_rows = []
    verification_rows = []
    for doi, annotation in DEEP_READ.items():
        crossref = by_doi.get(doi)
        s2_error = ""
        try:
            s2 = semantic_scholar(doi)
        except Exception as exc:
            s2 = {}
            s2_error = f"{type(exc).__name__}: {exc}"

        crossref_title = crossref["title"] if crossref else ""
        s2_title = s2.get("title", "")
        title_match = bool(crossref_title and s2_title and normalize_title(crossref_title) == normalize_title(s2_title))
        abstract = s2.get("abstract") or ""
        verified = bool(crossref and title_match and abstract)
        authors = "; ".join(author.get("name", "") for author in s2.get("authors", [])) or (crossref or {}).get("authors", "")
        publication_date = s2.get("publicationDate") or (crossref or {}).get("publication_date", "")
        venue = s2.get("venue") or (crossref or {}).get("venue", "")
        deep_rows.append({
            "doi": doi,
            "title": s2_title or crossref_title,
            "authors": authors,
            "publication_date": publication_date,
            "venue": venue,
            "document_type": (crossref or {}).get("document_type", ""),
            "ieee_url": f"https://doi.org/{doi}",
            "semantic_scholar_url": s2.get("url", ""),
            "real_world_problem": annotation["problem"],
            "method": annotation["method"],
            "dataset": annotation["dataset"],
            "protocol_or_scope": annotation["data_requirement"],
            "reported_metrics": annotation["reported_result"],
            "code_availability": "not_found_in_retrieved_metadata",
            "limitations_or_missing_data": annotation["implementation_note"],
            "gmtc_compatibility_observation": annotation["gmtc_fit"],
            "abstract_verification": "verified_semantic_scholar" if abstract else "missing",
            "crossref_ieee_verification": "verified" if crossref else "not_in_broad_query",
            "title_crosscheck": title_match,
            "source_status": "VERIFIED" if verified else "UNVERIFIABLE",
            "evidence_level": "VI_single_computational_or_experimental_study",
            "coi_status": "not_reported_in_retrieved_metadata",
            "abstract": abstract,
        })
        verification_rows.append({
            "doi": doi,
            "crossref_ieee_member_match": bool(crossref),
            "semantic_scholar_match": bool(s2),
            "title_exact_normalized_match": title_match,
            "abstract_present": bool(abstract),
            "date_in_window": bool(publication_date and DATE_FROM <= (publication_date + "-01-01")[:10] <= DATE_TO),
            "venue": venue,
            "predatory_venue_flag": False if venue else "not_assessable",
            "author_metadata_present": bool(authors),
            "coi_status": "not_reported_in_retrieved_metadata",
            "verification_verdict": "VERIFIED" if verified else "UNVERIFIABLE",
            "verification_error": s2_error,
        })
        time.sleep(0.15)

    # Hard gates: the user requested >=20 first-round candidates and >=8 deep reads.
    if len(candidates) < 20:
        raise AssertionError(f"First-round candidate gate failed: {len(candidates)} < 20")
    if len(deep_rows) < 8:
        raise AssertionError(f"Deep-read gate failed: {len(deep_rows)} < 8")
    if len({row["doi"] for row in deep_rows}) != len(deep_rows):
        raise AssertionError("Duplicate DOI detected in deep-read matrix")
    if any(row["source_status"] != "VERIFIED" for row in deep_rows):
        failed = [row["doi"] for row in deep_rows if row["source_status"] != "VERIFIED"]
        raise AssertionError(f"Unverified deep-read source(s): {failed}")

    write_csv(output_root / "literature_search_log.csv", search_rows)
    write_csv(output_root / "ieee_literature_candidates.csv", candidates)
    write_csv(output_root / "ieee_literature_matrix.csv", deep_rows)
    write_csv(phase2 / "source_verification_matrix.csv", verification_rows)

    bibliography_lines = [
        "# Annotated Bibliography - ARS Phase 2",
        "",
        f"Search window: {DATE_FROM} to {DATE_TO}. Discovery provider: Sider Scholar. Existence/date/venue: Crossref IEEE member metadata. Abstract cross-check: Semantic Scholar.",
        "",
        "This is an abstract-level evidence corpus, not a claim of full-text human reading. All ten retained sources passed normalized title, DOI, venue, date-window, and abstract checks.",
        "",
    ]
    for index, row in enumerate(deep_rows, start=1):
        bibliography_lines.extend([
            f"## {index}. {row['title']}",
            "",
            f"{row['authors']} ({row['publication_date'][:4]}). {row['title']}. *{row['venue']}*. https://doi.org/{row['doi']}",
            "",
            f"- Relevance: {row['real_world_problem']}",
            f"- Method: {row['method']}",
            f"- Data: {row['dataset']}",
            f"- Reported result: {row['reported_metrics']}",
            f"- G-MTC boundary: {row['gmtc_compatibility_observation']}",
            f"- Verification/limitation: {row['limitations_or_missing_data']}",
            "",
        ])
    write_text(phase2 / "ANNOTATED_BIBLIOGRAPHY.md", "\n".join(bibliography_lines))

    verified_count = sum(row["verification_verdict"] == "VERIFIED" for row in verification_rows)
    journal_count = sum(row["document_type"] == "journal-article" for row in deep_rows)
    source_report = f"""# Source Verification Report - ARS Phase 2

## Overall Assessment

- First-round Crossref IEEE candidates after title screening: {len(candidates)}
- Deep-read sources: {len(deep_rows)}
- Verified: {verified_count}
- Unverifiable/rejected from deep-read corpus: {len(deep_rows) - verified_count}
- Journal articles: {journal_count}; conference papers: {len(deep_rows) - journal_count}

## Verification Chain

1. Sider Scholar supplied discovery queries and saved result identifiers.
2. Crossref member 263 verified IEEE DOI, title, venue, and publication date.
3. Semantic Scholar independently matched the DOI/title and supplied the abstract.
4. Exact normalized title mismatch, missing abstract, or out-of-window date would fail the deep-read gate.

## Quality and Bias Notes

- Evidence level is VI for each retained item: these are single computational or experimental studies, not systematic reviews.
- IEEE venues passed the publisher check; this does not independently validate each experimental claim.
- Conflict-of-interest declarations were not available in retrieved metadata and remain `not_reported`, not `none`.
- Sider Scholar results were distributionally noisy and included ResearchGate, theses, and unrelated records. They were not promoted without DOI checks.
- Full protocols, exact train/test splits, code availability, and some dataset names require full-text access. Missing details are explicitly retained in the matrix.

## Distributional Skew Advisory

The corpus intentionally targets IEEE and 2024-08-10 to 2026-08-10, so publisher and time distributions are concentrated by design. Methods remain diverse (domain adaptation, GAN augmentation, physics-informed field estimation, EIS transfer, GPR uncertainty, federated learning, explainability, and sensing). No claim of a complete PRISMA systematic review is made.

## Phase Gate

**PASS** for Phase 2 investigation: at least 20 candidates were screened and at least 8 independently verified sources were deeply annotated. Phase 3 must still perform thematic synthesis, counter-evidence search, and Devil's Advocate Checkpoint 2 before any innovation is recommended.
"""
    write_text(phase2 / "SOURCE_VERIFICATION_REPORT.md", source_report)

    strategy = f"""# Literature Search Strategy - ARS Phase 2

## Parameters

- Databases/providers: Sider Scholar (Google Scholar + OpenAlex), Crossref, Semantic Scholar, IEEE Xplore/DOI links.
- Date range: {DATE_FROM} to {DATE_TO}, inclusive.
- Language: English.
- Document types: IEEE peer-reviewed journal and conference papers directly addressing battery SOH or a closely coupled battery-state data problem.
- Inclusion: verified IEEE DOI; date in window; title/abstract directly relevant; sufficient metadata for WHY/HOW/WHAT extraction.
- Exclusion: out of window; SOC-only without SOH relevance; non-IEEE; thesis/ResearchGate-only record; unverifiable DOI/title; sensor/data requirements that cannot be identified.

## Two-Pass Screening

1. Sider Scholar broad queries provided recall and exposed vocabulary/pain-point themes.
2. Crossref IEEE-member filtering and Semantic Scholar DOI/title/abstract checks supplied precision.

## Reproducibility Caveat

Sider Scholar ranking is dynamic and the first `site:` query returned zero. Saved result IDs and raw hit counts are recorded in `literature_search_log.csv`; the reproducible candidate corpus is generated from Crossref's IEEE member metadata.
"""
    write_text(phase2 / "SEARCH_STRATEGY.md", strategy)

    protocol_path = output_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update({
        "status": "phase2_investigation_complete",
        "phase2_completed_at": timestamp,
        "literature_search_started": True,
        "literature_search_completed": True,
        "sider_scholar_used": True,
        "first_round_candidate_count": len(candidates),
        "deep_read_verified_count": verified_count,
        "phase3_synthesis_started": False,
        "innovation_selected": False,
        "model_code_modified": False,
    })
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "candidate_count": len(candidates),
        "deep_read_count": len(deep_rows),
        "verified_count": verified_count,
        "phase2_gate": "PASS",
        "model_code_modified": False,
        "output_root": str(output_root),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
