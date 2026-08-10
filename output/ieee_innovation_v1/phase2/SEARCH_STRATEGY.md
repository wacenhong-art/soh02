# Literature Search Strategy - ARS Phase 2

## Parameters

- Databases/providers: Sider Scholar (Google Scholar + OpenAlex), Crossref, Semantic Scholar, IEEE Xplore/DOI links.
- Date range: 2024-08-10 to 2026-08-10, inclusive.
- Language: English.
- Document types: IEEE peer-reviewed journal and conference papers directly addressing battery SOH or a closely coupled battery-state data problem.
- Inclusion: verified IEEE DOI; date in window; title/abstract directly relevant; sufficient metadata for WHY/HOW/WHAT extraction.
- Exclusion: out of window; SOC-only without SOH relevance; non-IEEE; thesis/ResearchGate-only record; unverifiable DOI/title; sensor/data requirements that cannot be identified.

## Two-Pass Screening

1. Sider Scholar broad queries provided recall and exposed vocabulary/pain-point themes.
2. Crossref IEEE-member filtering and Semantic Scholar DOI/title/abstract checks supplied precision.

## Reproducibility Caveat

Sider Scholar ranking is dynamic and the first `site:` query returned zero. Saved result IDs and raw hit counts are recorded in `literature_search_log.csv`; the reproducible candidate corpus is generated from Crossref's IEEE member metadata.
