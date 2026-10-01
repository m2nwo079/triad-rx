# 1단계 범위 조사 보고서

이 문서는 `scripts/scope/06_render_report.py`가 결과 파일과 설정 파일을 읽어 자동 생성합니다. 직접 수정하지 마세요.

| 항목 | 값 |
|---|---|
| 결과 파일 | `results/scope_survey_2026-10-01.json` |
| 스냅샷 날짜 | 2026-10-01 |
| 결과를 만든 코드 | `c891afa0d83d5e1b49c29f9caddce2fa0a694dcc` |
| 설계서 버전 | v0.3 |

## 1. 원본 데이터

| 파일 | SHA-256 |
|---|---|
| `methods.parquet` | `85c28648f762d79565a878ccaf08b5757782d4de4876446cb06de1ada3ae4abf` |
| `links.parquet` | `c845ff642112a345f6271bf954d007c6698088d7de525f3f8d5f7d975f142c76` |
| `pwa_0.parquet` | `766ec93cbc01de98519801886e98d5b2d9005a9f4e3dc3e60d1319a69383cda0` |
| `pwa_1.parquet` | `3c8e9c492f8ef5bbd7eb51e67918627b2b59ae7b7ecb4ad2e5c2f5837a263dfd` |
| `pwa_2.parquet` | `2b94e7b4bdea3a8b754a40edaf4899a4f308f9b043e7651ae25246bbb5a26302` |
| `pwa_3.parquet` | `0a2252584e69d93f7e0a3d2fbf62cec7a7b57c9a7f5781fd4a13f9881ea60af4` |


## 2. arXiv 연도별 논문 수

대상 카테고리 합산 기준입니다. 마지막 연도는 스냅샷 날짜까지의 값입니다.

| 연도 | 논문 수 | PWC arXiv 논문 수 |
|---|---|---|
| 2014 | 4,541 | 4,834 |
| 2015 | 6,309 | 8,121 |
| 2016 | 9,922 | 11,869 |
| 2017 | 14,696 | 16,570 |
| 2018 | 22,533 | 26,062 |
| 2019 | 32,601 | 36,939 |
| 2020 | 43,699 | 51,230 |
| 2021 | 48,302 | 57,514 |
| 2022 | 53,342 | 62,891 |
| 2023 | 66,911 | 77,533 |
| 2024 | 86,667 | 98,799 |
| 2025 | 106,859 | 51,798 |
| 2026 | 103,216 | — |


## 3. PWC 어휘

| 항목 | 값 |
|---|---|
| `methods_rows` | 8725 |
| `methods_num_papers_ge10` | 695 |
| `methods_spam_like_rows` | 2902 |
| `methods_introduced_year_share_2000` | 0.995 |
| `tasks_unique_in_annotations` | 4796 |
| `tasks_ge10_papers` | 2578 |
| `tasks_ge50_papers` | 1433 |
| `methods_unique_in_annotations` | 14827 |
| `methods_ge10_papers_in_annotations` | 896 |


## 4. 논문당 용어 수

### 4.1 PWC 주석 기준

| 연도 | 논문 수 | 평균 용어 수 | 3개 이상 비율 | 0개 비율 |
|---|---|---|---|---|
| 2016 | 16,275 | 2.012 | 0.308 | 0.275 |
| 2018 | 31,527 | 2.633 | 0.394 | 0.208 |
| 2020 | 59,014 | 3.224 | 0.413 | 0.199 |
| 2022 | 68,562 | 3.499 | 0.436 | 0.175 |
| 2024 | 100,322 | 3.855 | 0.482 | 0.142 |
| 2025 | 52,967 | 3.397 | 0.454 | 0.153 |


### 4.2 사전 매칭 기준 (탐색용 엄격 규칙)

| 연도 | 표본 | 평균 용어 수 | 3개 이상 비율 | 0개 비율 | 제거된 일반어 수 | 사용 어휘 수 |
|---|---|---|---|---|---|---|
| 2016 | 10,000 | 1.003 | 0.105 | 0.421 | 13 | 1016 |
| 2020 | 10,000 | 1.375 | 0.179 | 0.290 | 15 | 1332 |
| 2024 | 10,000 | 1.495 | 0.204 | 0.261 | 18 | 1325 |


## 5. PWC 논문–코드 연결 비율

| 연도 | 비율 |
|---|---|
| 2016 | 0.132 |
| 2017 | 0.203 |
| 2018 | 0.232 |
| 2019 | 0.267 |
| 2020 | 0.282 |
| 2021 | 0.285 |
| 2022 | 0.318 |
| 2023 | 0.34 |
| 2024 | 0.318 |
| 2025 | 0.252 |


## 6. 외부 문서에서 확인한 사실

확인 날짜: 2026-10-01

### 6.1 특허 (USPTO Open Data Portal)

- 상태: PatentsView migrated to USPTO Open Data Portal on 2026-03-20
- 접근: ODP requires registration and sign-in since 2026-06-18
- 같은 파일의 연간 다운로드 한도(API 키당): 20
- 테이블 릴리스: 2026-04-10

| 테이블 | 압축 크기(바이트) |
|---|---|
| `g_patent` | 232,076,712 |
| `g_patent_abstract` | 1,708,330,746 |
| `g_cpc_at_issue` | 338,958,464 |
| `g_assignee_disambiguated` | 362,741,666 |
| `pg_published_application` | 301,006,689 |
| `pg_published_application_abstract` | 1,597,376,724 |
| `pg_cpc_at_issue` | 202,652,269 |
| `pg_assignee_disambiguated` | 122,366,357 |
| `pg_granted_pgpubs_crosswalk` | 112,671,324 |


### 6.2 Gemini 무료 등급

프로젝트: Default Gemini Project

| 모델 | RPM | TPM | RPD |
|---|---|---|---|
| `gemini-3.5-flash-lite` | 15 | 250,000 | 500 |
| `gemini-3.1-flash-lite` | 15 | 250,000 | 500 |
| `gemini-2.5-flash-lite` | 10 | 250,000 | 20 |
| `gemini-3.8-flash` | 5 | 250,000 | 20 |
| `gemini-3.7-flash` | 5 | 250,000 | 20 |
| `gemini-3.6-flash` | 5 | 250,000 | 20 |
| `gemini-3.5-flash` | 5 | 250,000 | 20 |
| `gemini-3-flash` | 5 | 250,000 | 20 |
| `gemini-2.5-flash` | 5 | 250,000 | 20 |
| `gemma-4-26b` | 30 | 16,000 | 14,400 |
| `gemma-4-31b` | 30 | 16,000 | 14,400 |


### 6.3 기타

- pypistats 이력 기간(일): 186
- GitHub 검색(분당, 인증/비인증): 30 / 10

## 7. 확정된 시점 배치

설정 파일 `config/study.json`의 값입니다. 근거는 설계서 10.2절과 연구 과정 기록을 참고하세요.

| 시점 | 용도 | 구축 구간 | 버퍼 | 레이블 구간 |
|---|---|---|---|---|
| T1 | train | 2014–2018 | 2019 | 2020–2021 |
| T2 | evaluate | 2017–2021 | 2022 | 2023–2024 |


적용 모드는 스냅샷 날짜 기준 최근 5년을 구축 구간으로 씁니다.

## 8. 자동 점검

| 점검 | 결과 |
|---|---|
| arXiv: no missing years | 통과 |
| arXiv: all timepoint years present | 통과 |
| Result records a source commit | 통과 |
| Timepoints: buffer between build and label | 통과 |

