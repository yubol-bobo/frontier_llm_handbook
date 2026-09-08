# Translation maintenance / 双语维护

The website supports Chinese and English for its interface and all 51 published documents. The Chinese documents at their original paths remain canonical. Their English counterparts live in `en/`, with the same relative filenames. Read the [English website](https://yubol-bobo.github.io/frontier_llm_handbook/?lang=en#/learn) or [English Markdown](en/README.md).

## Updating a translation

1. Update the canonical Chinese source and its English counterpart together. Preserve the section structure, module IDs, examples, executable code, fixed source SHAs, and evidence scope. Do not turn a translation into a summary.
2. Use the bilingual term table in [GLOSSARY](../GLOSSARY.md). Paper titles, project names, API names, variables, and executable code keep their original spelling. Chinese descriptions inside prose diagrams can be translated.
3. Preserve existing explicit anchors in the English Markdown. Chinese heading aliases keep previously shared fragments usable. Links to translated documents are relative; links to nontranslated assets or upstream code point to canonical GitHub URLs.
4. After reviewing the English text against the changed source, update that file's SHA-256 entry in `en/manifest.json`. Hash the source as UTF-8 after normalizing CRLF to LF. The build fails if a source changed without a corresponding translation review.
5. Run `npm run check`. It checks both corpora, document links, heading aliases, reference and code preservation, interface translation coverage, personal-data boundaries, and mathematical consistency. With restored upstream clones, also run `python tools/validate_learning_repo.py`.

For a complete set of independently reviewed drafts using the original source-relative links, the maintainer can run `node tools/import_english_drafts.mjs <draft-directory> [...]`. This checks structural fidelity before writing translations and records source hashes. It also generates heading aliases and canonical URLs for nontranslated assets. Do not run it over unreviewed drafts or use it to bypass the source-review check.

## Interface and personal data

`website/i18n.mjs` localizes authored strings and literal template segments using `website/locales/en-ui.json`. Interpolated personal notes, search queries, code identifiers, record keys, and status values are not translated. `en-content.json` contains translations of stage and connection metadata; curriculum text comes from Markdown.

An explicit `?lang=zh` or `?lang=en` URL takes precedence over the saved language preference. Both languages share the same personal learning records. The language preference uses a separate localStorage key. No third-party translation service or user-data upload is involved.

译文与中文原稿同步维护；翻译完整性不等于已执行模型训练或获得外部认证。英文术语采用一手资料中的标准写法，项目特有语义以固定源码为准。
