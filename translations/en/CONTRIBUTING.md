<a id="怎样把一份公开材料变成可学习的知识"></a>

# Turning public material into learnable knowledge

This project serves beginners and advancing practitioners who want to learn frontier LLMs, especially learners without access to a frontier lab's internal resources. Contributions should help readers understand end-to-end training and build or advance engineering skills; explain whether the contribution supports foundational entry or further advancement, what readers will deliver, and how completion will be assessed. See [ROADMAP](ROADMAP.md) for positioning and sequence.

<a id="新资源的最低信息"></a>

## Minimum information for a new resource

State the corresponding module, prerequisites, core question, chapters/files to read first, expected artifact, and which [coverage gap](COVERAGE.md) it fills. Provide an author or official project source, access date, and version, and distinguish the openness of papers, weights, training code, inference code, data, and run records. Recommendations containing only a name and URL remain candidates and do not enter core reading directly.

<a id="新-lesson-的结构"></a>

## Structure of a new lesson

Start with a concrete question and inputs/outputs, then explain the mechanism; provide a worked example, at least one incorrect example, source entry points, reader exercises, and answers or criteria for completion. Clearly state which parts can be completed on a CPU and which require a GPU. Explain terminology for Chinese-language readers and retain English names to support further source reading.

<a id="实验与研究结论"></a>

## Experiments and research conclusions

Use the [experiment/decision template](templates/research-decision.md) to record versions, data, environment, commands, metrics, raw results, and limitations. Do not report simulator speed as GPU performance or substitute training-set reward for independent capability; explain failures, discards, and budget changes. Keep large datasets, weights, and credentials out of the learning repository.

<a id="修改与验证"></a>

## Changes and validation

Preserve historical evidence and create new records for new versions; do not overwrite the basis of an old run with a dynamic main link. After changing the curriculum, update navigation, module relationships, the resource atlas, and coverage status together, avoiding two competing default learning sequences.

Run `python tools/validate_learning_repo.py` in a local repository with restored sources to check documentation and pinned references. Changes to runnable examples also require running the relevant examples and saving actual results that match the stated claims. After validation passes, commit through the existing Git workflow; this repository has no automatic scheduled fetching or background training.

When updating the Chinese source, review the English counterpart too, preserving fixed references, code, and evidence scope. Follow the [translation maintenance guide](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/translations/README.md) to update the translation record and validate it. Use the [bilingual terminology table](GLOSSARY.md) for English terms.
