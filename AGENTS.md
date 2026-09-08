# Learning repository workflow

This is a Chinese-language, source-grounded learning notebook. Start with README.md,
PROGRESS.md, sources.lock.json and the current repository note. The user wants
progressive study, practical experiments and relationships between projects.

Canonical repository: https://github.com/yubol-bobo/frontier_llm_handbook.git.
Use this handbook as the main project for future work; the default remote is
origin and its primary branch is main. Keep new learning artifacts here, and use
origin/main when synchronizing them. Preserve upstream histories and do not force-push.
The independent repositories under sources/ retain their own upstream remotes;
publish their manifest and recorded commits, not nested copies of those repositories.

- Follow HOW_TO_STUDY.md; distinguish source facts, inference and executed results.
- Preserve fixed-commit evidence for existing notes. Record new source revisions explicitly.
- Keep upstream checkouts under sources/ independent; avoid modifying them during reading.
- Add your own experiments under experiments/ and document their actual execution scope.
- Update the relevant note, relationship evidence, knowledge tree and PROGRESS.md after a study session.
- Existing GPU/cloud experiments are plans unless an execution record proves otherwise.
- Use python tools/validate_learning_repo.py to check sources and documentation links before delivery.
- Do not auto-pull all upstream projects or treat their independent HEADs as a compatible environment.
- The repository itself has no recurring automation; continue when the user requests further work.
