# Learning repository workflow

This is a Chinese-language, source-grounded public curriculum and resource handbook
for learners without access to frontier labs. Start with README.md, ROADMAP.md,
COVERAGE.md, PROGRESS.md, sources.lock.json and the current module or repository note.
The user wants detailed coverage, explicit prerequisites and learning order,
primary sources, practical exercises and connections between projects.

Canonical repository: https://github.com/yubol-bobo/frontier_llm_handbook.git.
Use this handbook as the main project for future work; the default remote is
origin and its primary branch is main. Keep new learning artifacts here, and use
origin/main when synchronizing them. Preserve upstream histories and do not force-push.
The independent repositories under sources/ retain their own upstream remotes;
publish their manifest and recorded commits, not nested copies of those repositories.

- Follow HOW_TO_STUDY.md; distinguish source facts, inference and executed results.
- ROADMAP.md is the canonical learning order (M00-M15 plus F01-F06 electives).
  Registry numbers and maintainer reading history are not learner prerequisites.
- Keep curriculum specifications, authored lessons, source reviews and executed
  experiments distinct. Do not mark a module completed because its guide exists.
- Every core module should name prerequisites, ordered reading, CPU-accessible
  exercises, GPU extensions where relevant, deliverables and exit criteria.
- Prefer upstream fixed-SHA links for public-facing source entry points; ignored
  sources/ paths alone are unusable to readers browsing the handbook on GitHub.
- Keep gaps and openness boundaries explicit. Public inference code or weights
  do not imply access to all training data or a closed lab's complete recipe.
- Preserve fixed-commit evidence for existing notes. Record new source revisions explicitly.
- Keep upstream checkouts under sources/ independent; avoid modifying them during reading.
- Add your own experiments under experiments/ and document their actual execution scope.
- Update the relevant note, relationship evidence, knowledge tree and PROGRESS.md after a study session.
- Existing GPU/cloud experiments are plans unless an execution record proves otherwise.
- Use python tools/validate_learning_repo.py to check sources and documentation links before delivery.
- Do not auto-pull all upstream projects or treat their independent HEADs as a compatible environment.
- The repository itself has no recurring automation; continue when the user requests further work.
