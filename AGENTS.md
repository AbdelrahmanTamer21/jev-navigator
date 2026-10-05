# jev-navigator agent rules

JVN is a library of building blocks for searching code. These rules are always loaded; the detail, and
which blocks are built or being built, lives in the README's
[Architecture](README.md#architecture-blocks-mini-workflows-and-configurations) section.

- **Levels.** Code primitives compose into mini-workflows (`find`, `find_all`, `trace`, `find_text`,
  `find_all_text`); mini-workflows compose into larger workflows through typed configurations.
- **Default.** `jvn search` is the default configuration; every other use is another named configuration.
- **Optional steps.** A Jev step (a bounded decision) or an LLM step (generation over an open space) can
  sit between stages. Which steps run is configuration, a recipe the caller passes as data; never an
  environment or deploy flag.
- **Blocks, not copies.** A capability that is not about one caller's domain is a block any caller can
  use. Callers configure blocks; they never reimplement one.
- **No caller domain.** Nothing finding-, theme- or Engine-specific lives in JVN.
- **Experiments** compare named configurations, never tweaks inside one call.
- **Checks (André, 05.10.2026).** Locally run only the tests that cover or import changed files and ruff;
  never the whole suite on this Mac. The full suite runs on GitHub Actions (`gh workflow run tests.yml`
  on the branch) once the head is the one to merge, judged by its log. Detail: [Tests](README.md#tests).
