# Public release provenance

`public_release_manifest.json` lists hashes for the current public files and targets for relative navigation links. Run `python3 tools/verify_release.py` to verify integrity and navigation; `--checkpoints` also loads the retained grid checkpoints strictly on CPU.

Repository cleanup preserves the reviewed manuscript and bibliography, complete probing histories, retained model checkpoints, and numerical results. Machine-specific paths in public JSON metadata have been converted to relative/archive labels without changing numerical values. Internal notes, unrelated SIR records, obsolete diagnostics, scheduler logs, transfer receipts, and incomplete duplicates were moved to a private archive outside this repository, with a complete pre-cleanup working-tree backup and Git-history bundle.

This inventory covers the current public tree. It does not certify that earlier Git commits contain no historical metadata. Cleanup does not rewrite Git history or push changes. The scan checks selected credential formats; it is not a universal guarantee against every possible secret format.
