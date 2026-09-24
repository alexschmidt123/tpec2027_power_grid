# Sources and verification

[Back to overview](../README.md)

| File | Meaning |
|---|---|
| [source_map.json](source_map.json) | Original LabPC/HPRC run locations |
| [copy_manifest.json](copy_manifest.json) | Original file hashes and their current package paths |
| [verification.json](verification.json) | 45 evaluation sets checked against per-system records; checkpoints loaded successfully |
| [layout_verification.json](layout_verification.json) | Original file integrity and documentation-link checks after reorganization |
| [SHA256SUMS](SHA256SUMS) | Checksums for the complete current package; paths relative to package root |

Original configurations and model/result files remain byte-identical. Package index paths were updated for this layout. Historical hardware/source-revision information is incomplete; no missing provenance has been invented.

T=3 seed101 uses completed run 09032026_121824. T=4 seed303 uses the complete HPRC 09062026_221336 rerun. The source map identifies all nine runs.
