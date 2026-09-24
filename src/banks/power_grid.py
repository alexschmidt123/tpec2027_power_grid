from pathlib import Path
from src.config import SBOEDConfig

def resolve_dataset_dir(cfg: SBOEDConfig, project_root: Path | None=None) -> Path:
    """SIR dataset directory."""
    from src.banks.paths import resolve_shared_data_dir
    return resolve_shared_data_dir(project_root, cfg)

def system_name_from_cfg(cfg: SBOEDConfig) -> str:
    sys_sec = cfg.raw.get('system') or {}
    if isinstance(sys_sec, dict) and sys_sec.get('name'):
        return str(sys_sec['name'])
    return str(cfg.topology)
