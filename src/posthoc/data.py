def build_eval_loader(sr_cfg, posthoc_cfg, split):
    from sr.data import build_dataloader
    from sr.splits import ensure_synthetic_split
    from sr.utils.config import merge_dict

    if sr_cfg.data.name == "synthetic_microscopy":
        ensure_synthetic_split(sr_cfg)
    data_cfg = merge_dict(sr_cfg.data, sr_cfg.get(f"{split}_data", {}))
    data_cfg.return_pair = True
    data_cfg.return_labels = True
    data_cfg.return_metadata = True
    if data_cfg.name == "synthetic_microscopy":
        data_cfg.return_masks = True
    batch_size = int(posthoc_cfg.get("sr", {}).get("batch_size", 4))
    num_workers = int(posthoc_cfg.get("data", {}).get("num_workers", 0))
    return build_dataloader(data_cfg, split=split, shuffle=False, batch_size=batch_size, num_workers=num_workers)
