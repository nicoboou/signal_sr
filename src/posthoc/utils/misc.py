import torch


def safe_tag(value):
    text = str(value).replace("-", "m").replace(".", "p")
    return "".join(char if char.isalnum() or char in {"_", "-", "p", "m"} else "_" for char in text)


def choose_device(device):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device {device}, but CUDA is unavailable. Falling back to CPU.")
        return "cpu"
    return str(device)


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value
