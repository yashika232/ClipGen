"""
File and directory utilities for the video synthesis pipeline.
"""

import os
import shutil
from pathlib import Path
from typing import Union, List, Optional
import hashlib
import json

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object of the directory
    """
    path_obj = Path(path)
    
    # Check if it's a file that exists
    if path_obj.exists() and path_obj.is_file():
        raise ValueError(f"Path exists as a file, not a directory: {path}")
    
    # Create directory if it doesn't exist
    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)
    
    return path_obj

def check_file_exists(path: Union[str, Path]) -> bool:
    """
    Check if a file exists and is readable.
    
    Args:
        path: File path to check
        
    Returns:
        True if file exists and is readable, False otherwise
    """
    try:
        path_obj = Path(path)
        return path_obj.is_file() and os.access(path_obj, os.R_OK)
    except Exception:
        return False

def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of the file hash
        
    Raises:
        ValueError: If algorithm is not supported
        FileNotFoundError: If file doesn't exist
    """
    if algorithm not in ['md5', 'sha1', 'sha256']:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    if not check_file_exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()

def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            return False
        
        # Ensure destination directory exists
        ensure_directory(dst_path.parent)
        
        shutil.copy2(src_path, dst_path)
        return True
    except Exception:
        return False

def load_json(file_path: Union[str, Path]) -> dict:
    """
    Load JSON data from file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    if not check_file_exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: dict, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        ensure_directory(path.parent)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception:
        return False

def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return path.stat().st_size

def list_files_with_extension(directory: Union[str, Path], extension: str) -> List[Path]:
    """
    List all files with a specific extension in a directory.
    
    Args:
        directory: Directory to search
        extension: File extension (e.g., '.mp4', '.wav')
        
    Returns:
        List of file paths with the specified extension
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    # Ensure extension starts with dot
    if not extension.startswith('.'):
        extension = '.' + extension
    
    return list(dir_path.glob(f"*{extension}"))

def create_temp_file(suffix: str = "", prefix: str = "tmp", dir: Optional[str] = None) -> str:
    """
    Create a temporary file and return its path.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        dir: Directory to create file in
        
    Returns:
        Path to temporary file
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=suffix, prefix=prefix, dir=dir, delete=False) as tmp:
        return tmp.name

def cleanup_temp_files(temp_dir: Union[str, Path]) -> bool:
    """
    Clean up temporary files and directories.
    
    Args:
        temp_dir: Temporary directory to clean
        
    Returns:
        True if successful, False otherwise
    """
    try:
        temp_path = Path(temp_dir)
        if temp_path.exists():
            if temp_path.is_file():
                temp_path.unlink()
            else:
                shutil.rmtree(temp_path)
        return True
    except Exception:
        return False
