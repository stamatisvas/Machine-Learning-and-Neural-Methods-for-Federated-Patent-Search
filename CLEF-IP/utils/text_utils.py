"""
Text processing utilities for CLEF-IP data preparation.
"""

import re
from typing import List, Set
import xml.etree.ElementTree as ET


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and normalizing whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ''
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_text_aggressive(text: str) -> str:
    """
    Aggressively clean text by removing most special characters.
    
    Removes numbers, punctuation, and special characters.
    Used in some experimental variants.
    
    Args:
        text: Input text
        
    Returns:
        Aggressively cleaned text
    """
    if not text:
        return ''
    
    text = text.lower()
    remove = [
        "\n", "\t", "'", "-", ".", "!", "@", "#", "$", "%", "&", "*", "=", "+",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "(", ")", "{", "}", "[", "]", ",", '"', "?", "<", ">",
        ";", "¬", "°", ":"
    ]
    
    for char in remove:
        text = text.replace(char, ' ')
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_ipc_codes(root: ET.Element) -> Set[str]:
    """
    Extract IPC codes from a patent document.
    
    Extracts IPC codes from classifications-ipcr elements.
    
    Args:
        root: Root element of patent document
        
    Returns:
        Set of IPC codes (at section level)
    """
    ipc_codes = set()
    
    # Find bibliographic-data element
    bib_data = None
    for element in root:
        if element.tag == "bibliographic-data":
            bib_data = element
            break
    
    if bib_data is None:
        return ipc_codes
    
    # Find technical-data element
    for element in bib_data:
        if element.tag == "technical-data":
            # Find classifications-ipcr
            for subelement in element:
                if subelement.tag == "classifications-ipcr":
                    for ipc_code in subelement:
                        if ipc_code.text:
                            # Extract section level (first character)
                            code_parts = ipc_code.text.split()
                            if code_parts:
                                ipc_codes.add(code_parts[0])
    
    return ipc_codes


def extract_ipc_codes_level3(root: ET.Element) -> Set[str]:
    """
    Extract IPC codes at level 3 (subclass level).
    
    Args:
        root: Root element of patent document
        
    Returns:
        Set of IPC codes at subclass level (e.g., 'A01B')
    """
    ipc_codes = set()
    
    # Find bibliographic-data element
    bib_data = None
    for element in root:
        if element.tag == "bibliographic-data":
            bib_data = element
            break
    
    if bib_data is None:
        return ipc_codes
    
    # Find technical-data element
    for element in bib_data:
        if element.tag == "technical-data":
            # Find classifications-ipcr
            for subelement in element:
                if subelement.tag == "classifications-ipcr":
                    for ipc_code in subelement:
                        if ipc_code.text:
                            code_parts = ipc_code.text.split()
                            if code_parts:
                                # Extract up to subclass level (first 4 characters: section + class + subclass)
                                code = code_parts[0]
                                if len(code) >= 4:
                                    ipc_codes.add(code[:4])
    
    return ipc_codes
