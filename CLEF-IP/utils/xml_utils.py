"""
XML processing utilities for CLEF-IP data preparation.
"""

from xml.etree import ElementTree as ET
from typing import List


class hashabledict(dict):
    """Hashable dictionary for use as dictionary keys."""
    
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class XMLCombiner:
    """
    Combines multiple XML documents into a single document.
    
    Used to merge different patent document kinds (e.g., A1, B2) into
    a single patent document.
    """
    
    def __init__(self, filenames: List[str]):
        """
        Initialize XML combiner.
        
        Args:
            filenames: List of XML file paths to combine
        """
        assert len(filenames) > 0, 'No filenames!'
        self.roots = [ET.parse(f).getroot() for f in filenames]
    
    def combine(self) -> ET.ElementTree:
        """
        Combine all XML documents.
        
        Returns:
            Combined ElementTree
        """
        for r in self.roots[1:]:
            self.combine_element(self.roots[0], r)
        return ET.ElementTree(self.roots[0])
    
    def combine_element(self, one: ET.Element, other: ET.Element):
        """
        Recursively combine XML elements.
        
        Updates text or children of an element if found in `one`,
        or adds it from `other` if not found.
        
        Args:
            one: First element
            other: Second element to combine with first
        """
        mapping = {(el.tag, hashabledict(el.attrib)): el for el in one}
        for el in other:
            if len(el) == 0:
                # Leaf element
                try:
                    mapping[(el.tag, hashabledict(el.attrib))].text = el.text
                except KeyError:
                    mapping[(el.tag, hashabledict(el.attrib))] = el
                    one.append(el)
            else:
                # Nested element
                try:
                    self.combine_element(
                        mapping[(el.tag, hashabledict(el.attrib))], el
                    )
                except KeyError:
                    mapping[(el.tag, hashabledict(el.attrib))] = el
                    one.append(el)
