from datetime import datetime


def log_interaction(input_str, response_str=None, filename="interactions.txt"):
    # Open the file in append mode ('a')
    with open(filename, 'a') as f:

        if response_str is not None:
            f.write(f"{datetime.now()}:\n")
            f.write(f"Input: {input_str}\n")
            f.write(f"Response: {response_str}\n")
        
        else:
            f.write(f"--------------------------------\n")
            f.write(f"INFO: {input_str}\n")
            f.write("-------------------------------\n")

        # Write a new line for separating this interaction from the next one
        f.write("\n")


def read_str_prompt(filepath: str):

    with open(filepath, 'r') as file:
            template = file.read()

    return(template) 

#************* - A casser dans class MarkdownReader

import re
def correct_markdown(self, content: str) -> str:
        """Correct headers inside markdown links."""
        pattern = r"\[\s*\n*(#{1,6}.*?)\n*\]\((.*?)\)"
        # Split the header and the link, then reformulate
        def replacer(match):
            header = match.group(1)
            link = match.group(2)
            return header + "\n[Link](" + link + ")"
        content = re.sub(pattern, replacer, content)
        return content

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
def parse_tups(
    self, filepath: Path, errors: str = "ignore"
) -> List[Tuple[Optional[str], str]]:
    """Parse file into tuples."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if self._remove_hyperlinks:
        content = self.remove_hyperlinks(content)
    if self._remove_images:
        content = self.remove_images(content)
    content = self. correct_markdown(content)
    markdown_tups = self.markdown_to_tups(content)
    return markdown_tups
