from datetime import datetime
import re

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

def log_interaction(input_str, response_str=None):                                                                                                                   
    """                                                                                                                                                              
    Logs interactions between the user and the system.                                                                                                               
    If a response string is provided, it logs the date, input, and response.                                                                                         
    If no response string is provided, it logs the input string as info.                                                                                             
    """                                                                                                                                                              
    if response_str is not None:                                                                                                                                     
        return f"{datetime.now()}:\nInput: {input_str}\nResponse: {response_str}\n"                                                                                  
    else:                                                                                                                                                            
        return f"--------------------------------\nINFO: {input_str}\n-------------------------------\n"     


def read_str_prompt(filepath: str):
    """
    Reads a string prompt from a file.
    """
    with open(filepath, 'r') as file:
        template = file.read()
    return template

#************* - A casser dans class MarkdownReader