from datetime import datetime


                                                                                                                                                              
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


def correct_markdown(self, content: str) -> str:
    """
    Corrects headers inside markdown links.
    """
    pattern = r"\[\s*\n*(#{1,6}.*?)\n*\]\((.*?)\)"
    def replacer(match):
        header = match.group(1)
        link = match.group(2)
        return header + "\n[Link](" + link + ")"
    content = re.sub(pattern, replacer, content)
    return content


def parse_tups(self, filepath: Path, errors: str = "ignore") -> List[Tuple[Optional[str], str]]:
    """
    Parses a file into tuples.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if self._remove_hyperlinks:
        content = self.remove_hyperlinks(content)
    if self._remove_images:
        content = self.remove_images(content)
    content = self.correct_markdown(content)
    markdown_tups = self.markdown_to_tups(content)
    return markdown_tups
