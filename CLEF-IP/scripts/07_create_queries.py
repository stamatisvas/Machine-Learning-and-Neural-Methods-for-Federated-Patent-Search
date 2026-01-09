"""
Step 7: Create queries from CLEF-IP topics.

This script extracts queries from CLEF-IP topic files and creates
a queries file in the format required for experiments.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from googletrans import Translator
from utils.text_utils import clean_text
import config

translator = Translator()


def create_queries():
    """Create queries from CLEF-IP topic files."""
    queries_file = config.QUERIES_FILE
    count = 0
    
    with open(queries_file, 'w', encoding='utf-8') as writer:
        for file in os.listdir(config.TOPICS_PATH):
            try:
                topic_path = os.path.join(config.TOPICS_PATH, file)
                patent = ET.parse(topic_path)
                root = patent.getroot()
                
                topic_id = root.attrib["ucid"]
                lang = root.attrib["lang"]
                
                if lang != 'EN':
                    continue
                
                writer.write(f"{topic_id}<id_sep>")
                
                # Extract IPC codes
                ipc_list = []
                bib_data = None
                for element in root:
                    if element.tag == 'bibliographic-data':
                        bib_data = element
                        break
                
                if bib_data:
                    for element in bib_data:
                        if element.tag == "technical-data":
                            for subelement in element:
                                if subelement.tag == "classifications-ipcr":
                                    for ipc_code in subelement:
                                        if ipc_code.text:
                                            code_parts = ipc_code.text.split()
                                            if code_parts:
                                                code = code_parts[0]
                                                # Create IPC tokens
                                                ipc_list.append(f'section{code[0]}section')
                                                ipc_list.append(f'class{code[:3]}class')
                                                ipc_list.append(f'subclass{code}subclass')
                                                if len(code_parts) > 1:
                                                    group = code_parts[1].replace('/', 'slash')
                                                    if group.endswith('/00'):
                                                        ipc_list.append(f'group{code}{group}group')
                                                    else:
                                                        ipc_list.append(f'subgroup{code}{group}subgroup')
                
                # Write IPC codes
                for ipc in set(ipc_list):
                    writer.write(f"{ipc} ")
                
                # Extract and write title
                if bib_data:
                    for element in bib_data:
                        if element.tag == "technical-data":
                            for subelement in element:
                                if subelement.tag == "invention-title":
                                    if subelement.get('lang') == 'EN' and subelement.text:
                                        writer.write(f"{clean_text(subelement.text)} ")
                                        break
                
                # Extract and write abstract
                abstr = None
                for element in root:
                    if element.tag == 'abstract':
                        abstr = element
                        break
                
                if abstr:
                    for p in abstr:
                        if p.tag == 'p' and p.text:
                            text = clean_text(p.text)
                            if abstr.get('lang') != 'EN':
                                try:
                                    text = translator.translate(text).text
                                except:
                                    pass
                            writer.write(f"{text} ")
                
                # Extract and write description (first 500 words)
                desc = None
                for element in root:
                    if element.tag == 'description':
                        desc = element
                        break
                
                if desc:
                    break_counter = 0
                    for p in desc:
                        if p.tag == 'p' and p.text:
                            text = clean_text(p.text)
                            words = text.split()[:config.DESCRIPTION_MAX_WORDS]
                            length = len(words)
                            break_counter += length
                            text = ' '.join(words)
                            
                            if desc.get('lang') != 'EN':
                                try:
                                    text = translator.translate(text).text
                                except:
                                    pass
                            writer.write(f"{text} ")
                            
                            if break_counter >= config.DESCRIPTION_MAX_WORDS:
                                break
                        else:
                            try:
                                for pre in p:
                                    if pre.text:
                                        text = clean_text(pre.text)
                                        words = text.split()[:config.DESCRIPTION_MAX_WORDS]
                                        length = len(words)
                                        break_counter += length
                                        text = ' '.join(words)
                                        
                                        if desc.get('lang') != 'EN':
                                            try:
                                                text = translator.translate(text).text
                                            except:
                                                pass
                                        writer.write(f"{text} ")
                                        
                                        if break_counter >= config.DESCRIPTION_MAX_WORDS:
                                            break
                            except:
                                pass
                
                # Extract and write claims
                clm = None
                for element in root:
                    if element.tag == 'claims':
                        clm = element
                        break
                
                if clm:
                    for cl in clm:
                        for claim_text in cl:
                            if claim_text.tag == 'claim-text' and claim_text.text:
                                text = clean_text(claim_text.text)
                                if clm.get('lang') != 'EN':
                                    try:
                                        text = translator.translate(text, dest='en').text
                                    except:
                                        pass
                                writer.write(f"{text} ")
                
                count += 1
                if count >= config.NUM_QUERIES:
                    break
                
                writer.write("<seperator>")
                
            except Exception as e:
                print(f"Error processing topic {file}: {e}")
                continue
    
    print(f"Created {count} queries in: {queries_file}")


if __name__ == '__main__':
    create_queries()
